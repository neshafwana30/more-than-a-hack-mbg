"""Synthetic tests only; no labels/data from the competition match are used."""
from pathlib import Path

import numpy as np
import pytest
import yaml

from main import ROOT, run
from src.data_loader import load_tracking
from src.ball_recovery import recover_ball
from src.tactical_features import tactical_evidence
from src.submission import write_submission, validate_csv
from src.temporal_decoder import viterbi, merge_short_runs
from experiments.external_evaluation import compute_metrics


def make_synthetic_data(tmp_path: Path, n: int = 160):
    folder = tmp_path / "tactical"
    folder.mkdir()
    meta = folder / "BAR-ATH_tactical_atd.xml"
    raw = folder / "BAR-ATH_tactical_atd.txt"
    smart = folder / "BAR-ATH_tactical_smart_tagging.xml"
    # 12 slots, 6 per team, enough for the activity detector.
    players = ''.join(f'<Player id="{i}" teamId="{("h" if i<=6 else "a")}"/>'
                      for i in range(1,13))
    meta.write_text('<main><Teams><Team id="h"><Name>Home</Name></Team>'
                    '<Team id="a"><Name>Away</Name></Team></Teams>'
                    f'<Players>{players}</Players></main>', encoding='utf-8')
    lines = []
    for frame in range(n):
        home_pos, away_pos = ((0.2, 0.3), (0.8, 0.7))
        ball = home_pos if frame < 80 else away_pos
        groups = ([f'{home_pos[0]},{home_pos[1]},0']*6
                  + [f'{away_pos[0]},{away_pos[1]},0']*6)
        if 30 <= frame < 36:  # plausible 6-frame missing-ball gap
            ball_group = 'NaN,NaN,NaN'
        else:
            ball_group = f'{ball[0]},{ball[1]},1'
        lines.append(f'{frame+1}:' + ';'.join(groups) + ':' + ball_group)
    raw.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    smart.write_text('<file><ALL_INSTANCES>'
                     '<instance><start>1.0</start><end>2.5</end><code>BUILD UP</code>'
                     '<label><group>Team</group><text>Barcelona</text></label>'
                     '</instance></ALL_INSTANCES></file>', encoding='utf-8')
    return meta, raw, smart


def test_integrated_prediction_and_submission(tmp_path):
    make_synthetic_data(tmp_path)
    # The earlier evidence-fusion decoders; config.yaml now runs controlled_possession,
    # which has its own integrated test in tests/test_possession_v2.py.
    config = yaml.safe_load((ROOT / 'config_confidence.yaml').read_text())
    config['match']['frames'] = 160
    config['activity']['min_active_block_frames'] = 40
    labels, segments = run(config, tmp_path)
    assert segments == [(0,160)]
    assert labels[20] == 'H'
    assert labels[140] == 'A'
    output = write_submission(labels, tmp_path / 'team.csv', expected_rows=160)
    summary = validate_csv(output, expected_rows=160)
    assert sum(summary.values()) == 160
    assert output.read_text().splitlines()[0] == 'frame,label'
    assert output.read_text().splitlines()[-1].startswith('159,')


def test_missing_recovery_is_conservative():
    ball = np.full((25,2), np.nan, dtype=np.float32)
    ball[0] = (0.25,0.5)
    ball[4] = (0.26,0.5)
    ball[20] = (0.95,0.5)
    est = recover_ball(ball, 25, (105,68), max_gap_frames=5)
    assert np.isfinite(est.xy[1:4]).all()
    assert np.isnan(est.xy[8:15]).all()
    assert est.confidence[0] == 1.0
    assert est.confidence[2] < 1.0


def test_smart_tagging_uses_exact_team_map_and_soft_event_edges():
    events = [
        {"code": "BUILD UP", "start": 10, "end": 30,
         "tags": {"team": "  BARCELONA  "}},
        {"code": "BUILD UP", "start": 30, "end": 40,
         "tags": {"team": "Barcelona B"}},
        {"code": "SET PIECES", "start": 50, "end": 70,
         "tags": {"team": "N/A"}},
    ]
    evidence = tactical_evidence(events, 80, {
        "team_map": {"Barcelona": "H", "Atletic Club": "A"},
        "event_edge_taper_frames": 4,
        "event_weights": {"BUILD UP": 1.0},
    })

    assert evidence.log_odds_home[10] < evidence.log_odds_home[13]
    assert evidence.log_odds_home[13] == pytest.approx(1.0)
    assert np.all(evidence.log_odds_home[30:40] == 0.0)
    assert evidence.set_piece[50] < evidence.set_piece[59]
    assert evidence.set_piece[59] == pytest.approx(1.0)
    assert evidence.set_piece[70] == 0.0


def test_incomplete_tracking_rejected(tmp_path):
    meta, raw, _ = make_synthetic_data(tmp_path, n=12)
    with pytest.raises(ValueError, match='Incomplete ATD'):
        load_tracking(meta, raw, 199319, 25, (105.3,68.0))


def test_temporal_smoothing():
    p = np.full((80,3), [0.9, 0.08, 0.02], dtype=np.float32)
    p[36] = [0.03,0.95,0.02] # one-frame false turnover
    decoded = viterbi(p, {'home_away_switch_penalty':5.0,'play_dead_switch_penalty':3.0})
    assert np.all(decoded == 0)
    assert list(merge_short_runs(np.array([0,0,1,0,0], dtype=np.int8),2)) == [0]*5


def test_external_metrics():
    gt = np.array(list('HHHHHDDDAAAAAAXXXXXHHHH'), dtype='<U1')
    result = compute_metrics(gt,gt)
    assert result['macroF1'] == 1.0
    assert result['spellF1'] == 1.0
    assert result['reference_spells'] == result['predicted_spells'] == 3
