"""Descriptive diagnostics computed by the pipeline and consumed by notebooks."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
import games
from .features import _runs
from .submission import segments_for_labels


@dataclass
class FeatureDiagnostics:
    coverage: pd.DataFrame
    schema: pd.DataFrame
    missing_ball_gaps: pd.DataFrame
    gap_summary: dict
    tag_counts: pd.DataFrame


@dataclass
class CoreDiagnostics:
    score_summary: pd.DataFrame
    dead_score_summary: dict


@dataclass
class PredictionDiagnostics:
    segments: pd.DataFrame
    segment_summary: pd.DataFrame
    transition_counts: pd.DataFrame


def feature_diagnostics(bundle, fps=25):
    f = bundle.frames
    coverage = []
    for name, mask in [('all', np.ones(len(f), bool)), ('graded', f.graded),
                       ('half_1', f.half == 1), ('half_2', f.half == 2)]:
        g = f.loc[mask]
        coverage.append({'region': name, 'frames': len(g),
                         'ball_observed_fraction': g.ball_ok.mean(),
                         'mean_tracked_players': (g.home_count + g.away_count).mean()})
    rows = []
    for column in f:
        g = f.loc[f.graded, column]
        rows.append({'column': column, 'dtype': str(f[column].dtype),
                     'non_null_graded': g.notna().sum(), 'missing_fraction_graded': g.isna().mean(),
                     'unique_graded': g.nunique(dropna=True)})
    gaps = []
    for half in (1, 2):
        ids = np.flatnonzero(f.half.to_numpy() == half)
        if not len(ids):
            continue
        lo, hi = int(ids[0]), int(ids[-1])
        for start, end in _runs(~f.ball_ok.iloc[lo:hi+1].to_numpy(bool), lo):
            gaps.append({'half': half, 'start_frame': start, 'end_frame': end,
                         'n_frames': end-start+1, 'duration_s': (end-start+1)/fps})
    gaps = pd.DataFrame(gaps, columns=['half','start_frame','end_frame','n_frames','duration_s'])
    gap_summary = {'count': len(gaps), 'median_s': gaps.duration_s.median(),
                   'p95_s': gaps.duration_s.quantile(.95), 'max_s': gaps.duration_s.max()}
    tag_counts = (bundle.events.groupby(['code_clean', 'side'], dropna=False).size()
                  .rename('instances').reset_index())
    return FeatureDiagnostics(pd.DataFrame(coverage), pd.DataFrame(rows), gaps, gap_summary, tag_counts)


def core_diagnostics(frames, fusion, config):
    graded = frames.graded.to_numpy(bool)
    kickoff = config.get('fusion', {}).get('second_half_kickoff_frame', 113742)
    second_start = next(start for half, start, end in games.half_video_blocks(config.get('game', games.GAME)) if half == 2)
    forced = ((frames.half.to_numpy() == 2) & (frames.frame.to_numpy() >= second_start)
              & (frames.frame.to_numpy() < kickoff))
    mask = graded & ~forced
    scores = fusion.state_log_scores
    stats = []
    for i, label in enumerate(('H', 'A', 'D')):
        s = pd.Series(scores[mask, i])
        stats.append({'state': label, 'frames': len(s), 'min': s.min(), 'median': s.median(),
                      'p95': s.quantile(.95), 'max': s.max()})
    margin = pd.Series(scores[mask, 2] - scores[mask, :2].max(axis=1))
    summary = {'unforced_graded_frames': int(mask.sum()), 'forced_graded_frames': int((graded & forced).sum()),
               'dead_wins_unforced': int((margin > 0).sum()), 'dead_ties_unforced': int((margin == 0).sum()),
               'dead_advantage_median': margin.median(), 'dead_advantage_max': margin.max(),
               'score_semantics': 'uncalibrated additive utilities'}
    return CoreDiagnostics(pd.DataFrame(stats), summary)


def prediction_diagnostics(labels, half, fps=25):
    segments = segments_for_labels(labels, half, fps)
    summaries = []
    for label, group in segments.groupby('label'):
        summaries.append({'label': label, 'segments': len(group), 'mean_duration_s': group.duration_s.mean(),
                          'median_duration_s': group.duration_s.median(), 'p95_duration_s': group.duration_s.quantile(.95),
                          'segments_lt_1s': int((group.duration_s < 1).sum()),
                          'segments_lt_2s': int((group.duration_s < 2).sum())})
    labels = np.asarray(labels); half = np.asarray(half)
    changes = (labels[1:] != labels[:-1]) & (half[1:] == half[:-1]) & (half[1:] != 0)
    transitions = pd.DataFrame({'from': labels[:-1][changes], 'to': labels[1:][changes]})
    counts = transitions.groupby(['from', 'to']).size().rename('count').reset_index()
    return PredictionDiagnostics(segments, pd.DataFrame(summaries, columns=[
        'label','segments','mean_duration_s','median_duration_s','p95_duration_s','segments_lt_1s','segments_lt_2s']), counts)
