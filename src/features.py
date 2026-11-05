"""Reproducible tactical-frame features for BAR-ATH.

Uses games.asset_paths()/require(). No ground truth or human-authored game labels.
Run from project root or import from notebooks/; output never includes copyrighted assets.
Requires numpy, pandas, kloppy. The complete source data must exist locally.
"""
from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import games


@dataclass
class FeatureResult:
    frames: pd.DataFrame
    events: pd.DataFrame
    slots: pd.DataFrame


def _runs(mask: np.ndarray, offset: int = 0):
    """Inclusive [start, end] runs for true values, with duration in frames."""
    z = np.asarray(mask, dtype=bool)
    changes = np.diff(np.r_[False, z, False].astype(np.int8))
    begins = np.flatnonzero(changes == 1) + offset
    ends = np.flatnonzero(changes == -1) - 1 + offset
    return zip(begins, ends)


def _run_features(present: np.ndarray, half: np.ndarray):
    """Missing-run length and distances to observed ball, strictly within each half."""
    n = len(present)
    length = np.zeros(n, dtype=np.int32)
    since = np.full(n, np.nan)
    until = np.full(n, np.nan)
    for h in (1, 2):
        inds = np.flatnonzero(half == h)
        if not len(inds):
            continue
        lo, hi = int(inds[0]), int(inds[-1])
        for s, e in _runs(~present[lo:hi+1], lo):
            length[s:e+1] = e - s + 1
        last = None
        for i in range(lo, hi+1):
            if present[i]:
                last = i
            if last is not None:
                since[i] = i-last
        nxt = None
        for i in range(hi, lo-1, -1):
            if present[i]:
                nxt = i
            if nxt is not None:
                until[i] = nxt-i
    return length, since, until


def _team_for_slots(dataset, slots):
    mapping = {}
    for team in (getattr(dataset.metadata, 'teams', None) or []):
        name = str(getattr(team, 'name', '') or '').lower()
        side = 'H' if 'barcelona' in name else ('A' if 'athletic' in name or 'atletic' in name else None)
        if not side:
            continue
        for player in (getattr(team, 'players', None) or []):
            for field in ('player_id', 'name'):
                raw = getattr(player, field, None)
                match = re.fullmatch(r'(?:track\s*)?(\d+)', str(raw or '').strip(), re.I)
                if match and match.group(1) in slots:
                    slot = match.group(1)
                    if slot in mapping and mapping[slot] != side:
                        raise ValueError(f'Conflicting sides for track slot {slot}')
                    mapping[slot] = side
    if set(mapping) != set(slots) or set(mapping.values()) != {'H', 'A'}:
        raise ValueError(f'Incomplete team-slot map ({len(mapping)}/{len(slots)}). Check kloppy metadata; never guess sides.')
    return mapping


def _timedelta_seconds(values):
    if pd.api.types.is_numeric_dtype(values):
        return pd.to_numeric(values, errors='coerce').astype(float)
    return pd.to_timedelta(values, errors='coerce').dt.total_seconds()


def _interval_counts(events, n):
    diff = np.zeros(n+1, dtype=np.int32)
    for s, e in events[['start_frame', 'end_frame']].itertuples(index=False, name=None):
        lo = max(0, min(n, int(s)))
        hi = max(0, min(n, int(e)))
        if hi > lo:
            diff[lo] += 1
            diff[hi] -= 1
    return np.cumsum(diff[:-1])


def _centroid_speed(x, y, valid, side_mask, half, pitch, fps):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        cx = np.nanmean(np.where(valid[:, side_mask], x[:, side_mask], np.nan), axis=1)
        cy = np.nanmean(np.where(valid[:, side_mask], y[:, side_mask], np.nan), axis=1)
    ans = np.full(len(half), np.nan)
    continuity = (half[1:] == half[:-1]) & (half[1:] != 0)
    ok = continuity & np.isfinite(cx[1:]) & np.isfinite(cx[:-1]) & np.isfinite(cy[1:]) & np.isfinite(cy[:-1])
    delta = np.hypot(np.diff(cx) * pitch[0], np.diff(cy) * pitch[1]) * fps
    ans[1:] = np.where(ok, delta, np.nan)
    return ans


def build_features(game: str = games.GAME, data_dir: Optional[str] = None, source: str = 'tactical') -> FeatureResult:
    """Create one feature row per tactical video frame.

    Notes:
    * kloppy `frame_id` is sequential 1..N; raw IdFrame requires games.py per-half offsets.
    * Smart Tagging [start_frame, end_frame) is an analysis convention, NOT exact restart timing.
    * nearest_team_proxy is only geometry, never possession ground truth.
    * No public/private ground truth is read. Video is not needed for this baseline.
    """
    if source != 'tactical':
        raise ValueError('Submission timeline is tactical only; broadcast is deliberately unsupported.')
    try:
        from kloppy import metrica, sportscode
    except ImportError as exc:
        raise ImportError('Install dependencies: pip install numpy pandas kloppy') from exc

    needed = ('atd_meta', 'atd_raw', 'smart_tagging')
    games.require(needed, game=game, data_dir=data_dir, source=source)
    paths = games.asset_paths(game=game, data_dir=data_dir, source=source)
    dataset = metrica.load_tracking_epts(meta_data=paths['atd_meta'], raw_data=paths['atd_raw'])
    raw = dataset.to_df()
    n = games.source_frames(source=source, game=game)
    fps = games.GAMES[game]['fps']
    pitch = games.GAMES[game]['pitch']
    if len(raw) != n or 'frame_id' not in raw.columns or not np.array_equal(pd.to_numeric(raw.frame_id, errors='coerce').to_numpy(), np.arange(1, n+1)):
        raise ValueError('Unexpected kloppy timeline: require 199319 rows and sequential frame_id 1..N; do not silently realign.')

    frame = np.arange(n, dtype=np.int32)
    half = np.zeros(n, dtype=np.int8)
    for h, start, end in games.half_video_blocks(game):
        half[start:end+1] = h
    graded = half != 0

    slot_pat = re.compile(r'^(\d+)_(x|y|s)$')
    colmap = {}
    for col in raw.columns:
        m = slot_pat.fullmatch(str(col))
        if m:
            colmap.setdefault(m.group(1), {})[m.group(2)] = col
    slots = sorted((s for s, cols in colmap.items() if 'x' in cols and 'y' in cols), key=int)
    if not slots or not {'ball_x','ball_y'}.issubset(raw.columns):
        raise ValueError('ATD missing ball or player coordinates')
    mapping = _team_for_slots(dataset, slots)
    sides = np.array([mapping[s] for s in slots])
    px = raw[[colmap[s]['x'] for s in slots]].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=float)
    py = raw[[colmap[s]['y'] for s in slots]].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=float)
    valid = np.isfinite(px) & np.isfinite(py)
    bx = pd.to_numeric(raw['ball_x'], errors='coerce').to_numpy(dtype=float)
    by = pd.to_numeric(raw['ball_y'], errors='coerce').to_numpy(dtype=float)
    ball_ok = np.isfinite(bx) & np.isfinite(by)

    def nearest(side):
        mask = sides == side
        d = np.hypot((px[:, mask]-bx[:, None])*pitch[0], (py[:, mask]-by[:, None])*pitch[1])
        d[~(valid[:, mask] & ball_ok[:, None])] = np.nan
        result = np.full(n, np.nan)
        has_any = np.isfinite(d).any(axis=1)
        result[has_any] = np.nanmin(d[has_any], axis=1)
        return result

    dh, da = nearest('H'), nearest('A')
    proxy = np.full(n, '?', dtype='<U1')
    both = np.isfinite(dh) & np.isfinite(da)
    proxy[both & (dh < da)] = 'H'
    proxy[both & (da < dh)] = 'A'

    speed = np.full(n, np.nan)
    continuous = (half[1:] == half[:-1]) & (half[1:] != 0) & ball_ok[1:] & ball_ok[:-1]
    distance = np.hypot(np.diff(bx)*pitch[0], np.diff(by)*pitch[1])
    speed[1:] = np.where(continuous, distance*fps, np.nan)
    missing_run, since, until = _run_features(ball_ok, half)

    result = pd.DataFrame({
        'frame':frame, 'half':half, 'graded':graded, 'ball_x':bx, 'ball_y':by, 'ball_ok':ball_ok,
        'ball_missing_run_frames':missing_run, 'frames_since_ball_seen':since, 'frames_until_ball_seen':until,
        'home_count':valid[:,sides=='H'].sum(axis=1), 'away_count':valid[:,sides=='A'].sum(axis=1),
        'nearest_h_m':dh, 'nearest_a_m':da, 'distance_margin_a_minus_h_m':da-dh,
        'abs_distance_margin_m':np.abs(da-dh), 'nearest_team_proxy':proxy,
        'ball_speed2d_mps':speed,
        'centroid_speed_H_mps':_centroid_speed(px,py,valid,sides=='H',half,pitch,fps),
        'centroid_speed_A_mps':_centroid_speed(px,py,valid,sides=='A',half,pitch,fps),
    })

    raw_events = sportscode.load(paths['smart_tagging']).to_df().copy()
    if not {'timestamp','end_timestamp','code'}.issubset(raw_events.columns):
        raise ValueError('Unexpected Smart Tagging columns')
    events = raw_events.copy()
    events['start_s'] = _timedelta_seconds(events.timestamp)
    events['end_s'] = _timedelta_seconds(events.end_timestamp)
    events['start_frame'] = np.rint(events.start_s*fps).astype('Int64')
    events['end_frame'] = np.rint(events.end_s*fps).astype('Int64')
    events['code_clean'] = events.code.astype(str).str.strip().str.upper()
    team_name = events['Team'].fillna('').astype(str).str.lower() if 'Team' in events else pd.Series('', index=events.index)
    events['side'] = np.select([team_name.str.contains('barcelona'),team_name.str.contains('athletic|atletic')],['H','A'],default='?')
    good = events.dropna(subset=['start_frame','end_frame']).copy()
    good = good[good.end_frame > good.start_frame].copy()

    result['active_tag_count'] = _interval_counts(good, n)
    for side in ('H','A'):
        result[f'active_{side.lower()}_tags'] = _interval_counts(good[good.side==side], n)
    setpieces = good[good.code_clean=='SET PIECES']
    result['set_piece_tag'] = _interval_counts(setpieces, n) > 0
    start = np.zeros(n, dtype=bool)
    end = np.zeros(n, dtype=bool)
    for s, e in setpieces[['start_frame','end_frame']].itertuples(index=False,name=None):
        if 0 <= int(s) < n: start[int(s)] = True
        if 0 <= int(e) < n: end[int(e)] = True
    result['set_piece_start'] = start
    result['set_piece_end'] = end

    if len(result) != n or not np.array_equal(result.frame.to_numpy(), np.arange(n)):
        raise AssertionError('Invalid feature row count / timeline')
    slots_frame = pd.DataFrame({'slot':slots,'side':sides})
    return FeatureResult(frames=result, events=good, slots=slots_frame)
