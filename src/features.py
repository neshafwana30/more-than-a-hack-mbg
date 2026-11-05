"""Reproducible tactical-frame features for BAR-ATH.

Uses games.asset_paths()/require(). No ground truth or human-authored game labels.
Run from project root or import from notebooks/; output never includes copyrighted assets.
Requires numpy, pandas, kloppy. The complete source data must exist locally.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import games


@dataclass
class FeatureResult:
    frames: pd.DataFrame
    events: pd.DataFrame
    slots: pd.DataFrame
    player_positions: dict | None = None  # ephemeral arrays; never write restricted tracking to repo


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


def _robust_player_motion(x, y, valid, side_mask, half, pitch, fps, max_speed=12.0, min_tracks=4):
    """Per-frame median plausible same-slot motion; no centroid identity artifacts."""
    n = len(half)
    xx, yy, vv = x[:, side_mask], y[:, side_mask], valid[:, side_mask]
    delta = np.hypot(np.diff(xx, axis=0)*pitch[0], np.diff(yy, axis=0)*pitch[1])*fps
    ok = vv[1:] & vv[:-1] & np.isfinite(delta) & (delta <= max_speed)
    ok &= ((half[1:] == half[:-1]) & (half[1:] != 0))[:,None]
    counts = ok.sum(axis=1)
    masked = np.where(ok, delta, np.nan)
    # Explicitly avoid nanmean/nanmedian on all-NaN rows.
    result = np.full(n, np.nan, dtype=float)
    qualifying = np.flatnonzero(counts >= min_tracks)
    if len(qualifying):
        result[qualifying+1] = np.nanmedian(masked[qualifying], axis=1)
    return result, np.r_[False, counts >= min_tracks]


def _ball_geometry(bx, by, px, py, valid, sides, half, pitch, fps, max_ball_speed=45.0):
    n = len(bx)
    ball_ok = np.isfinite(bx) & np.isfinite(by)
    result = {}
    nearest_pos = {}
    for side, prefix in [('H','h'), ('A','a')]:
        mask = sides == side
        d = np.hypot((px[:,mask]-bx[:,None])*pitch[0], (py[:,mask]-by[:,None])*pitch[1])
        d[~(valid[:,mask] & ball_ok[:,None])] = np.inf
        arg = np.argmin(d, axis=1)
        close = d[np.arange(n),arg]
        exists = np.isfinite(close)
        result[f'nearest_{prefix}_m'] = np.where(exists, close, np.nan)
        xx = px[:,mask][np.arange(n),arg]
        yy = py[:,mask][np.arange(n),arg]
        nearest_pos[side] = (np.where(exists,xx,np.nan),np.where(exists,yy,np.nan))
    movement = np.full(n,np.nan)
    continuous = (half[1:]==half[:-1]) & (half[1:]!=0) & ball_ok[1:] & ball_ok[:-1]
    change = np.hypot(np.diff(bx)*pitch[0],np.diff(by)*pitch[1])
    movement[1:] = np.where(continuous,change,np.nan)
    speed = movement*fps
    quality = np.isfinite(speed) & (speed<=max_ball_speed)
    return result, nearest_pos, movement, speed, quality


def _tag_support(events, n):
    """One capped evidence per ATTACKING family, not independent correlated tag counts."""
    families = {
        'BUILD UP':'progress', 'PROGRESSION':'progress',
        'BALL IN FINAL THIRD':'territory', 'BALL IN THE BOX':'territory',
        'PLAYERS IN THE BOX':'territory', 'CREATING CHANCES':'chance',
        'LONG BALL':'distribution', 'ATTACKING TRANSITION':'transition',
    }
    outputs={}
    for side in ('H','A'):
        cumulative=np.zeros(n,dtype=float)
        for family in sorted(set(families.values())):
            subset=events[(events.side==side)&(events.code_clean.isin([c for c,f in families.items() if f==family]))]
            cumulative+=(_interval_counts(subset,n)>0).astype(float)
        outputs[f'tag_{side.lower()}_support']=np.minimum(cumulative,2.0)/2.0
    return outputs


def _tag_boundaries(events, n):
    pieces=events[events.code_clean=='SET PIECES'].sort_values(['start_frame','end_frame'])
    positions=[]
    kinds=np.full(n,'unknown',dtype=object)
    sides=np.full(n,'?',dtype='<U1')
    for ev in pieces.itertuples(index=False):
        start=max(0,min(n,int(ev.start_frame)))
        end=max(0,min(n,int(ev.end_frame)))
        if end<=start: continue
        positions.extend((start,end))
        # 'Type' may not exist in particular parser output; getattr safely.
        kind=str(getattr(ev,'Type','unknown') or 'unknown')
        kinds[start:end]=kind
        sides[start:end]=str(ev.side)
    if not positions:
        return kinds,sides,np.full(n,np.nan)
    arr=np.unique(np.asarray(positions,dtype=np.int64))
    ix=np.searchsorted(arr,np.arange(n))
    distleft=np.where(ix>0,np.arange(n)-arr[np.clip(ix-1,0,len(arr)-1)],np.inf)
    distright=np.where(ix<len(arr),arr[np.clip(ix,0,len(arr)-1)]-np.arange(n),np.inf)
    return kinds,sides,np.minimum(distleft,distright).astype(float)


def refresh_visual_ball(features, player_positions, observations, pitch=(105.3,68.0), fps=25, max_ball_speed=45.0):
    """Insert only QA-approved local video observations at missing-ball frames.

    observations: dataframe with frame, x, y, quality. Does not mutate input.
    Recomputes geometry, missing runs and ball motion. Sourced-video flags preserved.
    """
    df=features.copy()
    if observations is None or len(observations)==0: return df
    required={'frame','x','y','quality'}
    if not required.issubset(observations.columns):
        raise ValueError(f'Video observations need {required}')
    for row in observations.itertuples(index=False):
        i=int(row.frame)
        if i<0 or i>=len(df) or bool(df.at[i,'ball_ok']) or not (0<=row.x<=1 and 0<=row.y<=1):
            continue
        if float(row.quality)<0.5 or df.at[i,'half']==0: continue
        df.at[i,'ball_x']=float(row.x); df.at[i,'ball_y']=float(row.y)
        df.at[i,'ball_ok']=True
        df.at[i,'video_recovered_ball']=True
    bx=df.ball_x.to_numpy(float); by=df.ball_y.to_numpy(float)
    half=df.half.to_numpy()
    sides=player_positions['sides']
    distances,coords,movement,speed,speed_ok=_ball_geometry(bx,by,player_positions['x'],player_positions['y'],player_positions['valid'],sides,half,pitch,fps,max_ball_speed)
    for name,v in distances.items(): df[name]=v
    for side,prefix in [('H','h'),('A','a')]:
        df[f'nearest_{prefix}_x'],df[f'nearest_{prefix}_y']=coords[side]
    dh=df.nearest_h_m.to_numpy(float); da=df.nearest_a_m.to_numpy(float)
    df['distance_margin_a_minus_h_m']=da-dh;df['abs_distance_margin_m']=np.abs(da-dh)
    proxy=np.full(len(df),'?',dtype='<U1'); both=np.isfinite(dh)&np.isfinite(da)
    proxy[both&(dh<da)]='H';proxy[both&(da<dh)]='A'
    df['nearest_team_proxy']=proxy
    df['ball_displacement_m']=movement;df['ball_speed2d_mps']=speed;df['ball_speed_quality']=speed_ok
    missing,since,until=_run_features(df.ball_ok.to_numpy(bool),half)
    df['ball_missing_run_frames']=missing;df['frames_since_ball_seen']=since;df['frames_until_ball_seen']=until
    df['ball_visibility_age_s']=since/fps;df['ball_next_visibility_s']=until/fps
    return df

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

    geometry, nearest_pos, movement, speed, speed_quality = _ball_geometry(
        bx,by,px,py,valid,sides,half,pitch,fps)
    dh,da=geometry['nearest_h_m'],geometry['nearest_a_m']
    proxy=np.full(n,'?',dtype='<U1')
    both=np.isfinite(dh)&np.isfinite(da)
    proxy[both&(dh<da)]='H';proxy[both&(da<dh)]='A'
    missing_run,since,until=_run_features(ball_ok,half)
    motion_h,valid_h=_robust_player_motion(px,py,valid,sides=='H',half,pitch,fps)
    motion_a,valid_a=_robust_player_motion(px,py,valid,sides=='A',half,pitch,fps)

    result = pd.DataFrame({
        'frame':frame, 'half':half, 'graded':graded, 'ball_x':bx, 'ball_y':by, 'ball_ok':ball_ok,
        'ball_missing_run_frames':missing_run, 'frames_since_ball_seen':since, 'frames_until_ball_seen':until,
        'home_count':valid[:,sides=='H'].sum(axis=1), 'away_count':valid[:,sides=='A'].sum(axis=1),
        'nearest_h_m':dh, 'nearest_a_m':da, 'distance_margin_a_minus_h_m':da-dh,
        'abs_distance_margin_m':np.abs(da-dh), 'nearest_team_proxy':proxy,
        'ball_displacement_m': movement, 'ball_speed2d_mps':speed,
        'ball_speed_quality':speed_quality,
        'ball_visibility_age_s':since/fps, 'ball_next_visibility_s':until/fps,
        'valid_h_fraction':np.minimum(1.0,valid[:,sides=='H'].sum(axis=1)/11.0),
        'valid_a_fraction':np.minimum(1.0,valid[:,sides=='A'].sum(axis=1)/11.0),
        'player_motion_robust_h':motion_h, 'player_motion_robust_a':motion_a,
        'centroid_speed_H_mps':motion_h, 'centroid_speed_A_mps':motion_a,
        'centroid_valid_h':valid_h, 'centroid_valid_a':valid_a,
        'nearest_h_x':nearest_pos['H'][0], 'nearest_h_y':nearest_pos['H'][1],
        'nearest_a_x':nearest_pos['A'][0], 'nearest_a_y':nearest_pos['A'][1],
        'video_recovered_ball':np.zeros(n,dtype=bool),
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
    result['tag_set_piece_type'],result['tag_set_piece_team'],result['tag_boundary_distance_frames']=_tag_boundaries(good,n)
    for name, values in _tag_support(good,n).items():result[name]=values

    if len(result) != n or not np.array_equal(result.frame.to_numpy(), np.arange(n)):
        raise AssertionError('Invalid feature row count / timeline')
    slots_frame = pd.DataFrame({'slot':slots,'side':sides})
    return FeatureResult(frames=result, events=good, slots=slots_frame, player_positions={'x':px,'y':py,'valid':valid,'sides':sides})
