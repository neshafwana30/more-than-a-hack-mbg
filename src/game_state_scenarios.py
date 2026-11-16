"""Synthetic game-state contracts using production evidence/fusion/decoders only."""
from dataclasses import dataclass
from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import pandas as pd


@dataclass
class GameStateScenarios:
    frames: pd.DataFrame
    summary: pd.DataFrame
    metadata: dict


def scenario_inputs(name,n=320,fps=25,pitch=(105.3,68.0)):
    ids=np.arange(n);speed=8.;players=3.
    if name in ('placed_stoppage','quiet_live'):speed=0.
    if name=='placed_stoppage':players=.3
    if name=='fast_ball_slow_players':players=0.
    travel=ids*speed/fps/pitch[0]
    bx=.15+(.65-np.abs((travel%1.30)-.65));by=np.full(n,.5)
    ph=np.full(n,players);pa=ph.copy();valid=np.ones(n,bool)
    if name in ('missing_live','missing_all'):
        bx[:]=np.nan;by[:]=np.nan
        if name=='missing_all':ph[:]=np.nan;pa[:]=np.nan;valid[:]=False
    if name=='outside_missing':
        bx[80:100]=.5;by[80:100]=1.02;bx[100:200]=np.nan;by[100:200]=np.nan
    if name=='outside_outlier':bx[80]=.5;by[80]=1.02
    if name=='restart':
        bx[:150]=.4;by[:150]=.5;ph[:150]=.3;pa[:150]=.3
        bx[150:]=.4+np.arange(n-150)*8/fps/pitch[0]
    seen=np.isfinite(bx)&np.isfinite(by)
    v=np.r_[np.nan,np.hypot(np.diff(bx)*pitch[0],np.diff(by)*pitch[1])*fps]
    f=pd.DataFrame({'frame':ids,'half':1,'graded':True,'ball_ok':seen,'ball_x':bx,'ball_y':by,
        'ball_speed2d_mps':v,'ball_missing_run_frames':np.where(seen,0,n),
        'player_motion_robust_h':ph,'player_motion_robust_a':pa,'centroid_valid_h':valid,'centroid_valid_a':valid,
        'valid_h_fraction':valid.astype(float),'valid_a_fraction':valid.astype(float),
        'nearest_h_m':np.where(seen,.6,np.nan),'nearest_a_m':np.where(seen,6.,np.nan),
        'nearest_team_proxy':np.where(seen,'H','?'),'tag_h_support':0.,'tag_a_support':0.})
    events=pd.DataFrame({'frame':ids,'stoppage_anchor':0.,'restart_anchor':0.,'out_of_pitch_hint':0.,'is_set_piece_context':False})
    return f,events


def game_state_scenarios(settings,temporal_config=None,fps=25,pitch=(105.3,68.0)):
    from .live_dead import estimate_live_dead
    from .observation_quality import analyze_observations
    from .team_control import estimate_team_control
    from .evidence_fusion import fuse_evidence
    from .temporal_decoder import decode_states
    names=['placed_stoppage','running_live','quiet_live','fast_ball_slow_players','missing_live','missing_all','outside_outlier','outside_missing','restart']
    all_rows=[];summaries=[]
    for name in names:
        f,e=scenario_inputs(name,fps=fps,pitch=pitch);q=analyze_observations(f);cfg={'model':'signed','signed':deepcopy(settings)}
        ld=estimate_live_dead(f,q.frames,e,cfg,fps,pitch)
        control=estimate_team_control(f,q.frames)
        fusion=fuse_evidence(f,q,control,ld,SimpleNamespace(frames=e),{'control_normalization':'max'})
        methods={}
        for method in ('joint','hierarchical'):
            tc={**(temporal_config or {}),'method':method}
            methods[method]=decode_states(fusion.state_log_scores,f.half.to_numpy(),fusion.transition_hint,fusion.stop_hint,fusion.restart_hint,tc)
        checks=[]
        for method,r in methods.items():
            if name=='placed_stoppage':ok=bool(np.all(r.labels[80:240]=='D'))
            elif name=='restart':ok=bool(np.all(r.labels[70:120]=='D') and np.all(r.labels[230:290]!='D'))
            elif name=='outside_missing':ok=bool(ld.components.outside_confirmed.iloc[90] and np.any(r.labels[85:99]=='D') and np.all(r.labels[230:290]!='D'))
            else:ok=bool(np.all(r.labels[80:240]!='D'))
            checks.append(ok)
        rows=ld.components.copy();rows['joint_label']=methods['joint'].labels;rows['hierarchical_label']=methods['hierarchical'].labels
        rows.insert(0,'scenario',name);all_rows.append(rows)
        summaries.append({'scenario':name,'passed':all(checks),'joint_passed':checks[0],'hierarchical_passed':checks[1],
            'joint_D_frames':int((methods['joint'].labels=='D').sum()),'hierarchical_D_frames':int((methods['hierarchical'].labels=='D').sum()),
            'carry_frames':int(ld.components.outside_carry_support.gt(0).sum())})
    return GameStateScenarios(pd.concat(all_rows,ignore_index=True),pd.DataFrame(summaries),{
        'source':'synthetic observations only; no provided-match labels or public GT',
        'scope':'behavioral contracts, not match accuracy; both decoders use the same existing transition costs',
        'limitation':'quiet live and stoppage with identical observed cues cannot be guaranteed separable; running quiet-live case has active player motion',
        'carry':'support can decay before label expiry; no forced label hold until timeout'})
