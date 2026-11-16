"""Deterministic toy observations for local association and acquisition diagnostics."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from .player_acquisition_diagnostics import player_acquisition_diagnostics


@dataclass
class PlayerAcquisitionScenarios:
    frames: pd.DataFrame
    summary: pd.DataFrame
    events: pd.DataFrame
    metadata: dict


def scenario_observations(name,fps=25,pitch=(105.3,68.0)):
    n=120;ids=np.arange(n);fps=float(fps);pitch=np.asarray(pitch,float)
    ball_x=20.5+ids*2/fps
    ball_x[40:60]=ball_x[39]+np.arange(1,21)*12.5/fps
    arrival=ball_x[59]
    ball_x[60:]=arrival+np.arange(1,n-59)*2/fps
    receiver=arrival-.5
    ball_y=np.full(n,34.)
    x=np.tile([20.,receiver,50.,receiver+8.,55.],(n,1));y=np.tile([34.,34.,45.,44.,50.],(n,1))
    # Sender moves with the ball until release, receiver is stationary before reception.
    x[:40,0]=ball_x[:40]-.5
    x[40:,0]=x[39,0]
    x[60:,1]=ball_x[60:]-.5
    expected_side='H'
    if name in ('interception','short_control'):
        x[:,1]=receiver+8.;x[:,3]=receiver+.1;y[:,3]=34.
        x[60:,3]=ball_x[60:]-.4
        expected_side='A'
        if name=='short_control':ball_x[70:]=ball_x[69]+np.arange(1,n-69)*12.5/fps
    if name=='temporary_touch':ball_x[60:]=arrival+np.arange(1,n-59)*12.5/fps;x[:,1]=receiver
    if name=='contested_reception':
        x[:,3]=receiver+.1;y[:,3]=34.;x[60:,3]=ball_x[60:]-.4
        expected_side='both_ambiguous'
    if name=='ambiguous_duplicate':x[:,2]=x[:,1];y[:,2]=y[:,1]
    if name=='nearest_switch':
        ball_x=20.+ids*2/fps;x[:,0]=20.;x[:,1]=21.
    if name=='comoving_opponent':
        ball_x=30.+ids*2/fps;x[:,0]=ball_x-.6;x[:,1]=ball_x-10.
        x[:,3]=ball_x-.8;y[:,3]=34.
    valid=np.ones_like(x,bool)
    if name=='player_dropout':valid[54:67,1]=False
    if name=='slot_permutation':
        for i in range(n):
            if i%2:x[i,[0,1]]=x[i,[1,0]];y[i,[0,1]]=y[i,[1,0]]
    seen=np.ones(n,bool)
    if name=='occlusion':seen[54:67]=False
    bx=ball_x.copy();by=ball_y.copy();bx[~seen]=np.nan;by[~seen]=np.nan
    speed=np.r_[np.nan,np.hypot(np.diff(bx),np.diff(by))*fps]
    f=pd.DataFrame({'frame':ids,'half':np.ones(n,int),'graded':True,'ball_x':bx/pitch[0],
        'ball_y':by/pitch[1],'ball_ok':seen,'ball_speed2d_mps':speed,
        'ball_missing_run_frames':np.where(seen,0,int((~seen).sum())),
        'valid_h_fraction':1.,'valid_a_fraction':1.,'centroid_valid_h':True,'centroid_valid_a':True})
    positions={'x':x/pitch[0],'y':y/pitch[1],'valid':valid,'sides':np.array(['H','H','H','A','A'])}
    return f,positions,expected_side


def player_acquisition_scenarios(config,fps=25,pitch=(105.3,68.0)):
    names=['same_team_pass','interception','short_control','temporary_touch','nearest_switch','slot_permutation',
           'ambiguous_duplicate','occlusion','player_dropout','contested_reception','comoving_opponent']
    frames=[];events=[];summary=[];baseline=None
    for name in names:
        f,p,expected=scenario_observations(name,fps,pitch)
        r=player_acquisition_diagnostics(f,p,config,fps=fps,pitch=pitch)
        t=r.frames.copy();t.insert(0,'scenario',name);frames.append(t)
        e=r.events.copy();e.insert(0,'scenario',name);events.append(e)
        ch=int(t.candidate_h.sum());ca=int(t.candidate_a.sum());amb=int(t.candidate_status.eq('both_ambiguous').sum())
        if name in ('same_team_pass','slot_permutation'):passed=ch>0 and ca==0
        elif name in ('interception','short_control'):passed=ca>0 and ch==0
        elif name=='contested_reception':passed=amb>0
        else:passed=ch==0 and ca==0
        if name=='same_team_pass':baseline=t[['frame','candidate_h','candidate_a']].copy()
        invariant=True
        if name=='slot_permutation':
            invariant=bool(np.array_equal(baseline[['candidate_h','candidate_a']],t[['candidate_h','candidate_a']]))
            passed=passed and invariant
        if name=='ambiguous_duplicate':passed=passed and bool(t.association_status_h.eq('ambiguous').any())
        if name=='nearest_switch':passed=passed and bool(t.naive_high_associated_low_h.any())
        summary.append({'scenario':name,'passed':bool(passed),'expected_contract':expected if name in ('same_team_pass','interception','short_control','slot_permutation','contested_reception') else 'no_acquisition_candidate',
            'candidate_H_frames':ch,'candidate_A_frames':ca,'both_ambiguous_frames':amb,
            'nearest_switch_H_frames':int(t.nearest_candidate_switch_h.sum()),
            'column_changed_H_frames':int(t.observation_column_changed_h.sum()),
            'naive_high_associated_low_H_frames':int(t.naive_high_associated_low_h.sum()),
            'ambiguous_nearest_H_frames':int(t.association_status_h.eq('ambiguous').sum()),
            'slot_permutation_invariant':invariant})
    return PlayerAcquisitionScenarios(pd.concat(frames,ignore_index=True),pd.DataFrame(summary),pd.concat(events,ignore_index=True),{
        'source':'deterministic toy roster (3 H, 2 A); fractions=1 mean complete toy observation, not an 11-player match',
        'semantics':'diagnostic candidate contracts, not model possession labels, event precision/recall or match ground truth',
        'slot_permutation':'H observation columns swap every other frame; geometry and local association must preserve event candidates',
        'known_limit':'both teams can satisfy encounter criteria; contested proxy remains ambiguous; unique spatial links are not proof of identity',
        'no_scorer':'no fusion, Viterbi, tag or label adjustment is performed'})
