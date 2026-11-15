"""Small deterministic synthetic observations for diagnosing A1, never match labels."""
from dataclasses import dataclass
from copy import deepcopy
import numpy as np
import pandas as pd
from .team_control import estimate_team_control
from .observation_quality import analyze_observations
from .live_dead import estimate_live_dead
from .evidence_fusion import fuse_evidence
from .temporal_decoder import decode_states


@dataclass
class AcquisitionScenarios:
    frames: pd.DataFrame
    summary: pd.DataFrame
    metadata: dict


def acquisition_scenarios(control_config, fps=25, pitch=(105.3,68.0)):
    n=180;ids=np.arange(n);fps=float(fps);pitch=np.asarray(pitch,float)
    scenarios=['same_team_pass','reception_turnover','short_interception','loose_ball','contested_ball','occlusion','nearest_player_switch']
    fallback_enabled=control_config.get('acquisition',{}).get('enabled',False) and control_config.get('acquisition',{}).get('fallback_unavailable',False)
    bonus_enabled=control_config.get('acquisition',{}).get('enabled',False) and control_config.get('acquisition',{}).get('application','contract')=='bonus'
    if fallback_enabled or bonus_enabled:scenarios+=['occlusion_same_team_tag','occlusion_turnover_tag','occlusion_turnover_stale_tag','low_quality_tag']
    if bonus_enabled:scenarios+=['temporary_challenger_comotion','misleading_comotion_tag']
    all_frames=[];rows=[]
    for name in scenarios:
        bx=25.+ids*2/fps;by=np.full(n,34.)
        hx=bx-.6;hy=by.copy();ax=bx-6.;ay=by.copy()
        expected=np.full(n,'H',object);phase=(ids>=60)&(ids<100)
        tag_h=np.zeros(n);tag_a=np.zeros(n);coverage=np.ones(n)
        if name=='same_team_pass':
            bx[60:100]=bx[59]+np.arange(1,41)*8/fps
            bx[100:]=bx[99]+np.arange(1,n-99)*2/fps
            hx=bx-.6
            # Nearest of a stationary sender and receiver during flight; no identity inference.
            sender=bx[59]-.6;receiver=bx[100]+.6
            hx[phase]=np.where(np.abs(bx[phase]-sender)<=np.abs(bx[phase]-receiver),sender,receiver)
            ax=bx-6.;ay=by+8.
        elif name=='reception_turnover':
            phase=ids>=90;hx[phase]=bx[phase]-6.;ax[phase]=bx[phase]-.6;expected[phase]='A'
        elif name=='short_interception':
            phase=(ids>=80)&(ids<100);hx[phase]=bx[phase]-6.;ax[phase]=bx[phase]-.6;expected[phase]='A'
        elif name=='loose_ball':
            hx[phase]=bx[phase]-6.;ax[phase]=bx[phase]+6.;expected[phase]=None
        elif name=='contested_ball':
            hx[phase]=bx[phase]-.8;ax[phase]=bx[phase]+.8;expected[phase]=None
        elif name=='nearest_player_switch':
            hx[phase]=bx[phase]+np.where(ids[phase]%2==0,-.6,.6)
        if name=='temporary_challenger_comotion':
            phase=(ids>=80)&(ids<92)
            hx=bx-1.05;ax[phase]=bx[phase]-(.6+.16*((ids[phase]-80)%3))
        elif name=='misleading_comotion_tag':
            hx[phase]=bx[phase]+np.where(ids[phase]%2==0,-.6,.6)
            ax[phase]=bx[phase]-.8;tag_a[phase]=1.
        if name.startswith('occlusion_') or name=='low_quality_tag':
            tag_h[:]=1.
        if name in ('occlusion_turnover_tag','occlusion_turnover_stale_tag'):
            expected[80:]='A';hx[80:]=bx[80:]-6.;ax[80:]=bx[80:]-.6
            if name=='occlusion_turnover_tag':tag_h[80:]=0.;tag_a[80:]=1.
        if name=='low_quality_tag':coverage[phase]=.2
        seen=np.ones(n,bool)
        if name=='occlusion' or name.startswith('occlusion_'):seen[phase]=False;bx[phase]=np.nan;by[phase]=np.nan
        dh=np.hypot(bx-hx,by-hy);da=np.hypot(bx-ax,by-ay)
        speed=np.r_[np.nan,np.hypot(np.diff(bx),np.diff(by))*fps]
        f=pd.DataFrame({'frame':ids,'half':np.ones(n,int),'graded':True,'ball_ok':seen,
            'ball_x':bx/pitch[0],'ball_y':by/pitch[1],'ball_speed2d_mps':speed,
            'ball_missing_run_frames':np.where(seen,0,int(phase.sum())),
            'valid_h_fraction':coverage,'valid_a_fraction':coverage,'centroid_valid_h':True,'centroid_valid_a':True,
            'player_motion_robust_h':2.,'player_motion_robust_a':2.,
            'nearest_h_m':dh,'nearest_a_m':da,'nearest_h_x':hx/pitch[0],'nearest_h_y':hy/pitch[1],
            'nearest_a_x':ax/pitch[0],'nearest_a_y':ay/pitch[1],
            'nearest_team_proxy':np.where(~seen,'?',np.where(dh<da,'H',np.where(da<dh,'A','?'))),
            'tag_h_support':tag_h,'tag_a_support':tag_a})
        quality=analyze_observations(f);control=estimate_team_control(f,quality.frames,control_config,fps,pitch)
        events=pd.DataFrame({'frame':ids,'stoppage_anchor':0.,'restart_anchor':0.,'out_of_pitch_hint':0.})
        from types import SimpleNamespace
        event=SimpleNamespace(frames=events)
        live=estimate_live_dead(f,quality.frames,events)
        fusion=fuse_evidence(f,quality,control,live,event,{'control_normalization':'max'})
        decoded=decode_states(fusion.state_log_scores,f.half.to_numpy(),fusion.transition_hint,fusion.stop_hint,fusion.restart_hint)
        c=control.components
        a1_decoded=decoded
        if fallback_enabled or bonus_enabled:
            a1_cfg=deepcopy(control_config);a1_cfg['acquisition']['fallback_unavailable']=False;a1_cfg['acquisition']['application']='contract'
            a1_control=estimate_team_control(f,quality.frames,a1_cfg,fps,pitch)
            a1_fusion=fuse_evidence(f,quality,a1_control,live,event,{'control_normalization':'max'})
            a1_decoded=decode_states(a1_fusion.state_log_scores,f.half.to_numpy(),a1_fusion.transition_hint,a1_fusion.stop_hint,a1_fusion.restart_hint)
        f1_control=estimate_team_control(f,quality.frames,{k:v for k,v in control_config.items() if k!='acquisition'},fps,pitch)
        f1_fusion=fuse_evidence(f,quality,f1_control,live,event,{'control_normalization':'max'})
        f1_decoded=decode_states(f1_fusion.state_log_scores,f.half.to_numpy(),f1_fusion.transition_hint,f1_fusion.stop_hint,f1_fusion.restart_hint)
        # Expected labels are only checked away from synthetic phase boundaries.
        check=np.ones(n,bool);check[:12]=False
        changes=(phase[1:]!=phase[:-1])|(pd.Series(expected).fillna('?').to_numpy()[1:]!=pd.Series(expected).fillna('?').to_numpy()[:-1])
        for boundary in np.flatnonzero(changes)+1:check[max(0,boundary-6):boundary+7]=False
        check&=pd.notna(expected)
        passed=bool(np.all(decoded.labels[check]==expected[check]))
        interior=phase.copy()
        interior[:65]=False
        if name=='short_interception':interior=(ids>=87)&(ids<94)
        if name=='reception_turnover':interior=ids>=100
        if name in ('loose_ball','occlusion','nearest_player_switch'):
            passed=passed and bool(c.loc[interior,['acquisition_gate_h','acquisition_gate_a']].eq(0).all().all())
        if name=='contested_ball':
            passed=passed and bool(c.loc[interior,['acquisition_gate_h','acquisition_gate_a']].gt(0).all().all())
        limitation=bool(name=='occlusion_turnover_stale_tag' and np.any(decoded.labels[check]!=expected[check]))
        if name=='occlusion_turnover_stale_tag':
            passed=limitation and bool(c.loc[phase,'fallback_applied'].all() if fallback_enabled else c.loc[phase,['bonus_h','bonus_a']].eq(0).all().all()) and bool(np.all(decoded.labels[110:]=='A'))
        if fallback_enabled and name in ('occlusion_same_team_tag','occlusion_turnover_tag','low_quality_tag'):
            passed=passed and bool(c.loc[phase,'fallback_applied'].all())
        if name=='misleading_comotion_tag':
            limitation=bool(np.any(decoded.labels[check]!=expected[check]))
            passed=limitation and bool(np.all(f1_decoded.labels[check]==expected[check])) and bool(c.loc[65:95,'bonus_a'].gt(0).all())
        out=c[['frame','acquisition_gate_h','acquisition_gate_a','winner_gate','winner_primary_reason','winner_evidence_status','fallback_applied','effective_gate','fallback_restored_gap','bonus_h','bonus_a','bonus_gap']].copy()
        out.insert(0,'scenario',name);out['phase']=np.where(phase,'challenge','context')
        out['expected_label']=expected;out['decoded_label']=decoded.labels;out['a1_decoded_label']=a1_decoded.labels;out['f1_decoded_label']=f1_decoded.labels;out['expected_label_checked']=check
        all_frames.append(out)
        rows.append({'scenario':name,'passed':passed,'expectation_kind':('known_limitation_stale_tag' if name=='occlusion_turnover_stale_tag' else 'known_limitation_misleading_comotion' if name=='misleading_comotion_tag' else 'label_and_support'),'limitation_detected':limitation,'checked_label_frames':int(check.sum()),
            'checked_label_matches':int((decoded.labels[check]==expected[check]).sum()),
            'f1_checked_label_matches':int((f1_decoded.labels[check]==expected[check]).sum()),
            'changed_from_f1_frames':int((decoded.labels!=f1_decoded.labels).sum()),
            'challenge_bonus_differential_frames':int(c.loc[phase,'bonus_gap'].abs().gt(1e-12).sum()),
            'a1_checked_label_matches':int((a1_decoded.labels[check]==expected[check]).sum()),
            'changed_from_a1_frames':int((decoded.labels!=a1_decoded.labels).sum()),
            'challenge_fallback_fraction':c.loc[phase,'fallback_applied'].mean(),
            'challenge_effective_zero_fraction':c.loc[phase,'effective_gate'].eq(0).mean(),
            'challenge_frames':int(phase.sum()),'challenge_zero_gate_fraction':c.loc[phase,'winner_gate'].eq(0).mean(),
            'challenge_unavailable_fraction':c.loc[phase,'winner_evidence_status'].eq('unavailable').mean(),
            'challenge_not_supporting_proxy_fraction':c.loc[phase,'winner_evidence_status'].eq('observed_not_supporting_proxy').mean(),
            'challenge_both_supported_fraction':(c.loc[phase,'acquisition_gate_h'].gt(0)&c.loc[phase,'acquisition_gate_a'].gt(0)).mean()})
    return AcquisitionScenarios(pd.concat(all_frames,ignore_index=True),pd.DataFrame(rows),{
        'source':'deterministic synthetic feature sequences only; no competition observations, labels or GT',
        'expected_labels':'scenario construction; loose/contested challenge phases unlabelled; +/-6 frames around phase boundaries excluded from label checks',
        'scorer':'production quality, team_control, live_dead, F1 fusion and temporal decoder; no new scoring logic in notebook',
        'meaning':'passing/switch/occlusion can lose acquisition support despite continued team possession; both-supported contested phase is not assigned a ground-truth owner',
        'scope':'behavioral specification checks, not match accuracy or turnover recall',
        'passed_semantics':'scenario contract; known_limitation_misleading_comotion passes when an opponent co-moving with the ball plus a misleading automatic tag causes errors that F1 avoids;  known_limitation_stale_tag passes when fallback demonstrably retains stale evidence and loses synthetic label matches before reacquisition; inspect checked_label_matches and limitation_detected separately',
        'a1_counterfactual':'same synthetic features/config with fallback_unavailable=False; scored by production functions',
        'fallback_scenarios':'same-team occlusion with tags, turnover during occlusion with responsive/stale tags, and low player quality; included for fallback or bonus; bonus adds temporary challenger and misleading co-motion/tag counterexample',
        'fps':fps,'pitch_m':pitch.tolist()})
