"""Production game-state diagnostics; no labels or evidence are modified."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from .features import _runs


@dataclass
class GameStateDiagnostics:
    frames: pd.DataFrame
    natural_stoppages: pd.DataFrame
    boundary_context: pd.DataFrame
    availability_summary: pd.DataFrame
    summary: dict


def game_state_diagnostics(f,q,live,fusion,decoded,labels,fps=25):
    half=f.half.to_numpy(int);ids=f.frame.to_numpy(int);labels=np.asarray(labels)
    natural=f.graded.to_numpy(bool)&~fusion.control_components.forced_d.to_numpy(bool)
    delta=fusion.state_log_scores[:,2]-fusion.state_log_scores[:,:2].max(axis=1)
    t=pd.DataFrame({'frame':ids,'half':half,'natural':natural,'ball_observed':q.frames.ball_observed,
        'ball_motion_reliable':q.frames.ball_motion_reliable,'player_quality_weight':q.frames.player_quality_weight,
        'motion_evidence_available':live.frames.has_player_motion,'raw_live_utility':live.live_score,
        'raw_dead_utility':live.dead_score,'dead_minus_best_live':delta,
        'decoded_label':decoded.labels,'final_label':labels})
    stops=[];positive=[];bounds=[]
    for h in (1,2):
        for l,r in _runs(natural&(half==h)&(labels=='D')):
            stops.append({'half':h,'start_frame':int(ids[l]),'end_frame':int(ids[r]),'n_frames':r-l+1,
                'duration_s':(r-l+1)/fps,'observed_fraction':float(t.ball_observed.iloc[l:r+1].mean()),
                'mean_dead_advantage':float(delta[l:r+1].mean())})
        positive.extend(list(_runs(natural&(half==h)&(delta>0))))
    for kind,values in [('decoded',np.asarray(decoded.labels)),('final',labels)]:
        changes=np.flatnonzero((values[1:]!=values[:-1])&(half[1:]==half[:-1])&(half[1:]!=0))+1
        for i in changes:
            if 'D' not in (values[i-1],values[i]) or not natural[i]:continue
            bounds.append({'stage':kind,'half':int(half[i]),'frame':int(ids[i]),'from':values[i-1],'to':values[i],
                'ball_observed':bool(t.ball_observed.iloc[i]),'motion_evidence_available':bool(t.motion_evidence_available.iloc[i]),
                'dead_advantage':float(delta[i])})
    stop_table=pd.DataFrame(stops,columns=['half','start_frame','end_frame','n_frames','duration_s','observed_fraction','mean_dead_advantage'])
    availability=t.loc[natural].groupby(['half','ball_observed','ball_motion_reliable','motion_evidence_available']).agg(
        frames=('frame','size'),dead_frames=('final_label',lambda x:int(x.eq('D').sum())),mean_dead_advantage=('dead_minus_best_live','mean')).reset_index()
    summary={'natural_frames':int(natural.sum()),'natural_D_frames':int((natural&(labels=='D')).sum()),
        'natural_D_intervals':len(stops),'median_D_duration_s':stop_table.duration_s.median(),
        'positive_dead_preference_frames':int((natural&(delta>0)).sum()),'positive_dead_preference_runs':len(positive),
        'D_mask_refinement_changed_frames':int((natural&((np.asarray(decoded.labels)=='D')!=(labels=='D'))).sum()),
        'semantics':'behavioral counts, not accuracy; natural excludes published forced D; unknown observations are not GT dead'}
    return GameStateDiagnostics(t,stop_table,pd.DataFrame(bounds,columns=['stage','half','frame','from','to','ball_observed','motion_evidence_available','dead_advantage']),availability,summary)
