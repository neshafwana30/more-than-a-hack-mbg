"""Read-only A1 diagnostics. None of these outputs feeds the model."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from .diagnostics import _runs


@dataclass
class AcquisitionDiagnostics:
    frames: pd.DataFrame
    zero_reason_summary: pd.DataFrame
    failure_flag_summary: pd.DataFrame
    zero_episodes: pd.DataFrame
    boundary_support: pd.DataFrame
    summary: dict


def _support_distances(ids, half, supported):
    previous=np.full(len(ids),np.nan);following=np.full(len(ids),np.nan)
    for h in (1,2):
        ix=np.flatnonzero(half==h);pos=ids[ix[supported[ix]]]
        if not len(pos):continue
        left=np.searchsorted(pos,ids[ix],side='right')-1
        right=np.searchsorted(pos,ids[ix],side='left')
        ok=left>=0;previous[ix[ok]]=ids[ix[ok]]-pos[left[ok]]
        ok=right<len(pos);following[ix[ok]]=pos[right[ok]]-ids[ix[ok]]
    return previous,following


def acquisition_diagnostics(frames, control, fusion, decoded, final_labels, fps=25):
    c=control.components;half=frames.half.to_numpy(int);ids=frames.frame.to_numpy(int)
    natural=frames.graded.to_numpy(bool)&~fusion.control_components.forced_d.to_numpy(bool)
    t=c[['frame','legacy_winner','winner_tie','winner_gate','winner_primary_reason','winner_evidence_status']].copy()
    for field in ['bonus_h','bonus_a','bonus_gap']:
        t[field]=c[field].to_numpy(float)
    t['fallback_applied']=c.fallback_applied.to_numpy(bool)
    t['effective_gate']=c.effective_gate.to_numpy(float)
    t['fallback_restored_gap']=c.fallback_restored_gap.to_numpy(float)
    t['half']=half;t['natural']=natural;t['ball_observed']=frames.ball_ok.to_numpy(bool)
    t['decoded_label']=decoded.labels;t['final_label']=np.asarray(final_labels)
    t['legacy_abs_gap']=np.abs(c.legacy_h_score-c.legacy_a_score)
    t['adjusted_abs_gap']=np.abs(c.adjusted_h_score-c.adjusted_a_score)
    t['removed_gap']=np.maximum(0,t.legacy_abs_gap-t.adjusted_abs_gap)
    t['legacy_gap_nonzero']=t.legacy_abs_gap>1e-12
    for side in ('h','a'):
        prev,nxt=_support_distances(ids,half,(c[f'acquisition_gate_{side}'].to_numpy()>0)&natural)
        t[f'previous_support_{side}_frames']=prev;t[f'next_support_{side}_frames']=nxt
    choose_h=np.where(c.legacy_h_score>c.legacy_a_score,True,
                      np.where(c.legacy_a_score>c.legacy_h_score,False,c.acquisition_gate_h>=c.acquisition_gate_a))
    t['previous_winner_support_frames']=np.where(choose_h,t.previous_support_h_frames,t.previous_support_a_frames)
    t['next_winner_support_frames']=np.where(choose_h,t.next_support_h_frames,t.next_support_a_frames)
    t['nearest_winner_support_frames']=t[['previous_winner_support_frames','next_winner_support_frames']].min(axis=1)
    zero=natural&t.winner_gate.eq(0).to_numpy();z=t.loc[zero]
    grouped=z.groupby(['half','winner_evidence_status','winner_primary_reason'],dropna=False)
    reasons=grouped.agg(frames=('frame','size'),legacy_gap_nonzero_frames=('legacy_gap_nonzero','sum'),
                        removed_gap_sum=('removed_gap','sum'),ball_observed_fraction=('ball_observed','mean')).reset_index()
    flag_rows=[]
    for key in [k for k in c if k.startswith('winner_fail_')]:
        for h in (1,2):
            mask=zero&(half==h);denom=int(mask.sum());count=int(c.loc[mask,key].sum())
            flag_rows.append({'half':h,'flag':key,'frames':count,'zero_frames_denominator':denom,
                              'fraction':count/denom if denom else np.nan})
    episode_rows=[];episode_ids=np.full(len(t),-1,int)
    for h in (1,2):
        for start,end in _runs(zero&(half==h)):
            g=t.iloc[start:end+1];episode_id=len(episode_rows);episode_ids[start:end+1]=episode_id
            episode_rows.append({'episode_id':episode_id,'half':h,'start_frame':int(ids[start]),'end_frame':int(ids[end]),
                'n_frames':len(g),'duration_s':len(g)/float(fps),'primary_reason_mode':g.winner_primary_reason.mode().iloc[0],
                'unavailable_fraction':g.winner_evidence_status.eq('unavailable').mean(),
                'not_supporting_proxy_fraction':g.winner_evidence_status.eq('observed_not_supporting_proxy').mean(),
                'legacy_gap_nonzero_fraction':g.legacy_gap_nonzero.mean(),'removed_gap_sum':g.removed_gap.sum(),
                'ball_observed_fraction':g.ball_observed.mean(),
                'decoded_final_disagreement_frames':int(g.decoded_label.ne(g.final_label).sum())})
    t['zero_episode_id']=episode_ids
    episode_columns=['episode_id','half','start_frame','end_frame','n_frames','duration_s','primary_reason_mode','unavailable_fraction',
        'not_supporting_proxy_fraction','legacy_gap_nonzero_fraction','removed_gap_sum','ball_observed_fraction','decoded_final_disagreement_frames']
    boundary_rows=[]
    for r in decoded.transitions.to_dict('records'):
        if r['from'] not in ('H','A') or r['to'] not in ('H','A'):continue
        ix=int(np.searchsorted(ids,int(r['frame'])));side=r['to'].lower();h=int(r['half'])
        window=(ids>=ids[ix])&(ids<ids[ix]+5)&(half==h)&natural
        boundary_rows.append({'half':h,'frame':int(r['frame']),'from':r['from'],'to':r['to'],
            'destination_gate_at_boundary':float(c.iloc[ix][f'acquisition_gate_{side}']),
            'mean_destination_gate_next_5':float(c.loc[window,f'acquisition_gate_{side}'].mean()),
            'previous_destination_support_frames':t.iloc[ix][f'previous_support_{side}_frames'],
            'next_destination_support_frames':t.iloc[ix][f'next_support_{side}_frames'],
            'destination_reason_at_boundary':c.iloc[ix][f'primary_reason_{side}']})
    boundary_columns=['half','frame','from','to','destination_gate_at_boundary','mean_destination_gate_next_5',
        'previous_destination_support_frames','next_destination_support_frames','destination_reason_at_boundary']
    nonzero=natural&t.legacy_gap_nonzero.to_numpy()
    summary={'natural_frames':int(natural.sum()),'zero_gate_frames':int(zero.sum()),
        'zero_gate_already_zero_gap_frames':int((zero&~t.legacy_gap_nonzero.to_numpy()).sum()),
        'zero_gate_removed_nonzero_gap_frames':int((zero&t.legacy_gap_nonzero.to_numpy()).sum()),
        'nonzero_legacy_gap_natural_frames':int(nonzero.sum()),
        'zero_gate_fraction_given_nonzero_gap':float(zero[nonzero].mean()) if nonzero.any() else None,
        'removed_gap_sum':float(t.loc[natural,'removed_gap'].sum()),
        'legacy_abs_gap_sum':float(t.loc[natural,'legacy_abs_gap'].sum()),
        'fallback_frames':int((natural&t.fallback_applied.to_numpy()).sum()),
        'fallback_restored_nonzero_gap_frames':int((natural&t.fallback_applied.to_numpy()&t.legacy_gap_nonzero.to_numpy()).sum()),
        'fallback_restored_gap_sum':float(t.loc[natural,'fallback_restored_gap'].sum()),
        'effective_zero_gate_frames':int((natural&t.effective_gate.eq(0).to_numpy()).sum()),
        'effective_zero_removed_nonzero_gap_frames':int((natural&t.effective_gate.eq(0).to_numpy()&t.legacy_gap_nonzero.to_numpy()).sum()),
        'score_application':control.metadata['acquisition']['application'],
        'bonus_active_frames':int((natural&(t.bonus_h.gt(0)|t.bonus_a.gt(0)).to_numpy()).sum()),
        'bonus_differential_frames':int((natural&t.bonus_gap.abs().gt(1e-12).to_numpy()).sum()),
        'bonus_opposes_legacy_frames':int((natural&((c.legacy_h_score-c.legacy_a_score)*t.bonus_gap<0).to_numpy()).sum()),
        'bonus_abs_gap_sum':float(t.loc[natural,'bonus_gap'].abs().sum()),
        'gap_nonzero_tolerance':1e-12,
        'semantics':'descriptive proxy failures, not negative possession labels; primary reasons are exclusive by precedence; flags overlap',
        'support_distance':'within same half, natural support only, 0 at supported frame, NaN if no earlier/later support; future distances diagnostic only',
        'tie_selection':'legacy tie selects the larger gate, H on equal gates; winner_tie is explicit, not a home-control claim',
        'fallback_scope':'in bonus mode effective_gate=1 denotes retained legacy coefficient; bonus_gap is additive; winner_gate, zero reason summaries and episodes describe raw acquisition support; effective_gate includes fallback, which is not acquisition support',
        'episode_scope':'contiguous zero winning-gate frames inside natural half; not labelled possession spells'}
    return AcquisitionDiagnostics(t,reasons,pd.DataFrame(flag_rows),pd.DataFrame(episode_rows,columns=episode_columns),
                                  pd.DataFrame(boundary_rows,columns=boundary_columns),summary)
