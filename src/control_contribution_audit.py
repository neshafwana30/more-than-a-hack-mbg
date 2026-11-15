"""Descriptive signed control-gap decomposition; no scoring or model selection."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from .artifacts import ArtifactRun,read_json,read_table


@dataclass
class ContributionAudit:
    frames: pd.DataFrame
    regime_summary: pd.DataFrame
    component_summary: pd.DataFrame
    metadata: dict


def audit_control_contributions(features, components, forced_d=None):
    if not features.frame.is_unique or not components.frame.is_unique:raise ValueError('Duplicate frames')
    if not np.array_equal(features.frame,components.frame):raise ValueError('Frame keys/order must match')
    c=components;f=features
    natural=f.graded.to_numpy(bool) if forced_d is None else f.graded.to_numpy(bool)&~np.asarray(forced_d,bool)
    t=f[['frame','half']].copy();t['natural']=natural
    for name in ['distance','tag','persistence','signed_margin']:
        t[f'{name}_gap']=c[f'{name}_h']-c[f'{name}_a']
    t['geometry_gap']=t.distance_gap+t.signed_margin_gap
    t['legacy_gap']=c.legacy_h_score-c.legacy_a_score
    t['support_difference']=c.acquisition_gate_h-c.acquisition_gate_a
    t['support_regime']=np.select([
        c.acquisition_gate_h.gt(0)&c.acquisition_gate_a.gt(0),
        c.acquisition_gate_h.gt(0),c.acquisition_gate_a.gt(0)],['both','H_only','A_only'],default='neither')
    t['support_opposes_legacy']=t.support_difference*t.legacy_gap<0
    t['support_opposes_geometry']=t.support_difference*t.geometry_gap<0
    t['support_opposes_persistence']=t.support_difference*t.persistence_gap<0
    t['support_agrees_legacy']=t.support_difference*t.legacy_gap>0
    np.testing.assert_allclose(t[['distance_gap','tag_gap','persistence_gap','signed_margin_gap']].sum(axis=1),t.legacy_gap,rtol=0,atol=1e-12)
    g=t.loc[natural]
    regimes=g.groupby(['half','support_regime']).agg(frames=('frame','size'),
        mean_legacy_gap=('legacy_gap','mean'),mean_geometry_gap=('geometry_gap','mean'),
        mean_persistence_gap=('persistence_gap','mean'),mean_tag_gap=('tag_gap','mean'),
        opposes_legacy_frames=('support_opposes_legacy','sum'),opposes_geometry_frames=('support_opposes_geometry','sum'),
        opposes_persistence_frames=('support_opposes_persistence','sum'),agrees_legacy_frames=('support_agrees_legacy','sum')).reset_index()
    rows=[]
    for name in ['distance_gap','signed_margin_gap','persistence_gap','tag_gap','legacy_gap','support_difference']:
        v=g[name]
        rows.append({'component':name,'signed_sum':float(v.sum()),'absolute_sum':float(v.abs().sum()),
                     'median_abs':float(v.abs().median()),'p95_abs':float(v.abs().quantile(.95))})
    return ContributionAudit(t,regimes,pd.DataFrame(rows),{
        'semantics':'signed H-A components; absolute sums are not additive shares because signed components may cancel',
        'support':'existing causal qualified co-motion gates; not labels, calibrated confidence or independent evidence',
        'limitation':'shared proximity threshold and nearest-point geometry; opposing preferences do not prove either is correct or information is independent',
        'scope':'natural frames exclude forced D if supplied; no GT, model ranking or threshold selection'})


def run_contribution_audit(reference_dir, *, project_dir, storage_config_path):
    reference_dir=Path(reference_dir);m=read_json(reference_dir);p=m['final_core_pass']
    with ArtifactRun(read_json(reference_dir,'config/model.json'),project_dir=project_dir,
        storage_config_path=storage_config_path,entrypoint='control_contribution_audit',
        execution={'experiment':'A2b_preflight_contributions','reference_run_id':m['run_id']}) as run:
        names=['features/initial/frames.csv',f'core/{p}/control/components.csv',f'core/{p}/fusion/control_components.csv']
        run.json('inputs/reference.json',{'run_id':m['run_id'],'path':str(reference_dir),
            'files':{name:m['files'][name]['sha256'] for name in names}})
        f=read_table(reference_dir,names[0]);c=read_table(reference_dir,names[1]);forced=read_table(reference_dir,names[2])
        if not np.array_equal(f.frame,forced.frame):raise ValueError('Forced-mask frame mismatch')
        run.capture('control_contributions',audit_control_contributions,f,c,forced.forced_d)
        folder=str(run.path)
    return folder
