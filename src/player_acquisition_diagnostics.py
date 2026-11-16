"""Offline, read-only local spatial association and reception-candidate diagnostics.

Observation columns are NOT identities. No result from this module feeds scoring.
"""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
from .artifacts import ArtifactRun,read_json,read_table
from .observation_quality import analyze_observations
from .features import _runs


DEFAULTS={'enabled':False,'association_max_speed_mps':12.,'association_ambiguity_margin_m':.15,
    'history_cap_frames':10,'pre_frames':5,'post_frames':5,'maximum_distance_m':2.5,
    'maximum_relative_speed_mps':3.,'minimum_player_quality':.65,'minimum_approach_drop_m':.5,
    'minimum_incoming_speed_mps':3.,'minimum_speed_drop_mps':2.,'minimum_turn_degrees':30.,
    'minimum_post_support_fraction':.8}


@dataclass
class PlayerAcquisitionDiagnostics:
    frames: pd.DataFrame
    association_summary: pd.DataFrame
    association_coverage: pd.DataFrame
    relative_speed_summary: pd.DataFrame
    candidate_summary: pd.DataFrame
    events: pd.DataFrame
    metadata: dict


def diagnostic_config(config):
    if not isinstance(config,dict) or set(config)-set(DEFAULTS):raise ValueError('Invalid player acquisition diagnostic settings')
    cfg={**DEFAULTS,**config}
    if not isinstance(cfg['enabled'],bool):raise ValueError('enabled must be boolean')
    for name in ('history_cap_frames','pre_frames','post_frames'):
        if isinstance(cfg[name],bool) or not isinstance(cfg[name],int) or cfg[name]<1:raise ValueError(f'{name} must be integer >=1')
    for name in set(cfg)-{'enabled','history_cap_frames','pre_frames','post_frames'}:
        if isinstance(cfg[name],bool) or not np.isfinite(float(cfg[name])) or float(cfg[name])<0:raise ValueError(f'Invalid {name}')
    if cfg['history_cap_frames']>32767:raise ValueError('History cap exceeds storage range')
    if cfg['association_ambiguity_margin_m']<=0:raise ValueError('Positive ambiguity margin required')
    if cfg['association_max_speed_mps']<=0 or cfg['maximum_distance_m']<=0:raise ValueError('Positive speed/radius required')
    if not 0<=cfg['minimum_player_quality']<=1 or not 0<cfg['minimum_post_support_fraction']<=1:raise ValueError('Invalid fraction')
    if not 0<=cfg['minimum_turn_degrees']<=180:raise ValueError('Invalid angle')
    if cfg['history_cap_frames']<max(cfg['pre_frames'],cfg['post_frames'])+1:raise ValueError('history cap must cover candidate windows')
    return cfg


def _gather(values, rows, columns):
    safe_rows=np.clip(rows,0,max(0,len(values)-1));safe_cols=np.maximum(columns,0)
    return values[safe_rows,safe_cols]


def player_acquisition_diagnostics(features,positions,config,quality=None,forced_d=None,fps=25,pitch=(105.3,68.0)):
    cfg=diagnostic_config(config)
    if not cfg['enabled']:raise ValueError('Diagnostic disabled')
    if positions is None:raise ValueError('Player positions unavailable; no slot identity fallback')
    f=features;n=len(f);ids=f.frame.to_numpy(int);half=f.half.to_numpy(int)
    if not n or not f.frame.is_unique or np.any(np.diff(ids)<=0):raise ValueError('Sorted unique nonempty frames required')
    fps=float(fps);pitch=np.asarray(pitch,float)
    if not np.isfinite(fps) or fps<=0 or pitch.shape!=(2,) or not np.isfinite(pitch).all() or (pitch<=0).any():raise ValueError('Invalid FPS/pitch')
    x=np.asarray(positions['x'],float);y=np.asarray(positions['y'],float);valid=np.asarray(positions['valid'],bool)
    sides=np.asarray(positions['sides']).astype(str);slots=len(sides)
    if x.shape!=(n,slots) or y.shape!=x.shape or valid.shape!=x.shape or set(sides)!={'H','A'}:raise ValueError('Invalid player position layout')
    p=np.stack([x*pitch[0],y*pitch[1]],axis=-1)
    valid=valid&np.isfinite(p).all(axis=-1)
    ball=f[['ball_x','ball_y']].to_numpy(float)*pitch
    seen=f.ball_ok.to_numpy(bool)&np.isfinite(ball).all(axis=1)
    q=analyze_observations(f).frames if quality is None else quality
    if not np.array_equal(q.frame,ids):raise ValueError('Quality frame mismatch')
    natural=f.graded.to_numpy(bool) if forced_d is None else f.graded.to_numpy(bool)&~np.asarray(forced_d,bool)
    if natural.shape!=(n,):raise ValueError('Invalid forced mask')
    cont=np.r_[False,(half[1:]==half[:-1])&(half[1:]!=0)&(ids[1:]==ids[:-1]+1)]
    links=np.full((n,slots),-1,np.int16);successor=np.full((n,slots),-1,np.int16)
    chains=np.zeros((n,slots),np.int16)
    nearest=np.full((n,2),-1,np.int16);distance=np.full((n,2),np.nan)
    counts=np.zeros((n,2,4),np.int16)  # observed, linked, ambiguous, no feasible match
    nearest_ambiguous=np.zeros((n,2),bool)
    for t in range(n):
        if half[t]==0:continue
        for si,side in enumerate(('H','A')):
            cur=np.flatnonzero(valid[t]&(sides==side));counts[t,si,0]=len(cur)
            if not len(cur):continue
            chains[t,cur]=1
            if seen[t]:
                d=np.linalg.norm(p[t,cur]-ball[t],axis=1);j=int(np.argmin(d));nearest[t,si]=cur[j];distance[t,si]=d[j]
            if not cont[t]:continue
            prev=np.flatnonzero(valid[t-1]&(sides==side))
            if not len(prev):counts[t,si,3]=len(cur);continue
            d=np.linalg.norm(p[t,cur,None,:]-p[t-1,prev][None,:,:],axis=-1)
            feasible=d<=cfg['association_max_speed_mps']/fps
            costs=np.where(feasible,d,np.inf)
            best_prev=np.argmin(costs,axis=1);best_cur=np.argmin(costs,axis=0)
            for ci,j in enumerate(cur):
                pi=int(best_prev[ci]);step=costs[ci,pi]
                if not np.isfinite(step):counts[t,si,3]+=1;continue
                row=np.sort(costs[ci]);col=np.sort(costs[:,pi])
                row_clear=len(row)<2 or not np.isfinite(row[1]) or row[1]-step>=cfg['association_ambiguity_margin_m']
                col_clear=len(col)<2 or not np.isfinite(col[1]) or col[1]-step>=cfg['association_ambiguity_margin_m']
                if best_cur[pi]!=ci or not row_clear or not col_clear:
                    counts[t,si,2]+=1
                    if nearest[t,si]==j:nearest_ambiguous[t,si]=True
                    continue
                k=prev[pi];links[t,j]=k;successor[t-1,k]=j
                chains[t,j]=min(int(chains[t-1,k])+1,cfg['history_cap_frames']);counts[t,si,1]+=1
    rows=np.arange(n);out={'frame':ids,'half':half,'natural':natural,'ball_observed':seen}
    ball_pair=cont&seen&np.r_[False,seen[:-1]]&q.ball_motion_reliable.to_numpy(bool)&np.r_[False,q.ball_motion_reliable.to_numpy(bool)[:-1]]
    quality_ok=q.ball_motion_reliable.to_numpy(bool)&seen&np.isfinite(q.player_quality_weight)&q.player_quality_weight.ge(cfg['minimum_player_quality']).to_numpy()
    pre=int(cfg['pre_frames']);post=int(cfg['post_frames'])
    window_ok=(rows>=pre)&(rows+post<n)
    prefix=np.r_[0,np.cumsum(quality_ok.astype(int))]
    lo=np.maximum(0,rows-pre);hi=np.minimum(n,rows+post+1)
    window_ok&=(half[np.maximum(rows-pre,0)]==half)&(half[np.minimum(rows+post,n-1)]==half)
    window_ok&=(ids[np.minimum(rows+post,n-1)]-ids[np.maximum(rows-pre,0)]==pre+post)
    reliable_window=window_ok&((prefix[hi]-prefix[lo])==pre+post+1)
    for si,side in enumerate(('h','a')):
        idx=nearest[:,si].astype(int);safe=np.maximum(idx,0);prev_rows=np.maximum(rows-1,0)
        predecessor=links[rows,safe].astype(int);predecessor[idx<0]=-1
        link_ok=cont&(predecessor>=0)&(idx>=0)
        prev_nearest=np.r_[-1,idx[:-1]]
        near_pair=ball_pair&(idx>=0)&(prev_nearest>=0)
        current_pos=p[rows,safe]
        previous_nearest_pos=_gather(p,prev_rows,prev_nearest)
        previous_associated_pos=_gather(p,prev_rows,predecessor)
        old_ball=np.vstack([ball[0],ball[:-1]])
        naive=np.linalg.norm((ball-current_pos)-(old_ball-previous_nearest_pos),axis=1)*fps
        associated=np.linalg.norm((ball-current_pos)-(old_ball-previous_associated_pos),axis=1)*fps
        point_speed=np.linalg.norm(current_pos-previous_nearest_pos,axis=1)*fps
        associated_speed=np.linalg.norm(current_pos-previous_associated_pos,axis=1)*fps
        naive=np.where(near_pair,naive,np.nan);point_speed=np.where(near_pair,point_speed,np.nan)
        associated=np.where(ball_pair&link_ok,associated,np.nan);associated_speed=np.where(link_ok,associated_speed,np.nan)
        reason=np.select([half==0,~seen,idx<0,~cont,nearest_ambiguous[:,si],~link_ok],
                         ['outside_half','ball_unavailable','no_player','block_start','ambiguous','unmatched'],default='linked')
        out.update({f'observed_points_{side}':counts[:,si,0],f'linked_points_{side}':counts[:,si,1],
            f'ambiguous_points_{side}':counts[:,si,2],f'unmatched_points_{side}':counts[:,si,3],
            f'nearest_index_{side}':idx,f'predecessor_index_{side}':predecessor,
            f'association_status_{side}':reason,f'nearest_distance_{side}_m':distance[:,si],
            f'local_history_{side}_frames':np.where(idx>=0,chains[rows,safe],0),
            f'observation_column_changed_{side}':near_pair&(idx!=prev_nearest),
            f'nearest_candidate_switch_{side}':link_ok&(prev_nearest>=0)&(predecessor!=prev_nearest),
            f'nearest_candidate_continues_{side}':link_ok&(predecessor==prev_nearest),
            f'naive_point_speed_{side}_mps':point_speed,f'associated_point_speed_{side}_mps':associated_speed,
            f'naive_relative_speed_{side}_mps':naive,f'associated_relative_speed_{side}_mps':associated,
            f'naive_high_associated_low_{side}':np.isfinite(naive)&np.isfinite(associated)&(naive>cfg['maximum_relative_speed_mps'])&(associated<=cfg['maximum_relative_speed_mps'])})
        previous_near=np.r_[False,distance[:-1,si]<=cfg['maximum_distance_m']]
        coverage=q.player_quality_weight.to_numpy(float)
        coverage_pair=(coverage>=cfg['minimum_player_quality'])&np.r_[False,coverage[:-1]>=cfg['minimum_player_quality']]
        near_context=link_ok&ball_pair&coverage_pair&(distance[:,si]<=cfg['maximum_distance_m'])&previous_near
        near_context&=np.linalg.norm(old_ball-previous_associated_pos,axis=1)<=cfg['maximum_distance_m']
        out[f'naive_high_associated_low_near_{side}']=out[f'naive_high_associated_low_{side}']&near_context
        # Follow only accepted adjacent-frame links; never interpolate an occlusion or force identity.
        back=idx.copy();back_ok=idx>=0
        for step in range(pre):
            rr=rows-step;back_ok&=rr>0
            back=_gather(links,np.maximum(rr,0),back).astype(int);back_ok&=back>=0
        before_pos=_gather(p,rows-pre,back)
        before_ball=ball[np.maximum(rows-pre,0)]
        approach=np.linalg.norm(before_ball-before_pos,axis=1)-distance[:,si]
        forward=idx.copy();forward_ok=idx>=0;close_count=np.zeros(n);motion_count=np.zeros(n)
        previous_position=current_pos;previous_ball=ball
        for step in range(1,post+1):
            rr=rows+step;forward_ok&=rr<n
            forward=_gather(successor,np.minimum(rr-1,n-1),forward).astype(int);forward_ok&=forward>=0
            pp=_gather(p,rr,forward);bb=ball[np.minimum(rr,n-1)]
            close_count+=(np.linalg.norm(bb-pp,axis=1)<=cfg['maximum_distance_m'])&forward_ok
            relative=np.linalg.norm((bb-pp)-(previous_ball-previous_position),axis=1)*fps
            motion_count+=(relative<=cfg['maximum_relative_speed_mps'])&forward_ok
            previous_position=pp;previous_ball=bb
        v_in=(ball-before_ball)*fps/pre
        v_out=(ball[np.minimum(rows+post,n-1)]-ball)*fps/post
        incoming=np.linalg.norm(v_in,axis=1);outgoing=np.linalg.norm(v_out,axis=1);drop=incoming-outgoing
        denom=incoming*outgoing
        cosine=np.divide(np.sum(v_in*v_out,axis=1),denom,out=np.full(n,np.nan),where=denom>1e-12)
        angle=np.degrees(np.arccos(np.clip(cosine,-1,1)))
        ball_change=(drop>=cfg['minimum_speed_drop_mps'])|(angle>=cfg['minimum_turn_degrees'])
        approached=(approach>=cfg['minimum_approach_drop_m'])&(incoming>=cfg['minimum_incoming_speed_mps'])
        stable=(close_count/post>=cfg['minimum_post_support_fraction'])&(motion_count/post>=cfg['minimum_post_support_fraction'])
        available=window_ok&reliable_window&back_ok&forward_ok&(half!=0)&np.isfinite(approach)
        candidate=available&(distance[:,si]<=cfg['maximum_distance_m'])&approached&ball_change&stable
        candidate_reason=np.select([half==0,~window_ok,~reliable_window,~(back_ok&forward_ok),
            ~(distance[:,si]<=cfg['maximum_distance_m']),~approached,~ball_change,~stable],
            ['outside_half','window_boundary','ball_or_quality_unavailable','association_unavailable','not_near','no_approach','no_ball_change','post_unstable'],default='candidate')
        out.update({f'candidate_{side}':candidate,f'candidate_reason_{side}':candidate_reason,
            f'approach_drop_{side}_m':np.where(available,approach,np.nan),f'incoming_speed_{side}_mps':np.where(available,incoming,np.nan),
            f'ball_speed_drop_{side}_mps':np.where(available,drop,np.nan),f'ball_turn_{side}_degrees':np.where(available,angle,np.nan),
            f'post_close_fraction_{side}':np.where(available,close_count/post,np.nan),
            f'post_comotion_fraction_{side}':np.where(available,motion_count/post,np.nan)})
    t=pd.DataFrame(out)
    t['candidate_status']=np.select([t.candidate_h&t.candidate_a,t.candidate_h,t.candidate_a],['both_ambiguous','H_only','A_only'],default='none')
    t['contested_geometry']=(t.nearest_distance_h_m<=cfg['maximum_distance_m'])&(t.nearest_distance_a_m<=cfg['maximum_distance_m'])
    associations=[];coverage_rows=[];speed_rows=[];candidates=[];events=[]
    for side in ('h','a'):
        for h in (1,2):
            g=t.loc[t.natural&t.half.eq(h)]
            observed=int(g[f'observed_points_{side}'].sum());linked=int(g[f'linked_points_{side}'].sum())
            coverage_rows.append({'half':h,'side':side.upper(),'observed_points':observed,'linked_points':linked,
                'ambiguous_points':int(g[f'ambiguous_points_{side}'].sum()),'unmatched_points':int(g[f'unmatched_points_{side}'].sum()),
                'linked_fraction_of_observed':linked/observed if observed else np.nan})
            paired=g.loc[np.isfinite(g[f'naive_relative_speed_{side}_mps'])&np.isfinite(g[f'associated_relative_speed_{side}_mps'])]
            speed_rows.append({'half':h,'side':side.upper(),'paired_measurement_frames':len(paired),
                'naive_exceeds_threshold_frames':int(paired[f'naive_relative_speed_{side}_mps'].gt(cfg['maximum_relative_speed_mps']).sum()),
                'associated_exceeds_threshold_frames':int(paired[f'associated_relative_speed_{side}_mps'].gt(cfg['maximum_relative_speed_mps']).sum()),
                'naive_high_associated_low_frames':int(paired[f'naive_high_associated_low_{side}'].sum()),
                'naive_high_associated_low_near_frames':int(g[f'naive_high_associated_low_near_{side}'].sum()),
                'nearest_candidate_switch_frames':int(g[f'nearest_candidate_switch_{side}'].sum()),
                'observation_column_changed_frames':int(g[f'observation_column_changed_{side}'].sum()),
                'column_changed_but_continues_frames':int((g[f'observation_column_changed_{side}']&g[f'nearest_candidate_continues_{side}']).sum()),
                'median_naive_relative_mps':float(paired[f'naive_relative_speed_{side}_mps'].median()),
                'median_associated_relative_mps':float(paired[f'associated_relative_speed_{side}_mps'].median())})
            for status,count in g[f'association_status_{side}'].value_counts().items():
                associations.append({'half':h,'side':side.upper(),'status':status,'frames':int(count),
                    'naive_high_associated_low_frames':int(g.loc[g[f'association_status_{side}'].eq(status),f'naive_high_associated_low_{side}'].sum())})
            for reason,count in g[f'candidate_reason_{side}'].value_counts().items():
                candidates.append({'half':h,'side':side.upper(),'reason':reason,'frames':int(count)})
            for start,end in _runs((t.natural&t.half.eq(h)&t[f'candidate_{side}']).to_numpy()):
                # Split an episode if the central nearest candidate itself changes.
                cuts=[start]+[i for i in range(start+1,end+1) if links[i,max(0,int(nearest[i,0 if side=='h' else 1]))]!=nearest[i-1,0 if side=='h' else 1]]+[end+1]
                for left,right in zip(cuts[:-1],cuts[1:]):
                    g=t.iloc[left:right];anchor=int(g[f'approach_drop_{side}_m'].idxmax())
                    before=max(0,anchor-pre);dh=t.iloc[before].nearest_distance_h_m;da=t.iloc[before].nearest_distance_a_m
                    prior='H' if np.isfinite(dh) and np.isfinite(da) and dh<da else 'A' if np.isfinite(dh) and np.isfinite(da) and da<dh else '?'
                    events.append({'prior_nearest_team':prior,'geometry_team_change':prior!='?' and prior!=side.upper(),'half':h,'side':side.upper(),'start_frame':int(ids[left]),'end_frame':int(ids[right-1]),
                        'anchor_frame':int(ids[anchor]),'candidate_frames':right-left,'duration_s':(right-left)/fps,
                        'both_candidates_any':bool(g.candidate_status.eq('both_ambiguous').any()),
                        'contested_geometry_any':bool(g.contested_geometry.any()),
                        'max_approach_drop_m':float(g[f'approach_drop_{side}_m'].max()),
                        'max_ball_speed_drop_mps':float(g[f'ball_speed_drop_{side}_mps'].max()),
                        'semantics':'reception_or_interception_proxy_not_possession_label'})
    event_cols=['prior_nearest_team','geometry_team_change','half','side','start_frame','end_frame','anchor_frame','candidate_frames','duration_s','both_candidates_any','contested_geometry_any','max_approach_drop_m','max_ball_speed_drop_mps','semantics']
    return PlayerAcquisitionDiagnostics(t,pd.DataFrame(associations,columns=['half','side','status','frames','naive_high_associated_low_frames']),
        pd.DataFrame(coverage_rows),pd.DataFrame(speed_rows),pd.DataFrame(candidates,columns=['half','side','reason','frames']),pd.DataFrame(events,columns=event_cols),{
        'config':cfg,'fps':fps,'pitch_m':pitch.tolist(),'observation_sides':sides.tolist(),
        'association':'same-team mutual nearest spatial links, speed-gated; both row/column alternative margins must clear threshold; no gap bridging',
        'indices':'observation-column indices at that frame, not player identities; predecessor index belongs to previous frame; history capped, no global track ID',
        'candidate':'near + approach + incoming speed + ball slowdown/turn + post close/co-motion support on accepted local links',
        'summary_scope':'natural central frames; windows are inside a contiguous half and may include adjoining forced-kickoff context',
        'lookahead_frames':post,'lookbehind_frames':pre,'offline_only':'post window is diagnostic lookahead; no scorer/hint/label changes',
        'near_speed_context':'near speed flag additionally requires current/previous nearest and previous associated point within radius, reliable ball pair and player coverage at both endpoints; still not a false-positive label',
        'coverage_semantics':'linked fraction is coverage of local hypotheses, not association accuracy; speed comparison is conditional on accepted associations and reliable ball endpoints',
        'event_kind':'reception_or_interception_proxy; prior_nearest_team and geometry_team_change are geometry context, not previous owner or verified interception',
        'limitations':'spatial uniqueness is not identity; equally plausible swaps/crossings/duplicates remain ambiguous; candidates do not prove reception/interception or controlled possession; both/contested do not choose an owner',
        'thresholds':'initial hypotheses, not tuned on public GT, match labels or target event counts'})



@dataclass
class AcquisitionReferenceContext:
    event_context: pd.DataFrame
    transition_context: pd.DataFrame
    metadata: dict


def acquisition_reference_context(audit,transitions,labels):
    t=audit.frames;ids=t.frame.to_numpy(int);labels=np.asarray(labels)
    if len(labels)!=len(t):raise ValueError('Reference label length mismatch')
    event_rows=[];transition_rows=[]
    width=int(audit.metadata['config']['post_frames'])
    for r in audit.events.to_dict('records'):
        ix=int(np.searchsorted(ids,int(r['anchor_frame'])))
        nearby=transitions.loc[transitions.half.eq(r['half'])&transitions.frame.between(r['anchor_frame']-width,r['anchor_frame']+width)]
        event_rows.append({**r,'reference_label_at_anchor':str(labels[ix]),'nearby_decoded_transitions':len(nearby)})
    for r in transitions.to_dict('records'):
        if r['from'] not in ('H','A') or r['to'] not in ('H','A'):continue
        ix=int(np.searchsorted(ids,int(r['frame'])))
        if ix>=len(t) or ids[ix]!=r['frame']:raise ValueError('Reference transition frame mismatch')
        g=t.loc[t.half.eq(r['half'])&t.frame.between(r['frame']-width,r['frame']+width)]
        transition_rows.append({**r,'candidate_H_nearby':bool(g.candidate_h.any()),'candidate_A_nearby':bool(g.candidate_a.any()),
            'both_candidate_nearby':bool(g.candidate_status.eq('both_ambiguous').any()),
            'association_H_at_boundary':t.iloc[ix].association_status_h,'association_A_at_boundary':t.iloc[ix].association_status_a})
    return AcquisitionReferenceContext(pd.DataFrame(event_rows,columns=[*audit.events.columns,'reference_label_at_anchor','nearby_decoded_transitions']),
        pd.DataFrame(transition_rows,columns=['half','frame','from','to','candidate_H_nearby','candidate_A_nearby','both_candidate_nearby','association_H_at_boundary','association_A_at_boundary']),{
        'semantics':'descriptive alignment with existing predictions; model transitions are not verified turnovers; overlap is not recall/precision',
        'window_radius_frames':width,'no_scorer':'reference predictions are read only; no labels or hints generated'})


def run_player_acquisition_diagnostics(reference_dir,diagnostics,*,project_dir,storage_config_path,diagnostic_config_source=None,execution_metadata=None):
    reference_dir=Path(reference_dir);m=read_json(reference_dir);core=m['final_core_pass'];cfg=diagnostic_config(diagnostics)
    with ArtifactRun(read_json(reference_dir,'config/model.json'),project_dir=project_dir,storage_config_path=storage_config_path,
        entrypoint='player_acquisition_diagnostics',execution={**(execution_metadata or {}),'experiment':'player_association_acquisition_diagnostics','reference_run_id':m['run_id']}) as run:
        run.json('config/diagnostic.json',{'settings':cfg,'source_file':str(diagnostic_config_source) if diagnostic_config_source else None})
        labels_path='result/labels.csv' if 'result/labels.csv' in m['files'] else 'inference/result/labels.csv'
        names=['features/initial/frames.csv','features/initial/player_positions.csv','features/initial/player_positions_metadata.json',
               f'core/{core}/quality/frames.csv',f'core/{core}/fusion/control_components.csv',labels_path,f'core/{core}/decode/transitions.csv']
        if core!='initial':raise ValueError('Standalone audit currently requires unchanged initial features; use inline pipeline for video-refreshed features')
        for name in names:
            if name not in m['files']:raise ValueError(f'Missing reference artifact {name}')
            if hashlib.sha256((reference_dir/name).read_bytes()).hexdigest()!=m['files'][name]['sha256']:raise ValueError(f'Reference checksum mismatch: {name}')
        run.json('inputs/reference.json',{'run_id':m['run_id'],'path':str(reference_dir),
            'files':{name:m['files'][name]['sha256'] for name in names},'prediction_policy':'reuse immutable reference predictions; no scoring or new submission'})
        f=read_table(reference_dir,names[0]);table=read_table(reference_dir,names[1]);meta=read_json(reference_dir,names[2])
        if not np.array_equal(table.frame,f.frame):raise ValueError('Player-position frame mismatch')
        order=meta['slot_order'];positions={key:np.column_stack([table[f'slot_{slot}_{key}'].to_numpy() for slot in order]) for key in ('x','y','valid')}
        positions['sides']=np.asarray(meta['sides']);del table
        q=read_table(reference_dir,names[3]);forced=read_table(reference_dir,names[4])
        if not np.array_equal(f.frame,forced.frame):raise ValueError('Forced mask frame mismatch')
        model=read_json(reference_dir,'config/model.json')
        audit=run.capture('diagnostics/player_acquisition',player_acquisition_diagnostics,f,positions,cfg,q,forced.forced_d,
                    model.get('fps',25),tuple(model.get('pitch',(105.3,68.0))))
        run.capture('diagnostics/player_acquisition_reference',acquisition_reference_context,audit,
            read_table(reference_dir,names[-1]),read_table(reference_dir,labels_path).label.to_numpy())
        from .player_acquisition_scenarios import player_acquisition_scenarios
        run.capture('diagnostics/player_acquisition_synthetic',player_acquisition_scenarios,cfg,model.get('fps',25),tuple(model.get('pitch',(105.3,68.0))))
        # Verify reference prediction bytes after all computations; never overwrite them.
        if hashlib.sha256((reference_dir/labels_path).read_bytes()).hexdigest()!=m['files'][labels_path]['sha256']:raise ValueError('Reference labels changed')
        folder=str(run.path)
    return folder
