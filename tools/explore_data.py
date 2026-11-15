"""Local aggregate exploration of supplied assets and the actual src pipeline.
No GT fitting, parameter sweep, frame exports, video clips, or submission writes.
Run: mbg/Scripts/python.exe tools/explore_data.py
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import platform
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import games
from src.features import build_features, _runs
from src.pipeline import _run_core
from src.boundary_refinement import refine_boundaries
from src.ambiguity_detector import select_ambiguities
from src.submission import aggregate_diagnostics, segments_for_labels, validate_labels, validate_submission_file


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def describe(values):
    x = pd.to_numeric(pd.Series(values), errors='coerce').to_numpy(float)
    x = x[np.isfinite(x)]
    return dict(zip(['count','min','p05','p25','median','p75','p95','p99','max'],
                    [len(x), *np.quantile(x, [0,.05,.25,.5,.75,.95,.99,1])])) if len(x) else {'count':0}


def schema(df, mask=None):
    sample = df if mask is None else df.loc[mask]
    out = []
    for col in df:
        s = sample[col]
        row = {'column':col,'dtype':str(df[col].dtype),'rows':len(s),
               'non_null':int(s.notna().sum()),'null_pct':float(s.isna().mean()*100),
               'unique_non_null':int(s.nunique(dropna=True))}
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            row.update(describe(s))
            row['infinite_count'] = int(np.isinf(s.to_numpy(float)).sum())
        out.append(row)
    return pd.DataFrame(out)


def table(df):
    def cell(x):
        if isinstance(x, float):
            return f'{x:.4f}' if np.isfinite(x) else '-'
        return str(x).replace('|','/').replace('\n',' ')
    return '\n'.join(['| '+' | '.join(map(str,df.columns))+' |',
                      '| '+' | '.join(['---']*len(df.columns))+' |',
                      *['| '+' | '.join(cell(x) for x in row)+' |' for row in df.itertuples(index=False,name=None)]])


def write_json(path, value):
    def default(x):
        if isinstance(x,np.generic):return x.item()
        if isinstance(x,np.ndarray):return x.tolist()
        if isinstance(x,Path):return str(x)
        raise TypeError(type(x).__name__)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,default=default,allow_nan=False)+'\n',encoding='utf-8')


def feature_definition(name):
    if name in ('frame','half','graded'): return 'timeline','frame / babak / boolean','Indeks video 0-based; half=0 di luar dua blok penilaian.'
    if name in ('centroid_speed_H_mps','centroid_speed_A_mps'): return 'alias gerak','m/s','Alias player_motion_robust, bukan kecepatan centroid tim.'
    if name.startswith('centroid_valid'): return 'kualitas gerak','boolean','Validitas gerak robust; nama centroid merupakan kompatibilitas historis.'
    if name.startswith('player_motion_robust'): return 'gerak pemain','m/s','Median perpindahan slot berurutan yang masuk akal; slot bukan identitas permanen.'
    if name.endswith('_mps'): return 'gerak bola','m/s','Turunan posisi 2D pada frame berurutan dalam babak yang sama; bukan kecepatan 3D.'
    if name.endswith('_m'): return 'geometri','meter','Jarak/margin dari koordinat normalisasi dikali ukuran lapangan 105.3 x 68 m.'
    if name in ('ball_x','ball_y') or name.startswith('nearest_') and name.endswith(('_x','_y')): return 'posisi','koordinat 0-1','Observasi posisi; dapat NaN, periksa outlier di luar lapangan.'
    if name.endswith('_s'): return 'visibilitas','detik','Jarak waktu ke observasi bola; fitur until/next memakai masa depan dalam inferensi offline.'
    if 'frames' in name or name.startswith('frames_'): return 'visibilitas/event','frame','Panjang gap/jarak ke observasi atau batas tag; bukan durasi possession.'
    if name in ('home_count','away_count'): return 'cakupan pemain','jumlah slot','Slot dengan x/y finite; bisa lebih dari 11 karena duplikasi tracking.'
    if name.startswith('valid_'): return 'cakupan pemain','0-1','Cakupan dibagi 11 dan dicap 1; duplikasi tidak terlihat setelah clipping.'
    if name=='nearest_team_proxy': return 'kontrol proxy','H/A/?','Tim terdekat secara geometri; bukan kepemilikan bola ground truth.'
    if name.startswith('active_'): return 'tagging','jumlah interval','Tag aktif menggunakan interval [start,end); kedua tim bisa bertag bersamaan.'
    if name.startswith('tag_'): return 'tagging','kategori / 0-1','Konteks otomatis. Support mencap keluarga tag menyerang yang berkorelasi.'
    return 'kualitas/konteks','boolean / kategori','Indikator observasi atau event; bukan label kebenaran.'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='evaluation/reports/exploration')
    parser.add_argument('--config',default='config/v2.yaml')
    args=parser.parse_args()
    out=ROOT/args.output;out.mkdir(parents=True,exist_ok=True)
    cfgpath=ROOT/args.config;cfg=yaml.safe_load(cfgpath.read_text(encoding='utf-8'))
    hashes={str(p.relative_to(ROOT)):digest(p) for p in sorted((ROOT/'src').glob('*.py'))}
    hashes.update({str(cfgpath.relative_to(ROOT)):digest(cfgpath),'games.py':digest(ROOT/'games.py'),
                   'tools/explore_data.py':digest(Path(__file__))})
    summary={'created_at':datetime.now(ZoneInfo('Asia/Bangkok')).isoformat(),
             'python':platform.python_version(),'code_hashes':hashes,
             'policy':'Descriptive only; no GT training/tuning; no per-frame tracking or video export.'}
    def save(name,df):df.to_csv(out/(name+'.csv'),index=False)
    print('1/7 Inventory, tracking counters, homographies and video metadata',flush=True)
    inv=[];homstats=[];video=[];homdata={};counter=[]
    import cv2
    for source in ('tactical','broadcast'):
        paths=games.asset_paths(source=source)
        for kind,fn in paths.items():
            p=Path(fn);size=p.stat().st_size if p.is_file() else None
            row={'source':source,'asset':kind,'path':str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else p.name,
                 'exists':p.is_file(),'bytes':size,'status':'ok' if size else 'empty_or_missing'}
            if size and kind!='video': row['sha256']=digest(p)
            inv.append(row)
        raw=Path(paths['atd_raw'])
        if raw.is_file() and raw.stat().st_size:
            n=bad=0;first=last=None
            with raw.open() as handle:
                for line in handle:
                    if not line.strip():continue
                    fid=int(line.split(':',1)[0]);n+=1
                    if first is None:first=fid
                    if last is not None and fid!=last+1:bad+=1
                    last=fid
            counter.append({'source':source,'records':n,'first':first,'last':last,'discontinuities':bad,
                            'expected':games.source_frames(source),'ok':n==games.source_frames(source) and first==1 and last==n and bad==0})
        hp=Path(paths['homography'])
        if hp.is_file():
            hom=pd.read_csv(hp);cols=[f'h{i}{j}' for i in range(3) for j in range(3)]
            matrices=hom[cols].to_numpy(float).reshape(-1,3,3)
            finite=np.isfinite(matrices).all(axis=(1,2));valid=hom.valid.to_numpy()==1
            det=np.full(len(hom),np.nan);det[finite]=np.linalg.det(matrices[finite])
            masks={'all':np.ones(len(hom),bool)}
            if source=='tactical':
                for h,lo,hi in games.half_video_blocks(games.GAME):
                    masks[f'half_{h}']=(hom.frame>=lo)&(hom.frame<=hi)
                masks['ungraded']=~(masks['half_1']|masks['half_2'])
                homdata={'df':hom,'matrices':matrices}
            for region,mask in masks.items():
                m=np.asarray(mask);usable=m&valid&finite
                er=describe(hom.loc[usable,'reprojection_error'])
                homstats.append({'source':source,'region':region,'rows':int(m.sum()),'valid_rows':int((m&valid).sum()),
                                 'invalid_finite_mismatch':int((m&(valid!=finite)).sum()),
                                 'near_singular_valid':int((usable&(np.abs(det)<1e-10)).sum()),
                                 'sequential_frame_index':bool(np.array_equal(hom.frame.to_numpy(),np.arange(len(hom)))),
                                 **{'error_'+k:v for k,v in er.items()}})
        cap=cv2.VideoCapture(paths['video'])
        video.append({'source':source,'opened':cap.isOpened(),'frames':int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                      'fps':cap.get(cv2.CAP_PROP_FPS),'width':int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                      'height':int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))})
        video[-1]['expected_frames'] = games.source_frames(source)
        video[-1]['frame_count_matches_expected'] = video[-1]['frames'] == video[-1]['expected_frames']
        if not video[-1]['frame_count_matches_expected']:
            cap.set(cv2.CAP_PROP_POS_FRAMES, games.source_frames(source)-1)
            tail_ok, _ = cap.read()
            video[-1]['expected_last_frame_decodes'] = bool(tail_ok)
        cap.release()
    save('asset_inventory',pd.DataFrame(inv));save('tracking_counters',pd.DataFrame(counter))
    save('homography_summary',pd.DataFrame(homstats));save('video_metadata',pd.DataFrame(video))
    summary['asset_issues']=[x for x in inv if x['status']!='ok']
    summary['tracking_counters']=counter;summary['video_metadata']=video
    summary['video_frame_count_issues']=[x for x in video if not x['frame_count_matches_expected']]
    assert all(x['ok'] for x in counter),'Unexpected raw tracking timeline'
    print('2/7 Raw tactical schema and optional-field availability',flush=True)
    from kloppy import metrica
    paths=games.asset_paths();ds=metrica.load_tracking_epts(meta_data=paths['atd_meta'],raw_data=paths['atd_raw'])
    raw=ds.to_df();save('raw_schema',schema(raw))
    summary['raw_shape']=list(raw.shape)
    summary['optional_fields']={c:{'non_null':int(raw[c].notna().sum()),'rows':len(raw)}
                                for c in ('ball_state','ball_owning_team_id','ball_z','ball_speed') if c in raw}
    del raw,ds;gc.collect()
    print('3/7 Build actual src/features.py tables (no cached notebook features)',flush=True)
    bundle=build_features();f=bundle.frames;graded=f.graded.to_numpy(bool);half=f.half.to_numpy(int);n=len(f)
    summary['feature_shape']=list(f.shape);summary['event_shape']=list(bundle.events.shape)
    summary['slot_counts']=bundle.slots.side.value_counts().to_dict()
    save('feature_schema_all',schema(f));save('feature_schema_graded',schema(f,graded))
    definitions=pd.DataFrame([{'column':c,'family':feature_definition(c)[0],'unit':feature_definition(c)[1],
                               'meaning':feature_definition(c)[2]} for c in f])
    save('feature_dictionary',definitions)
    coverage=[]
    regions={'all':np.ones(n,bool),'graded':graded,'half_1':half==1,'half_2':half==2,'ungraded':~graded}
    for region,m in regions.items():
        coverage.append({'region':region,'frames':int(m.sum()),'ball_seen_pct':float(f.loc[m,'ball_ok'].mean()*100),
                         'mean_players':float((f.home_count+f.away_count)[m].mean()),
                         'over_22_frames':int(((f.home_count+f.away_count>22)&m).sum())})
    save('coverage',pd.DataFrame(coverage));summary['coverage']=coverage
    gaps=[]
    for h in (1,2):
        ids=np.flatnonzero(half==h);lo=int(ids[0]);hi=int(ids[-1])
        for s,e in _runs(~f.ball_ok.to_numpy(bool)[lo:hi+1],lo):
            left=str(f.at[s-1,'nearest_team_proxy']) if s>lo else '?'
            right=str(f.at[e+1,'nearest_team_proxy']) if e<hi else '?'
            gaps.append({'half':h,'start':s,'end':e,'seconds':(e-s+1)/25,
                         'edge_proxy':('same' if left==right else 'different') if left in 'HA' and right in 'HA' else 'unavailable',
                         'set_piece_overlap':bool(f.set_piece_tag.iloc[s:e+1].any()),
                         'tag_coverage':float((f.active_tag_count.iloc[s:e+1]>0).mean())})
    gapdf=pd.DataFrame(gaps)
    summary['gaps']={**describe(gapdf.seconds),'set_piece_overlap':int(gapdf.set_piece_overlap.sum()),
                     'edge_proxy_counts':gapdf.edge_proxy.value_counts().to_dict()}
    save('gap_summary',pd.DataFrame([{'group':str(k),'count':len(g),'median_seconds':g.seconds.median(),
                                    'p95_seconds':g.seconds.quantile(.95),'set_piece_overlap':int(g.set_piece_overlap.sum())}
                                   for k,g in gapdf.groupby(['half','edge_proxy'])]))
    save('geometry_summary',pd.DataFrame([{'feature':c,**describe(f.loc[graded,c])}
             for c in ['nearest_h_m','nearest_a_m','abs_distance_margin_m','ball_speed2d_mps','player_motion_robust_h','player_motion_robust_a']]))
    # Zone membership is defined only where the ball was observed; no inferred missing positions.
    zones=[]
    for axis in ('ball_x','ball_y'):
        cats=pd.cut(f[axis],[-np.inf,0,1/3,2/3,1,np.inf],include_lowest=True)
        for zone,members in f[graded&f.ball_ok].groupby(cats,observed=True):
            zones.append({'axis':axis,'zone':str(zone),'observed_frames':len(members),
                          'median_abs_margin_m':members.abs_distance_margin_m.median(),
                          'median_ball_speed_mps':members.ball_speed2d_mps.median()})
    save('observed_zone_summary',pd.DataFrame(zones))
    ev=bundle.events.copy();tagstats=[]
    for code,group in ev.groupby('code_clean'):
        mask=np.zeros(n,bool)
        for s,e in group[['start_frame','end_frame']].itertuples(index=False,name=None):mask[max(0,int(s)):min(n,int(e))]=True
        mask &= graded
        tagstats.append({'code':code,'instances':len(group),'unknown_team':int((group.side=='?').sum()),
                         'median_duration_s':float((group.end_s-group.start_s).median()),'graded_frames':int(mask.sum()),
                         'missing_ball_pct':float((~f.ball_ok[mask]).mean()*100) if mask.any() else None})
    save('tag_summary',pd.DataFrame(tagstats))
    summary['tags']={'instances':len(ev),'unknown_team':int((ev.side=='?').sum()),
                     'coverage_pct':float((f.active_tag_count[graded]>0).mean()*100),
                     'both_team_active_frames':int((graded&(f.active_h_tags>0)&(f.active_a_tags>0)).sum()),
                     'both_attacking_support_frames':int((graded&(f.tag_h_support>0)&(f.tag_a_support>0)).sum()),
                     'invalid_or_outside_intervals':int(((ev.end_frame<=ev.start_frame)|(ev.start_frame<0)|(ev.end_frame>n)).sum())}
    print('4/7 Evidence components, fixed-config decoder and boundary diagnostics',flush=True)
    quality,event,control,live,fusion,decode=_run_core(f,ev,cfg)
    ambiguity=select_ambiguities(f,quality.frames,control.frames,fusion,decode.labels,cfg.get('ambiguity'))
    boundary=refine_boundaries(decode.labels,fusion.state_log_scores,half,quality.observation_weight,
                               fusion.stop_hint,fusion.restart_hint,cfg.get('boundary'))
    validate_labels(boundary.labels,half,n)
    forced=graded&(half==2)&(f.frame.to_numpy()<int(cfg['fusion']['second_half_kickoff_frame']))
    natural=graded&~forced
    scores=fusion.state_log_scores;d_margin=scores[:,2]-scores[:,:2].max(axis=1)
    summary['evidence']={'unforced_frames':int(natural.sum()),'dead_wins_unforced':int(((d_margin>0)&natural).sum()),
                         'dead_ties_unforced':int(((d_margin==0)&natural).sum()),
                         'dead_advantage_unforced':describe(d_margin[natural]),
                         'dead_decoded_unforced':int(((boundary.labels=='D')&natural).sum()),
                         'suspected_jumps_graded':int(quality.frames.suspected_ball_jump[graded].sum()),
                         'soft_short_spells_merged':decode.diagnostics['short_runs_merged'],
                         'boundaries_examined':len(boundary.report),
                         'boundaries_shifted':int((boundary.report.shift_frames!=0).sum())}
    stage=[]
    for name,labels in [('emission_argmax',np.asarray(['H','A','D'])[scores.argmax(axis=1)]),
                        ('decoded',decode.labels),('refined',boundary.labels)]:
        dg=aggregate_diagnostics(labels,half);seg=segments_for_labels(labels,half);poss=seg[seg.label.isin(['H','A'])]
        stage.append({'stage':name,'possessions':dg['possession_spells'],'dead_segments':dg['dead_segments'],
                      'transitions':dg['total_transitions'],'median_possession_s':dg['possession_median_s'],
                      'spells_lt_1s':int((poss.duration_s<1).sum()),'spells_lt_2s':int((poss.duration_s<2).sum())})
    save('stage_summary',pd.DataFrame(stage));summary['stages']=stage
    summary['prediction']=aggregate_diagnostics(boundary.labels,half)
    selected_frames=np.zeros(n,bool)
    for s,e in ambiguity.windows[['start_frame','end_frame']].itertuples(index=False,name=None):selected_frames[int(s):int(e)+1]=True
    summary['video_selection']={'windows':len(ambiguity.windows),'selected_frames':int(selected_frames.sum()),
                                'missing_frames_selected':int((selected_frames&~f.ball_ok).sum()),
                                'total_missing_graded':int((graded&~f.ball_ok).sum()),
                                'enabled_in_config':bool(cfg['video']['enabled']),
                                'weights_configured':bool(cfg['video'].get('local_yolo_weights'))}
    components=pd.DataFrame({'live':live.live_score,'dead':live.dead_score,'home_control':control.scores[:,0],
                             'away_control':control.scores[:,1],'fused_home':scores[:,0],'fused_away':scores[:,1],
                             'fused_dead':scores[:,2],'dead_minus_best_team':d_margin})
    scorestats=[]
    for condition,m in [('unforced',natural),('ball_observed',natural&f.ball_ok),('ball_missing',natural&~f.ball_ok),
                        ('set_piece_context',natural&f.set_piece_tag)]:
        for c in components:scorestats.append({'condition':condition,'component':c,**describe(components.loc[m,c])})
    save('score_summary',pd.DataFrame(scorestats))
    # Reconstruct additive contributions independently to verify the diagnosis of score scaling.
    lc=cfg['live_dead'];motion=live.frames.player_motion_evidence.to_numpy(float)
    good=quality.frames.ball_motion_reliable.to_numpy(bool);speed=f.ball_speed2d_mps.fillna(0).to_numpy(float)
    near=(event.frames.stoppage_anchor>0)|(event.frames.restart_anchor>0)
    terms={'live_prior':np.full(n,lc['live_prior_score']),
           'ball_motion_live':lc['ball_moving_live_weight']*good*np.clip(speed/5,0,1),
           'players_motion_live':lc['player_motion_live_weight']*live.frames.has_player_motion.to_numpy()*np.clip(motion/3,0,1),
           'dead_prior':np.full(n,lc['dead_prior_score']),
           'outside_dead':lc['out_of_pitch_weight']*event.frames.out_of_pitch_hint.to_numpy()*quality.frames.ball_weight.to_numpy(),
           'stoppage_dead':lc['stoppage_anchor_weight']*event.frames.stoppage_anchor.to_numpy(),
           'restart_dead':lc['restart_anchor_weight']*event.frames.restart_anchor.to_numpy(),
           'stationary_dead':lc['stationary_near_anchor_weight']*near.to_numpy()*good*(speed<.6)}
    assert np.allclose(sum(terms[k] for k in ['live_prior','ball_motion_live','players_motion_live']),live.live_score)
    assert np.allclose(sum(terms[k] for k in ['dead_prior','outside_dead','stoppage_dead','restart_dead','stationary_dead']),live.dead_score)
    save('score_contributions',pd.DataFrame([{'term':k,'mean_unforced':float(v[natural].mean()),
                                            'nonzero_frames_unforced':int((v[natural]!=0).sum()),'max_unforced':float(v[natural].max())} for k,v in terms.items()]))
    trans=decode.transitions
    transrows=[]
    for (a,b),g in trans.groupby(['from','to']):
        ids=g.frame.to_numpy(int)
        transrows.append({'from':a,'to':b,'count':len(g),'missing_ball_pct':float((~f.ball_ok.iloc[ids]).mean()*100),
                          'weak_margin_pct':float((control.frames.control_margin.iloc[ids]<cfg['ambiguity']['margin_threshold']).mean()*100),
                          'set_piece_context_pct':float(f.set_piece_tag.iloc[ids].mean()*100)})
    save('transition_summary',pd.DataFrame(transrows))
    seg=segments_for_labels(boundary.labels,half);pos=seg[seg.label.isin(['H','A'])]
    frag=[]
    for label,g in seg.groupby('label'):frag.append({'label':label,'segments':len(g),**describe(g.duration_s)})
    save('segment_duration_summary',pd.DataFrame(frag))
    context=[]
    for h in (1,2):
        ss=seg[seg.half==h].reset_index(drop=True)
        for i in range(1,len(ss)-1):
            row=ss.iloc[i];prev=ss.iloc[i-1];nxt=ss.iloc[i+1]
            if row.label in ('H','A') and prev.label==nxt.label and row.duration_s<2:
                s,e=int(row.start_frame),int(row.end_frame)
                context.append({'seconds':row.duration_s,'missing_fraction':float((~f.ball_ok.iloc[s:e+1]).mean()),
                                'set_piece_context':bool(f.set_piece_tag.iloc[s:e+1].any())})
    summary['sandwiched_short_spells']={'count':len(context),
          'any_missing_ball':sum(x['missing_fraction']>0 for x in context),
          'set_piece_overlap':sum(x['set_piece_context'] for x in context)}
    print('5/7 Existing predictions, public-GT contract and video geometry QA',flush=True)
    artifacts=[]
    for folder in ('evaluation/results','outputs'):
        for p in sorted((ROOT/folder).glob('*.csv')):
            row={'path':str(p.relative_to(ROOT)),'sha256':digest(p)}
            try:
                validate_submission_file(p,n);pred=pd.read_csv(p).label.to_numpy(str)
                row.update({'valid':True,'same_labels_as_current_run':bool(np.array_equal(pred,boundary.labels)),
                            **{k:v for k,v in aggregate_diagnostics(pred,half).items() if k!='frame_counts'}})
            except Exception as ex:row.update({'valid':False,'error':str(ex)})
            artifacts.append(row)
    save('prediction_artifacts',pd.DataFrame(artifacts));summary['prediction_artifacts']=artifacts
    gtpath=ROOT/'evaluation/gt/BAR-ATH_possession_gt_public.csv'
    validate_submission_file(gtpath,n);gt=pd.read_csv(gtpath);gm=gt.label!='X'
    summary['public_gt_contract']={'rows':len(gt),'graded':int(gm.sum()),'first':int(gt.frame[gm].min()),
                                  'last':int(gt.frame[gm].max()),'labels':gt.label[gm].value_counts().to_dict(),
                                  'use':'Format/alignment contract only; no accuracy comparison or tuning.'}
    qa=[];cap=cv2.VideoCapture(paths['video']);mat=homdata['matrices'];positions=bundle.player_positions
    for h in (1,2):
        ids=np.flatnonzero(half==h)
        for idx in np.linspace(0,len(ids)-1,4,dtype=int):
            frame=int(ids[idx]);cap.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,img=cap.read()
            row={'half':h,'decoded':bool(ok),'players_available':int(positions['valid'][frame].sum())}
            if ok:
                xy=np.c_[positions['x'][frame],positions['y'][frame],np.ones(len(bundle.slots))]
                pr=xy@mat[frame].T
                goodp=positions['valid'][frame]&np.isfinite(pr).all(axis=1)&(np.abs(pr[:,2])>1e-9)
                uv=np.full((len(xy),2),np.nan);uv[goodp]=pr[goodp,:2]/pr[goodp,2,None]
                inside=goodp&(uv[:,0]>=0)&(uv[:,0]<img.shape[1])&(uv[:,1]>=0)&(uv[:,1]<img.shape[0])
                row['players_projected_inside_image']=int(inside.sum())
                row['players_projectable']=int(goodp.sum())
            qa.append(row)
    cap.release()
    # Aggregate only: no frames, positions, screenshots or per-frame QA records are exported.
    summary['video_geometry_qa']={'deterministic_samples':len(qa),'decoded_samples':sum(x['decoded'] for x in qa),
                                'players_available_total':sum(x['players_available'] for x in qa),
                                'players_projected_inside_total':sum(x.get('players_projected_inside_image',0) for x in qa),
                                'limitation':'Numeric projection/decode QA only; in-bounds projection does not prove visual alignment.'}
    print('6/7 Aggregate plots',flush=True)
    fig,ax=plt.subplots(2,2,figsize=(12,8))
    ax[0,0].hist(gapdf.seconds,bins=25,color='#2563eb');ax[0,0].set(xlabel='Missing-ball gap (s)',ylabel='Gaps')
    cov=pd.DataFrame(coverage).set_index('region').loc[['half_1','half_2']]
    ax[0,1].bar(cov.index,cov.ball_seen_pct,color='#0d9488');ax[0,1].set(ylabel='Ball observed (%)',ylim=(0,100))
    ax[1,0].hist(f.loc[graded,'abs_distance_margin_m'].dropna(),bins=60,range=(0,25),color='#7c3aed')
    ax[1,0].set(xlabel='Absolute home-away distance margin (m)',ylabel='Observed frames',title='Display range 0-25 m; full range in CSV')
    ax[1,1].hist(d_margin[natural],bins=60,color='#dc2626');ax[1,1].axvline(0,color='black',ls='--')
    ax[1,1].set(xlabel='Dead score minus best team score (unforced)',ylabel='Frames')
    fig.tight_layout();fig.savefig(out/'data_and_evidence.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(1,3,figsize=(14,4))
    ax[0].hist(pos.duration_s,bins=np.linspace(0,30,61),color='#2563eb');ax[0].set(xlabel='Possession duration (s)',ylabel='Spells',title='0-30 s display; full distribution in CSV')
    st=pd.DataFrame(stage);ax[1].bar(st.stage,st.possessions,color='#0d9488');ax[1].tick_params(axis='x',rotation=20);ax[1].set(ylabel='Possession spells')
    tr=pd.DataFrame(transrows)
    if len(tr):ax[2].bar(tr['from']+'->'+tr['to'],tr['count'],color='#7c3aed')
    ax[2].set(ylabel='Transitions',title='Decoded transitions before refinement')
    fig.tight_layout();fig.savefig(out/'temporal_diagnostics.png',dpi=150);plt.close(fig)
    rel=np.arange(-125,126);profiles=[]
    for boundary_col in ('start_frame','end_frame'):
        arrays=[]
        for anchor in ev.loc[ev.code_clean=='SET PIECES',boundary_col]:
            a=int(anchor);idx=a+rel
            if (0<=a<n and half[a]!=0 and idx.min()>=0 and idx.max()<n
                    and np.all(half[idx]==half[a]) and not np.any(forced[idx])):
                arrays.append(np.c_[f.ball_ok.iloc[idx],live.live_score[idx],live.dead_score[idx],d_margin[idx]])
        if arrays:
            mean=np.mean(arrays,axis=0)
            for j,t in enumerate(rel):profiles.append({'boundary':boundary_col,'relative_s':t/25,'anchors':len(arrays),
                                                      'ball_seen_fraction':mean[j,0],'live':mean[j,1],'dead':mean[j,2],
                                                      'dead_minus_best_team':mean[j,3]})
    profile=pd.DataFrame(profiles)
    assert profile['dead_minus_best_team'].abs().max() < 100, 'Forced kickoff leaked into score profile'
    save('set_piece_profiles',profile)
    fig,ax=plt.subplots(1,2,figsize=(12,4))
    for axis,(kind,group) in zip(ax,profile.groupby('boundary')):
        for c in ('live','dead','dead_minus_best_team'):axis.plot(group.relative_s,group[c],label=c)
        axis.axvline(0,color='gray',ls='--');axis.set(title=f'{kind}; n={int(group.anchors.iloc[0])}',xlabel='Seconds relative to tag boundary',ylabel='Mean score');axis.legend(fontsize=8)
    fig.tight_layout();fig.savefig(out/'set_piece_evidence.png',dpi=150);plt.close(fig)
    print('7/7 Save reproducible aggregate summary',flush=True)
    summary['checks']={'tracking_counters':all(x['ok'] for x in counter),'canonical_features':bool(np.array_equal(f.frame,np.arange(n))),
                       'labels_valid':True,'contributions_reconstructed':True,'public_gt_format_valid':True,
                       'homography_canonical':all(x['sequential_frame_index'] for x in homstats),
                       'video_samples_decoded':all(x['decoded'] for x in qa),
                       'profiles_exclude_forced_kickoff':True}
    write_json(out/'summary.json',summary)
    (out/'README.md').write_text('# Local exploration outputs\n\nAggregate diagnostics only. Generated by `tools/explore_data.py`.\n'
          'Raw/feature schema statistics cover all frames unless filename says graded. Scores exclude forced kickoff where indicated.\n'
          'Tag rows overlap: do not add their frame counts. Zone analysis includes observed balls only.\n'
          'A short spell or a same/different gap-edge proxy is not a truth label.\n'
          'Source hashes and run time are recorded in summary.json. Video hashes are omitted to avoid rereading 19 GB.\n',encoding='utf-8')
    print(json.dumps({'feature_shape':summary['feature_shape'],'gaps':summary['gaps'],'evidence':summary['evidence'],
                      'prediction':summary['prediction'],'checks':summary['checks']},indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else x),flush=True)


if __name__=='__main__':
    main()
