"""Strict CSV writer, validator and privacy-safe aggregate diagnostics."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import csv
import hashlib
import json
import subprocess
import sys
import numpy as np
import pandas as pd


VALID_LABELS={'H','A','D','X'}


def validate_labels(labels, half, expected_count:int) -> None:
    labels=np.asarray(labels);half=np.asarray(half)
    if len(labels)!=expected_count or len(half)!=expected_count:
        raise ValueError(f'Expected {expected_count} labels and halves; got {len(labels)}')
    if any(str(s) not in VALID_LABELS for s in labels):raise ValueError('Invalid label')
    if np.any((half==0)&(labels!='X')):raise ValueError('Ungraded blocks must be X in v2')
    if np.any((half!=0)&(labels=='X')):raise ValueError('Graded block contains unapproved X')


def save_submission(labels, path, expected_count=199319, half=None) -> dict:
    labels=np.asarray(labels,dtype='<U1')
    if half is not None:validate_labels(labels,half,expected_count)
    elif len(labels)!=expected_count or not set(labels).issubset(VALID_LABELS):
        raise ValueError('Wrong row count or invalid label')
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8',newline='') as handle:
        writer=csv.writer(handle,lineterminator='\n')
        writer.writerow(['frame','label'])
        writer.writerows((i,str(s)) for i,s in enumerate(labels))
    validate_submission_file(path,expected_count)
    checksum=hashlib.sha256(path.read_bytes()).hexdigest()
    return {'path':str(path),'rows':expected_count,'bytes':path.stat().st_size,'sha256':checksum}


def validate_submission_file(path,expected_count=199319):
    path=Path(path)
    df=pd.read_csv(path,keep_default_na=False)
    if list(df.columns)!=['frame','label'] or len(df)!=expected_count:
        raise ValueError('CSV header or row count invalid')
    frames=pd.to_numeric(df.frame,errors='raise').to_numpy()
    if not np.array_equal(frames,np.arange(expected_count)):
        raise ValueError('Frame IDs must be ordered 0..N-1')
    if any(str(x) not in VALID_LABELS for x in df.label):raise ValueError('Invalid CSV label')
    return True


def official_validator_if_available(project_dir, output_path) -> dict:
    official=Path(project_dir)/'evaluation'/'validate_submission.py'
    if not official.is_file():return {'status':'not_available','note':'Official scorer not included; local strict validation passed.'}
    r=subprocess.run([sys.executable,str(official),str(output_path)],cwd=str(project_dir),text=True,
                     capture_output=True,timeout=120)
    if r.returncode!=0:raise RuntimeError(f'Official validator failed: {r.stdout}\n{r.stderr}')
    return {'status':'passed','message':r.stdout.strip()[:1000]}


def segments_for_labels(labels,half,fps=25):
    labels=np.asarray(labels);half=np.asarray(half)
    out=[]
    for h in (1,2):
        ix=np.flatnonzero(half==h)
        if len(ix)==0:continue
        lo=int(ix[0]);hi=int(ix[-1])+1
        local=labels[lo:hi]
        boundaries=np.r_[0,np.flatnonzero(local[1:]!=local[:-1])+1,len(local)]
        for s,e in zip(boundaries[:-1],boundaries[1:]):
            out.append({'half':h,'start_frame':int(lo+s),'end_frame':int(lo+e-1),
                        'label':str(local[s]),'n_frames':int(e-s),'duration_s':(e-s)/fps})
    return pd.DataFrame(out,columns=['half','start_frame','end_frame','label','n_frames','duration_s'])


def aggregate_diagnostics(labels,half,extra=None,fps=25):
    labels=np.asarray(labels);half=np.asarray(half)
    seg=segments_for_labels(labels,half,fps=fps)
    graded=half!=0
    counts={state:int(np.sum(labels[graded]==state)) for state in ('H','A','D')}
    spells=seg[seg.label.isin(('H','A'))]
    dead=seg[seg.label=='D']
    result={'graded_frames':int(np.sum(graded)), 'frame_counts':counts,
            'possession_spells':len(spells),'dead_segments':len(dead),
            'total_transitions':int(sum(max(0,len(seg[seg.half==h])-1) for h in (1,2))),
            'possession_median_s':float(spells.duration_s.median()) if len(spells) else None,
            'possession_mean_s':float(spells.duration_s.mean()) if len(spells) else None,
            'dead_median_s':float(dead.duration_s.median()) if len(dead) else None}
    if extra: result.update(extra)
    return result


def save_diagnostics(diag,path):
    """Writes aggregate counts, not frame-level private assets."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(diag,indent=2,sort_keys=True,ensure_ascii=False)+'\n',encoding='utf-8')
