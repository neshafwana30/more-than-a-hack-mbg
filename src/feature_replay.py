"""Verified feature reuse for isolated experiments; no prediction/GT reuse as evidence."""
from pathlib import Path
import hashlib
from .artifacts import read_json,read_table
from .features import FeatureResult


def load_feature_replay(folder,config):
    folder=Path(folder);m=read_json(folder)
    if m['status']!='completed':raise ValueError('Feature source must be completed')
    original=read_json(folder,'config/model.json')
    for key in ('game','source','fps','pitch'):
        current=list(config[key]) if key=='pitch' else config[key]
        saved=list(original[key]) if key=='pitch' else original[key]
        if current!=saved:raise ValueError(f'Feature replay mismatch: {key}')
    if config.get('video',{}).get('enabled',False) or config.get('diagnostics',{}).get('player_acquisition',{}).get('enabled',False):
        raise ValueError('Feature replay without player arrays cannot enable video/player association audit')
    paths=['features/initial/frames.csv','features/initial/events.csv','features/initial/slots.csv']
    for name in paths:
        if name not in m['files'] or hashlib.sha256((folder/name).read_bytes()).hexdigest()!=m['files'][name]['sha256']:
            raise ValueError(f'Feature checksum mismatch: {name}')
    tables=[read_table(folder,p) for p in paths]
    return FeatureResult(*tables,player_positions=None),{'run_id':m['run_id'],'path':str(folder),
        'files':{p:m['files'][p]['sha256'] for p in paths},
        'scope':'verified initial features/events/slots; model evidence/decoding/submission recomputed; player arrays not loaded'}
