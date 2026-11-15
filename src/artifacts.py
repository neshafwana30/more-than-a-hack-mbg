"""CSV/JSON run artifacts and read-only notebook helpers; no inference logic."""
from __future__ import annotations
from dataclasses import fields, is_dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import hashlib
import json
import math
import time
import uuid
import numpy as np
import pandas as pd
import yaml

PROJECT_DIR = Path(__file__).resolve().parents[1]
NULL_TOKEN = '__NULL__'


def _now():
    return datetime.now(ZoneInfo('Asia/Bangkok')).isoformat()


def _json_value(value):
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if isinstance(value, (Path, pd.Timestamp, pd.Timedelta)):
        return str(value)
    return value


def load_storage_config(project_dir=None, config=None, config_path=None):
    root = Path(project_dir or PROJECT_DIR).resolve()
    if config is not None and config_path is not None:
        raise ValueError('Pass storage_config or storage_config_path, not both')
    source = None
    if config is None:
        source = Path(config_path or 'config/artifacts.yaml').expanduser()
        if not source.is_absolute():
            source = root / source
        config = yaml.safe_load(source.read_text(encoding='utf-8-sig'))
    if not isinstance(config, dict):
        raise ValueError('Storage config must be a YAML mapping')
    unknown = set(config) - {'root_dir', 'submission_filename'}
    if unknown:
        raise ValueError(f'Unknown storage settings: {sorted(unknown)}')
    effective = {'root_dir': 'artifacts', 'submission_filename': 'baseline_v2.csv', **config}
    filename = effective['submission_filename']
    if not isinstance(filename, str) or Path(filename).name != filename or '/' in filename or '\\' in filename or not filename.endswith('.csv'):
        raise ValueError('submission_filename must be a CSV filename without directories')
    if not isinstance(effective['root_dir'], str) or not effective['root_dir'].strip():
        raise ValueError('root_dir must be a nonempty path string')
    output_root = Path(effective['root_dir']).expanduser()
    if not output_root.is_absolute():
        output_root = root / output_root
    effective['resolved_root_dir'] = str(output_root.resolve())
    effective['source_file'] = str(source.resolve()) if source else None
    return effective


class ArtifactRun:
    """One non-overwriting run. Save results immediately so failures stay inspectable."""
    def __init__(self, model_config, *, project_dir=None, storage_config=None,
                 storage_config_path=None, entrypoint='run_pipeline', execution=None):
        self.project_dir = Path(project_dir or PROJECT_DIR).resolve()
        self.storage = load_storage_config(self.project_dir, storage_config, storage_config_path)
        if self.storage['source_file'] is None and execution:
            self.storage['source_file'] = execution.get('storage_config_source')
        output_root = Path(self.storage['resolved_root_dir'])
        output_root.mkdir(parents=True, exist_ok=True)
        while True:
            self.run_id = datetime.now(ZoneInfo('Asia/Bangkok')).strftime('%Y%m%d_%H%M%S_%f') + '_' + uuid.uuid4().hex[:8]
            self.path = output_root / self.run_id
            try:
                self.path.mkdir(exist_ok=False)
                break
            except FileExistsError:
                continue
        source_paths = [*sorted((self.project_dir / 'src').glob('*.py')),
                        self.project_dir / 'games.py', self.project_dir / 'run_pipeline.py']
        self.manifest = {'schema_version': 1, 'run_id': self.run_id, 'status': 'running',
                         'entrypoint': entrypoint, 'started_at': _now(), 'finished_at': None,
                         'execution': execution or {}, 'stages': {}, 'files': {},
                         'source_sha256': {str(p.relative_to(self.project_dir)): hashlib.sha256(p.read_bytes()).hexdigest()
                                           for p in source_paths if p.is_file()},
                         'csv_null_token': NULL_TOKEN,
                         'score_semantics': 'uncalibrated utilities, not probabilities'}
        self._flush()
        self.json('config/model.json', model_config)
        self.json('config/storage.json', self.storage)

    def _target(self, relative):
        target = (self.path / relative).resolve()
        if not target.is_relative_to(self.path.resolve()):
            raise ValueError('Artifact path must stay inside its run directory')
        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    @staticmethod
    def _replace(temp, path):
        # Windows readers/indexers can briefly hold a file without delete sharing.
        for attempt in range(7):
            try:
                temp.replace(path)
                return
            except PermissionError:
                if attempt == 6:
                    raise
                time.sleep(0.05 * 2**attempt)

    def _atomic_json(self, path, value):
        temp = path.with_name(path.name + '.tmp')
        temp.write_text(json.dumps(_json_value(value), ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        self._replace(temp, path)

    def _flush(self):
        self._atomic_json(self.path / 'manifest.json', self.manifest)

    def register(self, path, kind, **metadata):
        path = Path(path)
        relative = path.relative_to(self.path).as_posix()
        h = hashlib.sha256()
        with path.open('rb') as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                h.update(chunk)
        self.manifest['files'][relative] = {'format': kind, 'bytes': path.stat().st_size,
                                             'sha256': h.hexdigest(), **metadata}
        self._flush()

    def json(self, relative, value):
        path = self._target(relative)
        self._atomic_json(path, value)
        self.register(path, 'json')

    def csv(self, relative, value):
        path = self._target(relative)
        if not isinstance(value, pd.DataFrame):
            raise TypeError('CSV artifacts must be named DataFrame columns')
        for column in value.select_dtypes(include=['object', 'string']).columns:
            if value[column].eq(NULL_TOKEN).fillna(False).any():
                raise ValueError(f'Literal {NULL_TOKEN} conflicts with CSV missing-value token in {column}')
        temp = path.with_name(path.name + '.tmp')
        value.to_csv(temp, index=False, lineterminator='\n', na_rep=NULL_TOKEN)
        self._replace(temp, path)
        self.register(path, 'csv', rows=len(value), columns=list(value.columns),
                      dtypes={str(c): str(t) for c, t in value.dtypes.items()}, null_token=NULL_TOKEN)

    def _positions(self, prefix, positions, frame_ids, slots):
        if positions is None:
            self.json(prefix + '/player_positions.json', None)
            return
        slot_ids = slots.slot.astype(str).tolist()
        columns = {'frame': frame_ids}
        for j, slot in enumerate(slot_ids):
            for coordinate in ('x', 'y', 'valid'):
                columns[f'slot_{slot}_{coordinate}'] = positions[coordinate][:, j]
        self.csv(prefix + '/player_positions.csv', pd.DataFrame(columns))
        self.json(prefix + '/player_positions_metadata.json',
                  {'slot_order': slot_ids, 'sides': positions['sides'],
                   'array_shape': list(positions['x'].shape),
                   'layout': 'frame plus slot_{slot}_{x,y,valid}; order from slot_order'})

    def save(self, prefix, value, frame_ids=None, name='result', owner=None):
        """Serialize every returned dataclass field, including separate arrays and metadata."""
        if is_dataclass(value) and not isinstance(value, type):
            if hasattr(value, 'frames') and 'frame' in value.frames:
                frame_ids = value.frames.frame.to_numpy()
            for field in fields(value):
                item = getattr(value, field.name)
                if field.name == 'player_positions':
                    self._positions(prefix, item, frame_ids, value.slots)
                else:
                    self.save(prefix, item, frame_ids, field.name, type(value).__name__)
        elif isinstance(value, pd.DataFrame):
            self.csv(f'{prefix}/{name}.csv', value)
        elif isinstance(value, np.ndarray):
            columns = {'ControlResult': {'scores': ['H', 'A']},
                       'EvidenceBundle': {'state_log_scores': ['H', 'A', 'D']}}.get(owner, {}).get(name)
            if value.ndim == 1:
                columns = ['label' if name == 'labels' else name]
            if value.ndim not in (1, 2):
                raise ValueError(f'No CSV layout defined for {prefix}/{name} shape {value.shape}')
            if columns is None:
                raise ValueError(f'Array columns must be explicitly named: {prefix}/{name}')
            df = pd.DataFrame(value, columns=columns)
            if frame_ids is not None:
                if len(frame_ids) != len(df):
                    raise ValueError(f'Frame count mismatch in {prefix}/{name}')
                df.insert(0, 'frame', frame_ids)
            self.csv(f'{prefix}/{name}.csv', df)
        else:
            self.json(f'{prefix}/{name}.json', value)

    def capture(self, stage, function, *args, frame_ids=None, **kwargs):
        self.manifest['stages'][stage] = {'status': 'running', 'started_at': _now()}
        self._flush()
        try:
            result = function(*args, **kwargs)
            self.save(stage, result, frame_ids)
        except Exception as error:
            self.manifest['stages'][stage].update(status='failed', finished_at=_now(),
                                                  error={'type': type(error).__name__, 'message': str(error)})
            self._flush()
            raise
        self.manifest['stages'][stage].update(status='completed', finished_at=_now())
        self._flush()
        return result

    def skipped(self, stage, reason, result=None):
        if result is not None:
            self.save(stage, result)
        self.manifest['stages'][stage] = {'status': 'skipped', 'reason': reason, 'finished_at': _now()}
        self._flush()

    def __enter__(self):
        return self

    def __exit__(self, error_type, error, traceback):
        self.manifest.update(status='failed' if error else 'completed', finished_at=_now())
        if error:
            self.manifest['error'] = {'type': error_type.__name__, 'message': str(error)}
        self._flush()
        return False


def list_runs(root_dir=None, *, project_dir=None, status='completed'):
    """Notebook discovery uses the same storage YAML as the pipeline."""
    if root_dir is None:
        root_dir = load_storage_config(project_dir)['resolved_root_dir']
    runs = []
    for path in Path(root_dir).glob('*/manifest.json'):
        manifest = json.loads(path.read_text(encoding='utf-8'))
        if status is None or manifest['status'] == status:
            runs.append({**{k: manifest.get(k) for k in ('run_id', 'status', 'entrypoint', 'started_at', 'finished_at')},
                         'path': str(path.parent)})
    return sorted(runs, key=lambda r: r['started_at'])


def read_json(run_dir, relative='manifest.json'):
    return json.loads((Path(run_dir) / relative).read_text(encoding='utf-8'))


def read_table(run_dir, relative):
    entry = read_json(run_dir)['files'][relative]
    if entry['format'] != 'csv':
        raise ValueError(f'Not a CSV artifact: {relative}')
    # Preserve literal N/A and empty strings; only the explicit token represents null.
    types = entry.get('dtypes', {})
    dtype = {c: t for c, t in types.items()
             if t in ('object', 'str', 'string', 'bool', 'boolean', 'int8', 'int16', 'int32', 'int64',
                      'uint8', 'uint16', 'uint32', 'uint64', 'Int64', 'Float64', 'float32', 'float64')}
    precision = {'float_precision': 'round_trip'} if int(pd.__version__.split('.')[0]) < 3 else {}
    result = pd.read_csv(Path(run_dir) / relative, keep_default_na=False, dtype=dtype,
                         na_values=[entry.get('null_token', NULL_TOKEN)], **precision)
    for column, kind in types.items():
        if kind.startswith('timedelta'):
            result[column] = pd.to_timedelta(result[column])
        elif kind.startswith('datetime'):
            result[column] = pd.to_datetime(result[column])
    return result
