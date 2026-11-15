"""Run possession inference; storage settings live separately from model settings."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import yaml
from src.pipeline import run_pipeline


def main(argv=None):
    parser = argparse.ArgumentParser(description='BAR-ATH offline pipeline with CSV/JSON run artifacts.')
    parser.add_argument('--config', default='config/d1.yaml', help='Problem/model YAML')
    parser.add_argument('--artifacts-config', default='config/artifacts.yaml', help='Storage-only YAML')
    parser.add_argument('--data-dir', default=None)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--video', action='store_true')
    group.add_argument('--no-video', action='store_true')
    args = parser.parse_args(argv)
    project_dir = Path(__file__).resolve().parent
    def resolve(value):
        path = Path(value).expanduser()
        return path if path.is_absolute() else project_dir / path
    config_path = resolve(args.config)
    storage_path = resolve(args.artifacts_config)
    for path in (config_path, storage_path):
        if not path.is_file():
            parser.error(f'Config not found: {path}')
    cfg = yaml.safe_load(config_path.read_text(encoding='utf-8-sig'))
    if not isinstance(cfg, dict):
        parser.error('Model config must be a YAML mapping')
    override = True if args.video else (False if args.no_video else None)
    result = run_pipeline(cfg, project_dir=project_dir, data_dir=args.data_dir, enable_video=override,
                          storage_config_path=storage_path,
                          execution_metadata={'model_config_source': str(config_path), 'cli_arguments': vars(args)})
    print(json.dumps({'run_id': result.run_id, 'artifact_dir': result.artifact_dir,
                      'submission': result.output, 'diagnostics': result.diagnostics}, indent=2, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
