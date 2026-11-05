"""Command-line entry point for the implemented v2 offline pipeline."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import yaml
from src.pipeline import run_pipeline


def main(argv=None):
    parser=argparse.ArgumentParser(description='BAR-ATH v2 possession H/A/D offline inference; no GT tuning.')
    parser.add_argument('--config',default='config/v2.yaml')
    parser.add_argument('--output',default='outputs/baseline_v2.csv')
    parser.add_argument('--data-dir',default=None,help='Optional local data bundle root; defaults to games.py / MTAH_DATA_DIR')
    group=parser.add_mutually_exclusive_group()
    group.add_argument('--video',action='store_true',help='Enable local detector if configured')
    group.add_argument('--no-video',action='store_true',help='Disable optional video')
    args=parser.parse_args(argv)
    project_dir=Path(__file__).resolve().parent
    path=Path(args.config).expanduser()
    if not path.is_absolute():path=project_dir/path
    output=Path(args.output).expanduser()
    if not output.is_absolute():output=project_dir/output
    if not path.is_file():parser.error(f'Config not found: {path}')
    cfg=yaml.safe_load(path.read_text(encoding='utf-8'))
    if not isinstance(cfg,dict):parser.error('Config must be a YAML mapping')
    override=True if args.video else (False if args.no_video else None)
    result=run_pipeline(cfg,output,project_dir,data_dir=args.data_dir,enable_video=override)
    print(json.dumps({'submission':result.output,'diagnostics':result.diagnostics},indent=2,ensure_ascii=False))
    return 0

if __name__=='__main__':raise SystemExit(main())
