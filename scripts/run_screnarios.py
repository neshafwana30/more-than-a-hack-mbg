"""Skenario config-only di match penuh. TIDAK memakai ground truth.
    python scripts/run_scenarios.py
    python scripts/run_scenarios.py --only shape_clip_calibrated no_shape
"""
import argparse
import copy
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

FPS = 25


def find_root() -> Path:
    here = Path.cwd().resolve()
    for p in [here, *here.parents]:
        if (p / "main.py").is_file():
            return p
    raise SystemExit("main.py tidak ditemukan dari folder kerja.")


ROOT = find_root()
OUT = ROOT / "outputs" / "scenarios"
OUT.mkdir(parents=True, exist_ok=True)

ap = argparse.ArgumentParser()
ap.add_argument("--file", default=str(ROOT / "scripts" / "scenarios.yaml"))
ap.add_argument("--only", nargs="*")
args = ap.parse_args()

base_cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
loaded = yaml.safe_load(Path(args.file).read_text(encoding="utf-8"))
if args.only:
    loaded = {k: v for k, v in loaded.items() if k in args.only}
scenarios = {"base": {"why": "config.yaml saat ini", "set": {}}}
scenarios.update({k: v for k, v in loaded.items() if k != "base"})


def set_path(cfg, dotted, value):
    keys = dotted.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node[k]
    if keys[-1] not in node:
        raise KeyError(f"kunci tidak ada di config: {dotted}")
    node[keys[-1]] = value


def run(name, cfg):
    cp, op = OUT / f"{name}.yaml", OUT / f"{name}.csv"
    cp.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    r = subprocess.run([sys.executable, "main.py", "--config", str(cp), "--output", str(op)],
                       cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-500:], r.stderr[-1500:])
        raise SystemExit(f"main.py gagal untuk skenario '{name}'")
    shape = next((l.split(":")[-1].strip() for l in r.stdout.splitlines()
                  if l.startswith("team-shape")), "-")
    lab = pd.read_csv(op)["label"].astype(str).str.strip().str.upper().to_numpy()
    return lab, shape


def durations(lab):
    out, prev, n = [], None, 0
    for x in lab:
        if x in ("H", "A") and x == prev:
            n += 1
            continue
        if prev is not None:
            out.append(n)
        prev, n = (x, 1) if x in ("H", "A") else (None, 0)
    if prev is not None:
        out.append(n)
    return np.array(out) / FPS


rows, base_lab = [], None
for name, spec in scenarios.items():
    cfg = copy.deepcopy(base_cfg)
    for dotted, value in (spec.get("set") or {}).items():
        set_path(cfg, dotted, value)
    lab, shape = run(name, cfg)
    if base_lab is None:
        base_lab = lab
    d = durations(lab)
    graded = base_lab != "X"
    live = (lab == "H") | (lab == "A")
    rows.append({
        "skenario": name,
        "poss": len(d),
        "rata_s": round(float(d.mean()), 1),
        "<1s": int((d < 1).sum()),
        "<2s": int((d < 2).sum()),
        "D%": round(100 * float((lab == "D").sum() / max((lab != "X").sum(), 1)), 1),
        "H%live": round(100 * float((lab == "H").sum() / max(live.sum(), 1)), 1),
        "berubah%": round(100 * float((lab[graded] != base_lab[graded]).mean()), 2),
        "shape_cocok": shape,
    })
    print(f"selesai: {name}", flush=True)

print()
print(pd.DataFrame(rows).to_string(index=False))