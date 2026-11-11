#!/usr/bin/env python
"""Run one MAIA stage using the values in maia_params.yaml.

Usage (from anywhere; paths come from the MAIA_* env vars in set_paths.sh):
    python maia_run.py prepare
    python maia_run.py audit
    python maia_run.py train --task-index 7 --profile pilot --device cuda
    python maia_run.py train --task-index 7 --profile smoke --device cpu --dry-run

It only builds the same `python -m maiaStudies.<stage>` command the repo's
.slurm files use.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent  # SmartPixels/, where `python -m maiaStudies...` resolves
PARAMS = HERE / "maia_params.yaml"


def need(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        sys.exit(f"{name} is not set. Run: source {HERE / 'set_paths.sh'}")
    return value


def flags(pairs):
    out = []
    for key, value in pairs:
        out += [f"--{key.replace('_', '-')}", str(value)]
    return out


def task_alias(campaign: str, index: int) -> str:
    return subprocess.run(
        [sys.executable, "-m", "maiaStudies.campaign", "--campaign", campaign,
         "--task-index", str(index)],
        cwd=REPO_ROOT, check=True, capture_output=True, text=True,
    ).stdout.strip()


def build(args, cfg):
    py = [sys.executable, "-m"]
    if args.stage == "prepare":
        p = cfg["prepare"]
        return py + ["maiaStudies.prepare"] + flags([
            ("data_root", need("MAIA_DATA_ROOT")),
            ("cache_dir", need("MAIA_CACHE_ROOT")),
            ("timing_column", p["timing_column"]),
            ("batch_size", p["batch_size"]),
            ("thresholds", p["thresholds"]),
            ("threshold_sample_rows", p["threshold_sample_rows"]),
            ("seed", p["seed"]),
        ])
    if args.stage == "audit":
        a = cfg["audit"]
        return py + ["maiaStudies.audit_cache"] + flags([
            ("cache_dir", need("MAIA_CACHE_ROOT")),
            ("maximum_per_class", a["maximum_per_class"]),
            ("minimum_baseline_auc", a["minimum_baseline_auc"]),
        ])
    if args.stage == "pilot_gate":
        return py + ["maiaStudies.pilot_gate"] + flags([("result_root", need("MAIA_RESULT_ROOT"))])
    t = cfg["train"]
    if args.task_index is None:
        sys.exit("train needs --task-index")
    alias = task_alias(t["campaign"], args.task_index)
    family = "dense" if alias.startswith("dense") else "conv"
    if args.stage == "evaluate":
        e = cfg["evaluate"]
        out_dir = args.output or str(Path(need("MAIA_RESULT_ROOT")) / alias)
        return py + ["maiaStudies.evaluate"] + flags([
            ("campaign", t["campaign"]),
            ("task_index", args.task_index),
            ("cache_dir", need("MAIA_CACHE_ROOT")),
            ("model_dir", out_dir),
            ("device", args.device),
            ("batch_size", e["batch_size"][family]),
            ("num_workers", e["num_workers"]),
            ("timing_min_ns", e["timing_min_ns"]),
            ("timing_max_ns", e["timing_max_ns"]),
            ("maximum_validation_examples", e["maximum_validation_examples"]),
            ("maximum_test_examples", e["maximum_test_examples"]),
        ])
    run = t["profiles"][args.profile]
    out_dir = args.output or str(Path(need("MAIA_RESULT_ROOT")) / alias)
    cmd = py + ["maiaStudies.train"] + flags([
        ("campaign", t["campaign"]),
        ("task_index", args.task_index),
        ("cache_dir", need("MAIA_CACHE_ROOT")),
        ("output", out_dir),
        ("device", args.device),
        ("batch_size", t["batch_size"][family]),
        ("num_workers", t["num_workers"]),
        ("learning_rate", t["learning_rate"]),
        ("weight_decay", t["weight_decay"]),
        ("minimum_steps", run["minimum_steps"]),
        ("minimum_training_hours", run["minimum_training_hours"]),
        ("eval_frequency", run["eval_frequency"]),
        ("validation_examples", run["validation_examples"]),
        ("seed", t["seed"]),
    ])
    return cmd


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=["prepare", "audit", "train", "evaluate", "pilot_gate"])
    ap.add_argument("--task-index", type=int)
    ap.add_argument("--profile", default="pilot", choices=["full", "pilot", "smoke"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output", help="override the result directory")
    ap.add_argument("--dry-run", action="store_true", help="print the command only")
    args = ap.parse_args()

    cfg = yaml.safe_load(PARAMS.read_text())
    cmd = build(args, cfg)
    print("[maia_run] " + " ".join(cmd), flush=True)
    if args.dry_run:
        return
    sys.exit(subprocess.call(cmd, cwd=REPO_ROOT))


if __name__ == "__main__":
    main()
