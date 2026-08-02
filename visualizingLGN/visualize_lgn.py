#!/usr/bin/env python3
"""Generate regular/compacted LGN circuit PNGs and prove equivalence."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models import classification as _classification  # noqa: F401
from models.registry import load_registered_model
from visualizingLGN.circuit import (
    build_compact_circuit,
    compact_summary,
    extract_dense_circuit,
    regular_summary,
    require_s08,
    run_equivalence_suite,
    save_compact_netlist,
    save_regular_netlist,
)
from visualizingLGN.plotting import plot_compact_circuit, plot_regular_circuit


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _feature_bits(metadata: dict, input_width: int) -> tuple[int, int] | None:
    shape = metadata.get("threshold_shape")
    if shape and len(shape) == 2 and int(shape[0]) * int(shape[1]) == input_width:
        return int(shape[0]), int(shape[1])
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize a discrete sequential dense TorchLogix LGN before and "
            "after exact Boolean compaction, then verify equal outputs."
        )
    )
    parser.add_argument("checkpoint", type=Path, help="Existing clean LGN .pth file")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/LGN_visualization/s08"),
    )
    parser.add_argument(
        "--require-s08",
        action="store_true",
        help="Reject any model except s08_p1202M_tau40.",
    )
    parser.add_argument("--random-cases", type=int, default=32)
    parser.add_argument("--seed", type=int, default=808)
    parser.add_argument(
        "--skip-checkpoint-forward-check",
        action="store_true",
        help=(
            "Still prove regular-vs-compact equality, but skip the additional "
            "check against the loaded TorchLogix layers."
        ),
    )
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint does not exist: {args.checkpoint}")
    if args.random_cases < 1:
        parser.error("--random-cases must be positive")

    print(f"[LOAD] {args.checkpoint}", flush=True)
    model = load_registered_model(args.checkpoint, map_location="cpu")
    circuit, model_metadata = extract_dense_circuit(model)
    if args.require_s08:
        require_s08(circuit)
        print("[MODEL] verified s08_p1202M_tau40 architecture", flush=True)

    print(
        f"[EXTRACT] {len(circuit.layers)} layers, "
        f"{circuit.raw_gate_count:,} regular gates",
        flush=True,
    )
    compact = build_compact_circuit(circuit)
    print(
        f"[COMPACT] {compact.physical_gate_count:,} reachable physical gates",
        flush=True,
    )

    equivalence = run_equivalence_suite(
        circuit,
        compact,
        model=None if args.skip_checkpoint_forward_check else model,
        random_cases=args.random_cases,
        seed=args.seed,
        feature_bits=_feature_bits(model_metadata, circuit.input_width),
    )
    print(
        f"[EQUIVALENCE PASS] {equivalence['total_cases']} edge/structured/random cases",
        flush=True,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    regular = regular_summary(circuit)
    compacted = compact_summary(compact)
    summary = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "model": model_metadata,
        "regular": regular,
        "compacted": compacted,
        "compaction_rules": [
            "constant propagation",
            "passthrough elimination",
            "inverter folding",
            "dead-code elimination from GroupSum roots",
        ],
        "equivalence": equivalence,
    }

    regular_png = args.output_dir / "regular_network.png"
    compact_png = args.output_dir / "compacted_network.png"
    plot_regular_circuit(circuit, regular, regular_png)
    plot_compact_circuit(compact, compacted, compact_png)
    save_regular_netlist(args.output_dir / "regular_netlist.npz", circuit)
    save_compact_netlist(args.output_dir / "compacted_netlist.npz", compact)
    (args.output_dir / "circuit_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(f"[DONE] {regular_png}", flush=True)
    print(f"[DONE] {compact_png}", flush=True)
    print(f"[DONE] {args.output_dir / 'circuit_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
