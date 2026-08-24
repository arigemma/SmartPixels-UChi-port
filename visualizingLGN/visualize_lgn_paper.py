#!/usr/bin/env python3
"""Generate large standalone paper plots for a dense LGN circuit overview."""

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
    require_s02,
    require_s08,
    run_equivalence_suite,
)
from visualizingLGN.paper_overviews import (
    SUPPORTED_FORMATS,
    render_paper_overview_set,
)


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
            "Split regular and compacted non-atlas dense-LGN circuit overviews "
            "into large-text standalone PNG/PDF paper figures."
        )
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/LGN_visualization/s08_clean/paper_overviews"),
    )
    architecture = parser.add_mutually_exclusive_group()
    architecture.add_argument(
        "--require-s02",
        action="store_true",
        help=(
            "Reject checkpoints other than the tau-20 s02 dense LGN used by "
            "the selected input-bit-flip study."
        ),
    )
    architecture.add_argument(
        "--require-s08",
        action="store_true",
        help="Reject checkpoints other than the tau-40 s08 dense LGN.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=SUPPORTED_FORMATS,
        default=list(SUPPORTED_FORMATS),
        help="Figure formats to write (default: png pdf).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="PNG and PDF embedded-image resolution (default: 300).",
    )
    parser.add_argument(
        "--font-scale",
        type=float,
        default=1.0,
        help="Multiply all already-large paper font sizes (default: 1.0).",
    )
    parser.add_argument("--random-cases", type=int, default=32)
    parser.add_argument("--seed", type=int, default=808)
    parser.add_argument(
        "--skip-checkpoint-forward-check",
        action="store_true",
        help="Skip only the additional extracted-netlist vs checkpoint check.",
    )
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint does not exist: {args.checkpoint}")
    if args.random_cases < 1:
        parser.error("--random-cases must be positive")
    if args.dpi < 72:
        parser.error("--dpi must be at least 72")
    if args.font_scale <= 0:
        parser.error("--font-scale must be positive")

    print(f"[LOAD] {args.checkpoint}", flush=True)
    model = load_registered_model(args.checkpoint, map_location="cpu")
    circuit, model_metadata = extract_dense_circuit(model)
    if args.require_s02:
        require_s02(circuit)
        print("[MODEL] verified s02_p163M_tau20 architecture", flush=True)
    elif args.require_s08:
        require_s08(circuit)
        print("[MODEL] verified s08_p1202M_tau40 architecture", flush=True)
    else:
        print(
            f"[MODEL] dense LGN with {len(circuit.layers)} LogicDense layers, "
            f"{circuit.raw_gate_count:,} regular gates, tau={circuit.tau:g}",
            flush=True,
        )

    compact = build_compact_circuit(circuit)
    equivalence = run_equivalence_suite(
        circuit,
        compact,
        model=None if args.skip_checkpoint_forward_check else model,
        random_cases=args.random_cases,
        seed=args.seed,
        feature_bits=_feature_bits(model_metadata, circuit.input_width),
    )
    print(f"[EQUIVALENCE PASS] {equivalence['total_cases']} cases", flush=True)

    regular = regular_summary(circuit)
    compacted = compact_summary(compact)
    outputs = render_paper_overview_set(
        circuit,
        compact,
        regular,
        compacted,
        args.output_dir,
        formats=args.formats,
        dpi=args.dpi,
        font_scale=args.font_scale,
    )
    manifest = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "model": model_metadata,
        "scope": "non-atlas paper-ready regular and compacted overview panels",
        "formats": list(dict.fromkeys(args.formats)),
        "dpi": args.dpi,
        "font_scale": args.font_scale,
        "figure_subjects": 8,
        "files": [str(path.relative_to(args.output_dir)) for path in outputs],
        "regular": regular,
        "compacted": compacted,
        "equivalence": equivalence,
    }
    manifest_path = args.output_dir / "paper_overview_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[DONE] {len(outputs)} figure files in {args.output_dir}", flush=True)
    print(f"[DONE] {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
