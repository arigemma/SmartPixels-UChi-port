#!/usr/bin/env python3
"""Generate two additional similarity-ready atlas PNGs for one clean s08."""

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
    extract_dense_circuit,
    require_s08,
    run_equivalence_suite,
)
from visualizingLGN.similarity_atlas import (
    assert_matches_reference_compactor,
    build_atlas_layout,
    build_compact_with_provenance,
    plot_compacted_similarity_atlas,
    plot_regular_similarity_atlas,
    save_atlas_layout,
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
            "Generate regular and compacted one-pixel-per-gate similarity "
            "atlases for one clean s08 checkpoint. No model comparison occurs."
        )
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/LGN_visualization/s08_clean"),
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

    print(f"[LOAD] {args.checkpoint}", flush=True)
    model = load_registered_model(args.checkpoint, map_location="cpu")
    circuit, metadata = extract_dense_circuit(model)
    require_s08(circuit)
    print("[MODEL] verified s08_p1202M_tau40 architecture", flush=True)

    provenance = build_compact_with_provenance(circuit)
    assert_matches_reference_compactor(circuit, provenance)
    print(
        "[COMPACTION] provenance netlist exactly matches the existing compactor",
        flush=True,
    )
    equivalence = run_equivalence_suite(
        circuit,
        provenance.circuit,
        model=None if args.skip_checkpoint_forward_check else model,
        random_cases=args.random_cases,
        seed=args.seed,
        feature_bits=_feature_bits(metadata, circuit.input_width),
    )
    print(
        f"[EQUIVALENCE PASS] {equivalence['total_cases']} cases",
        flush=True,
    )

    layout = build_atlas_layout(circuit, provenance)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    regular_png = args.output_dir / "regular_similarity_atlas.png"
    compact_png = args.output_dir / "compacted_similarity_atlas.png"
    layout_path = args.output_dir / "similarity_atlas_layout.npz"
    plot_regular_similarity_atlas(circuit, layout, regular_png)
    plot_compacted_similarity_atlas(circuit, provenance, layout, compact_png)
    save_atlas_layout(layout_path, layout)

    physical_by_layer = [
        int((provenance.circuit.node_layers == index).sum())
        for index in range(len(circuit.layers))
    ]
    summary = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "model": metadata,
        "scope": "one clean s08 checkpoint; no cross-model comparison",
        "required_plots": [regular_png.name, compact_png.name],
        "layout": {
            "coordinate_definition": (
                "baseline structural order by downstream class influence, "
                "compact reachability, ordered A/B connections, original index"
            ),
            "saved_layout": layout_path.name,
            "layer_grid_shapes": [list(shape) for shape in layout.shapes],
            "regular_gate_count": circuit.raw_gate_count,
            "one_pixel_per_regular_gate": True,
        },
        "compacted": {
            "physical_gate_count": provenance.circuit.physical_gate_count,
            "physical_gates_by_source_layer": physical_by_layer,
            "removed_or_absorbed_gate_count": (
                circuit.raw_gate_count - provenance.circuit.physical_gate_count
            ),
            "projected_to_regular_gate_coordinates": True,
            "matches_existing_compactor": True,
        },
        "equivalence": equivalence,
    }
    summary_path = args.output_dir / "similarity_atlas_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[DONE] {regular_png}", flush=True)
    print(f"[DONE] {compact_png}", flush=True)
    print(f"[DONE] {layout_path}", flush=True)
    print(f"[DONE] {summary_path}", flush=True)


if __name__ == "__main__":
    main()
