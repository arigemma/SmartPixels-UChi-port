from pathlib import Path

import numpy as np

from visualizingLGN.circuit import (
    DenseCircuit,
    LayerNetlist,
    build_compact_circuit,
    compact_summary,
    regular_summary,
)
from visualizingLGN.paper_overviews import render_paper_overview_set


def _small_circuit() -> DenseCircuit:
    first = LayerNetlist(
        lut_ids=np.asarray([1, 6, 15, 0, 3, 5, 12, 1, 7]),
        connections=np.asarray(
            [[0, 1, 0, 1, 2, 3, 0, 2, 0], [1, 2, 1, 2, 3, 0, 1, 2, 3]]
        ),
    )
    second = LayerNetlist(
        lut_ids=np.asarray([1, 3, 6, 14, 5, 9]),
        connections=np.asarray([[0, 4, 2, 6, 5, 3], [2, 1, 3, 0, 7, 3]]),
    )
    return DenseCircuit(4, (first, second), num_classes=3, tau=2.0)


def test_non_atlas_panels_are_separate_png_and_pdf_files(tmp_path: Path) -> None:
    circuit = _small_circuit()
    compact = build_compact_circuit(circuit)
    outputs = render_paper_overview_set(
        circuit,
        compact,
        regular_summary(circuit),
        compact_summary(compact),
        tmp_path,
        formats=("png", "pdf"),
        dpi=72,
    )
    subjects = {
        "regular/regular_pipeline",
        "regular/regular_gate_counts",
        "regular/regular_connectivity",
        "regular/regular_lut_legend",
        "compacted/compacted_pipeline",
        "compacted/compacted_gate_counts",
        "compacted/compacted_reachable_connections",
        "compacted/compacted_lut_legend",
    }
    assert {str(path.relative_to(tmp_path).with_suffix("")) for path in outputs} == subjects
    assert len(outputs) == 16
    for path in outputs:
        header = path.read_bytes()[:8]
        if path.suffix == ".png":
            assert header == b"\x89PNG\r\n\x1a\n"
        else:
            assert header.startswith(b"%PDF")
