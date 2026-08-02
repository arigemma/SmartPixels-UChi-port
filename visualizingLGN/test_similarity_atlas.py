import numpy as np

from visualizingLGN.circuit import DenseCircuit, LayerNetlist, run_equivalence_suite
from visualizingLGN.similarity_atlas import (
    PADDING_PIXEL,
    REMOVED_PIXEL,
    assert_matches_reference_compactor,
    build_atlas_layout,
    build_compact_with_provenance,
    compact_atlas_grids,
    plot_compacted_similarity_atlas,
    plot_regular_similarity_atlas,
    regular_atlas_grids,
)


def _circuit():
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


def test_provenance_compactor_is_identical_and_equivalent():
    regular = _circuit()
    provenance = build_compact_with_provenance(regular)
    assert_matches_reference_compactor(regular, provenance)
    proof = run_equivalence_suite(
        regular,
        provenance.circuit,
        random_cases=9,
        seed=17,
        feature_bits=(2, 2),
    )
    assert proof["passed"] is True
    assert len(provenance.node_origins) == provenance.circuit.physical_gate_count


def test_layout_is_deterministic_and_each_regular_gate_gets_one_pixel():
    regular = _circuit()
    provenance = build_compact_with_provenance(regular)
    first = build_atlas_layout(regular, provenance)
    second = build_atlas_layout(regular, provenance)
    regular_grids = regular_atlas_grids(regular, first)
    compact_grids = compact_atlas_grids(regular, provenance, first)

    for layer, order_a, order_b, regular_grid, compact_grid in zip(
        regular.layers, first.orders, second.orders, regular_grids, compact_grids
    ):
        np.testing.assert_array_equal(order_a, order_b)
        np.testing.assert_array_equal(np.sort(order_a), np.arange(layer.width))
        assert np.count_nonzero(regular_grid >= 0) == layer.width
        assert np.count_nonzero(regular_grid == PADDING_PIXEL) == (
            regular_grid.size - layer.width
        )
        assert np.count_nonzero(compact_grid == PADDING_PIXEL) == (
            compact_grid.size - layer.width
        )

    physical_pixels = sum(np.count_nonzero(grid >= 0) for grid in compact_grids)
    removed_pixels = sum(
        np.count_nonzero(grid == REMOVED_PIXEL) for grid in compact_grids
    )
    assert physical_pixels == provenance.circuit.physical_gate_count
    assert removed_pixels == regular.raw_gate_count - physical_pixels


def test_exactly_two_atlas_pngs_render(tmp_path):
    regular = _circuit()
    provenance = build_compact_with_provenance(regular)
    layout = build_atlas_layout(regular, provenance)
    regular_png = tmp_path / "regular_similarity_atlas.png"
    compact_png = tmp_path / "compacted_similarity_atlas.png"
    plot_regular_similarity_atlas(regular, layout, regular_png)
    plot_compacted_similarity_atlas(regular, provenance, layout, compact_png)
    assert regular_png.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert compact_png.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert sorted(path.name for path in tmp_path.glob("*.png")) == [
        "compacted_similarity_atlas.png",
        "regular_similarity_atlas.png",
    ]
