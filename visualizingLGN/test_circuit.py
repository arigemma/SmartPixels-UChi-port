import itertools
import json
from types import SimpleNamespace

import pytest

import numpy as np

from visualizingLGN.circuit import (
    GATE_NAMES,
    DenseCircuit,
    LayerNetlist,
    build_compact_circuit,
    compact_summary,
    regular_summary,
    require_s02,
    run_equivalence_suite,
)
from visualizingLGN.plotting import plot_compact_circuit, plot_regular_circuit


def _all_binary(width):
    return np.asarray(list(itertools.product((0, 1), repeat=width)), dtype=np.uint8)


def test_all_torchlogix_truth_table_ids():
    circuit = DenseCircuit(
        input_width=2,
        layers=(
            LayerNetlist(
                lut_ids=np.arange(16),
                connections=np.tile(np.asarray([[0], [1]]), (1, 16)),
            ),
        ),
        num_classes=1,
    )
    inputs = _all_binary(2)
    output = circuit.evaluate_bits(inputs)
    expected = np.asarray(
        [[(lut_id >> (3 - (2 * a + b))) & 1 for lut_id in range(16)] for a, b in inputs],
        dtype=np.uint8,
    )
    np.testing.assert_array_equal(output, expected)


def _mixed_circuit():
    # L1 includes AND, dead XOR, constants, passthroughs, NOT, and same-input gates.
    first = LayerNetlist(
        lut_ids=np.asarray([1, 6, 15, 0, 3, 5, 12, 1, 7]),
        connections=np.asarray(
            [
                [0, 1, 0, 1, 2, 3, 0, 2, 0],
                [1, 2, 1, 2, 3, 0, 1, 2, 3],
            ]
        ),
    )
    # Width 6 is divisible by the three-class GroupSum. Several L1 nodes are dead.
    second = LayerNetlist(
        lut_ids=np.asarray([1, 3, 6, 14, 5, 9]),
        connections=np.asarray(
            [
                [0, 4, 2, 6, 5, 3],
                [2, 1, 3, 0, 7, 3],
            ]
        ),
    )
    return DenseCircuit(
        input_width=4,
        layers=(first, second),
        num_classes=3,
        tau=2.0,
    )


def test_compaction_is_exact_exhaustively_and_removes_logic():
    regular = _mixed_circuit()
    compact = build_compact_circuit(regular)
    inputs = _all_binary(regular.input_width)
    np.testing.assert_array_equal(
        regular.evaluate_counts(inputs), compact.evaluate_counts(inputs)
    )
    assert compact.physical_gate_count < regular.raw_gate_count


def test_equivalence_suite_records_edge_structured_and_random_cases():
    regular = _mixed_circuit()
    compact = build_compact_circuit(regular)
    result = run_equivalence_suite(
        regular,
        compact,
        random_cases=11,
        seed=123,
        feature_bits=(2, 2),
    )
    assert result["passed"] is True
    assert result["case_counts"]["edge"] >= 6
    assert result["case_counts"]["structured"] >= 3
    assert result["case_counts"]["random"] == 11


def test_summaries_and_pngs_are_written(tmp_path):
    regular = _mixed_circuit()
    compact = build_compact_circuit(regular)
    regular_data = regular_summary(regular)
    compact_data = compact_summary(compact)
    assert set(regular_data["layers"][0]["gate_counts"]) == set(GATE_NAMES)
    assert compact_data["removed_or_absorbed_gates"] > 0
    json.dumps({"regular": regular_data, "compact": compact_data})

    regular_png = tmp_path / "regular.png"
    compact_png = tmp_path / "compact.png"
    plot_regular_circuit(regular, regular_data, regular_png)
    plot_compact_circuit(compact, compact_data, compact_png)
    assert regular_png.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert compact_png.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_require_s02_checks_widths_classes_and_tau():
    expected = SimpleNamespace(
        input_width=1_400,
        layer_widths=(6_300, 6_300, 6_300, 4_200, 4_200, 4_200, 3_150),
        num_classes=3,
        tau=20.0,
    )
    require_s02(expected)

    wrong_tau = SimpleNamespace(**{**vars(expected), "tau": 40.0})
    with pytest.raises(ValueError, match="tau-20"):
        require_s02(wrong_tau)
