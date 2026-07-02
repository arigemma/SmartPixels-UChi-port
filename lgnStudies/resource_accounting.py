"""Analytical resource accounting for sequential dense TorchLogix networks.

The discrete LUTs are simplified with constant propagation, passthrough
elimination, inverter folding, and dead-code elimination.  The GroupSum head is
kept fixed by construction: its cost depends only on the final layer width.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil, log2
from typing import Iterable, Sequence

import numpy as np


CONST_ZERO = -1
CONST_ONE = -2
INPUT_OFFSET = 3


@dataclass(frozen=True)
class ResourceEstimate:
    raw_logic_neurons: int
    reachable_binary_luts: int
    reachable_inverters: int
    simplified_logic_luts: int
    group_sum_luts: int
    total_luts: int
    final_layer_width: int
    neurons_per_class: int
    reachable_luts_by_layer: tuple[int, ...]

    def to_dict(self) -> dict:
        result = asdict(self)
        result["reachable_luts_by_layer"] = list(self.reachable_luts_by_layer)
        return result


def _truth(lut_id: int, a: int, b: int) -> int:
    """Return f(a, b) for TorchLogix's 0..15 LUT numbering."""
    return (int(lut_id) >> (3 - (2 * int(a) + int(b)))) & 1


def _permute_input(lut_id: int, input_index: int) -> int:
    """Absorb an inverted input into a two-input LUT truth table."""
    bits = []
    for a in (0, 1):
        for b in (0, 1):
            old_a = 1 - a if input_index == 0 else a
            old_b = 1 - b if input_index == 1 else b
            bits.append(_truth(lut_id, old_a, old_b))
    result = 0
    for bit in bits:
        result = (result << 1) | bit
    return result


def _group_sum_luts_one_class(n_inputs: int) -> int:
    """Estimate LUTs for a balanced unsigned popcount tree.

    A Xilinx-style carry-chain adder is approximated as one LUT per input bit
    of the wider operand.  This is an analytical estimate, not a synthesis
    report, but unlike parameter count it includes the fixed GroupSum cost.
    """
    if n_inputs < 1:
        raise ValueError("GroupSum needs at least one input per class.")

    maxima = [1] * n_inputs
    cost = 0
    while len(maxima) > 1:
        next_level = []
        for index in range(0, len(maxima), 2):
            if index + 1 == len(maxima):
                next_level.append(maxima[index])
                continue
            left, right = maxima[index], maxima[index + 1]
            left_width = max(1, ceil(log2(left + 1)))
            right_width = max(1, ceil(log2(right + 1)))
            cost += max(left_width, right_width)
            next_level.append(left + right)
        maxima = next_level
    return cost


def estimate_group_sum_luts(final_layer_width: int, num_classes: int) -> int:
    if final_layer_width % num_classes:
        raise ValueError(
            f"Final width {final_layer_width} is not divisible by {num_classes} classes."
        )
    per_class = final_layer_width // num_classes
    return num_classes * _group_sum_luts_one_class(per_class)


def estimate_dense_lgn_resources(
    layer_lut_ids: Sequence[np.ndarray],
    layer_connections: Sequence[np.ndarray],
    input_width: int,
    num_classes: int,
) -> ResourceEstimate:
    """Simplify a sequential dense LGN and estimate its FPGA LUT footprint."""
    if not layer_lut_ids or len(layer_lut_ids) != len(layer_connections):
        raise ValueError("LUT IDs and connection arrays must describe the same non-empty network.")

    # Negative references are constants or primary inputs; non-negative
    # references index the simplified nodes below.
    previous_refs = np.arange(
        -INPUT_OFFSET, -INPUT_OFFSET - input_width, -1, dtype=np.int64
    )
    node_kinds: list[str] = []
    node_children: list[tuple[int, ...]] = []
    node_layers: list[int] = []

    def add_node(kind: str, children: Iterable[int], layer_index: int) -> int:
        node_kinds.append(kind)
        node_children.append(tuple(int(child) for child in children))
        node_layers.append(layer_index)
        return len(node_kinds) - 1

    def invert(ref: int, layer_index: int) -> int:
        if ref == CONST_ZERO:
            return CONST_ONE
        if ref == CONST_ONE:
            return CONST_ZERO
        if ref >= 0 and node_kinds[ref] == "not":
            return node_children[ref][0]
        return add_node("not", (ref,), layer_index)

    def unary(ref: int, output_at_zero: int, output_at_one: int, layer_index: int) -> int:
        if output_at_zero == output_at_one:
            return CONST_ONE if output_at_zero else CONST_ZERO
        if output_at_zero == 0 and output_at_one == 1:
            return ref
        return invert(ref, layer_index)

    def apply_lut(lut_id: int, left: int, right: int, layer_index: int) -> int:
        # An inverter feeding a LUT is free after changing that LUT's truth table.
        if left >= 0 and node_kinds[left] == "not":
            left = node_children[left][0]
            lut_id = _permute_input(lut_id, 0)
        if right >= 0 and node_kinds[right] == "not":
            right = node_children[right][0]
            lut_id = _permute_input(lut_id, 1)

        if left in (CONST_ZERO, CONST_ONE) and right in (CONST_ZERO, CONST_ONE):
            value = _truth(lut_id, left == CONST_ONE, right == CONST_ONE)
            return CONST_ONE if value else CONST_ZERO

        if left == right:
            return unary(
                left, _truth(lut_id, 0, 0), _truth(lut_id, 1, 1), layer_index
            )

        if left in (CONST_ZERO, CONST_ONE):
            a = int(left == CONST_ONE)
            return unary(
                right, _truth(lut_id, a, 0), _truth(lut_id, a, 1), layer_index
            )

        if right in (CONST_ZERO, CONST_ONE):
            b = int(right == CONST_ONE)
            return unary(
                left, _truth(lut_id, 0, b), _truth(lut_id, 1, b), layer_index
            )

        independent_of_right = all(
            _truth(lut_id, a, 0) == _truth(lut_id, a, 1) for a in (0, 1)
        )
        if independent_of_right:
            return unary(
                left, _truth(lut_id, 0, 0), _truth(lut_id, 1, 0), layer_index
            )

        independent_of_left = all(
            _truth(lut_id, 0, b) == _truth(lut_id, 1, b) for b in (0, 1)
        )
        if independent_of_left:
            return unary(
                right, _truth(lut_id, 0, 0), _truth(lut_id, 0, 1), layer_index
            )

        return add_node("binary", (left, right), layer_index)

    raw_logic_neurons = 0
    for layer_index, (ids_raw, connections_raw) in enumerate(
        zip(layer_lut_ids, layer_connections)
    ):
        ids = np.asarray(ids_raw, dtype=np.int64).reshape(-1)
        connections = np.asarray(connections_raw, dtype=np.int64)
        if connections.shape != (2, len(ids)):
            raise ValueError(
                f"Layer {layer_index}: expected connections shape (2, {len(ids)}), "
                f"got {connections.shape}."
            )
        if connections.min() < 0 or connections.max() >= len(previous_refs):
            raise ValueError(f"Layer {layer_index}: connection index outside prior layer.")

        raw_logic_neurons += len(ids)
        current_refs = np.empty(len(ids), dtype=np.int64)
        for neuron, lut_id in enumerate(ids):
            left = int(previous_refs[connections[0, neuron]])
            right = int(previous_refs[connections[1, neuron]])
            current_refs[neuron] = apply_lut(
                int(lut_id), left, right, layer_index
            )
        previous_refs = current_refs

    # GroupSum consumes every final output, so all final references are roots.
    stack = [int(ref) for ref in previous_refs if ref >= 0]
    reachable: set[int] = set()
    while stack:
        node = stack.pop()
        if node in reachable:
            continue
        reachable.add(node)
        stack.extend(child for child in node_children[node] if child >= 0)

    binary_luts = sum(node_kinds[node] == "binary" for node in reachable)
    inverters = sum(node_kinds[node] == "not" for node in reachable)
    by_layer = [0] * len(layer_lut_ids)
    for node in reachable:
        by_layer[node_layers[node]] += 1

    final_width = len(previous_refs)
    group_sum_luts = estimate_group_sum_luts(final_width, num_classes)
    simplified_logic_luts = binary_luts + inverters
    return ResourceEstimate(
        raw_logic_neurons=raw_logic_neurons,
        reachable_binary_luts=binary_luts,
        reachable_inverters=inverters,
        simplified_logic_luts=simplified_logic_luts,
        group_sum_luts=group_sum_luts,
        total_luts=simplified_logic_luts + group_sum_luts,
        final_layer_width=final_width,
        neurons_per_class=final_width // num_classes,
        reachable_luts_by_layer=tuple(by_layer),
    )


def estimate_model_resources(model, num_classes: int = 3) -> ResourceEstimate:
    """Extract LUTs/connections from the model-2 dense LGN implementation."""
    layers = getattr(model, "layers", None)
    if layers is None:
        raise TypeError("Expected a model with an ordered `layers` collection.")

    logic_layers = [
        layer for layer in layers if type(layer).__name__ == "LogicDense"
    ]
    if not logic_layers:
        raise TypeError("No LogicDense layers found.")

    ids = []
    connections = []
    for layer in logic_layers:
        if getattr(layer, "lut_rank", None) != 2:
            raise ValueError("This resource study supports only rank-2 dense LUTs.")
        _, layer_ids = layer.get_luts_and_ids()
        ids.append(layer_ids.detach().cpu().numpy())
        indices = layer.connections.indices
        if indices.ndim != 2:
            raise ValueError("Only fixed dense connections are supported.")
        connections.append(indices.detach().cpu().numpy())

    return estimate_dense_lgn_resources(
        layer_lut_ids=ids,
        layer_connections=connections,
        input_width=int(logic_layers[0].in_dim),
        num_classes=num_classes,
    )
