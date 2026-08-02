"""Pure NumPy netlists and exact simplification for rank-2 dense LGNs.

The compacting rules intentionally match the rules used by
``lgnStudies.resource_accounting``: constant propagation, passthrough
elimination, inverter folding, and backwards dead-code elimination.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


CONST_ZERO = -1
CONST_ONE = -2
INPUT_OFFSET = 3
NOT_KIND = 1
BINARY_KIND = 2

GATE_NAMES = (
    "FALSE",
    "AND",
    "A & !B",
    "A",
    "!A & B",
    "B",
    "XOR",
    "OR",
    "NOR",
    "XNOR",
    "!B",
    "A | !B",
    "!A",
    "!A | B",
    "NAND",
    "TRUE",
)

# A stable, high-contrast palette derived from Matplotlib's open-source
# categorical palettes. Every truth table keeps the same color in both plots.
GATE_COLORS = (
    "#4d4d4d",
    "#1f77b4",
    "#aec7e8",
    "#ff7f0e",
    "#ffbb78",
    "#2ca02c",
    "#98df8a",
    "#d62728",
    "#ff9896",
    "#9467bd",
    "#c5b0d5",
    "#8c564b",
    "#c49c94",
    "#e377c2",
    "#17becf",
    "#bcbd22",
)


def _truth(lut_id: int, a: int, b: int) -> int:
    """Evaluate one TorchLogix rank-2 LUT truth-table entry."""
    return (int(lut_id) >> (3 - (2 * int(a) + int(b)))) & 1


def _permute_input(lut_id: int, input_index: int) -> int:
    """Absorb an inverted input into a rank-2 truth table."""
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


def _input_refs(width: int) -> np.ndarray:
    return np.arange(-INPUT_OFFSET, -INPUT_OFFSET - width, -1, dtype=np.int64)


def _input_index(ref: int) -> int:
    return -int(ref) - INPUT_OFFSET


@dataclass(frozen=True)
class LayerNetlist:
    """One discrete LogicDense layer."""

    lut_ids: np.ndarray
    connections: np.ndarray

    def __post_init__(self) -> None:
        ids_wide = np.asarray(self.lut_ids, dtype=np.int64).reshape(-1)
        connections = np.asarray(self.connections, dtype=np.int64)
        if connections.shape != (2, len(ids_wide)):
            raise ValueError(
                f"Expected connections shape (2, {len(ids_wide)}), got "
                f"{connections.shape}."
            )
        if np.any((ids_wide < 0) | (ids_wide > 15)):
            raise ValueError("Rank-2 LUT IDs must lie in [0, 15].")
        object.__setattr__(self, "lut_ids", ids_wide.astype(np.uint8))
        object.__setattr__(self, "connections", connections)

    @property
    def width(self) -> int:
        return int(len(self.lut_ids))


@dataclass(frozen=True)
class DenseCircuit:
    """Uncompacted sequential discrete LGN netlist."""

    input_width: int
    layers: tuple[LayerNetlist, ...]
    num_classes: int
    tau: float = 1.0
    beta: float = 0.0

    def __post_init__(self) -> None:
        if self.input_width < 1 or not self.layers:
            raise ValueError("A dense circuit needs inputs and at least one layer.")
        previous_width = self.input_width
        for index, layer in enumerate(self.layers):
            if layer.connections.size and (
                int(layer.connections.min()) < 0
                or int(layer.connections.max()) >= previous_width
            ):
                raise ValueError(
                    f"Layer {index} contains a connection outside [0, "
                    f"{previous_width - 1}]."
                )
            previous_width = layer.width
        if previous_width % self.num_classes:
            raise ValueError(
                f"Final width {previous_width} is not divisible by "
                f"{self.num_classes} classes."
            )

    @property
    def layer_widths(self) -> tuple[int, ...]:
        return tuple(layer.width for layer in self.layers)

    @property
    def raw_gate_count(self) -> int:
        return sum(self.layer_widths)

    @property
    def final_width(self) -> int:
        return self.layers[-1].width

    def evaluate_bits(self, inputs: np.ndarray) -> np.ndarray:
        values = _validate_boolean_inputs(inputs, self.input_width)
        for layer in self.layers:
            left = values[:, layer.connections[0]]
            right = values[:, layer.connections[1]]
            shift = 3 - (2 * left + right)
            values = ((layer.lut_ids[None, :] >> shift) & 1).astype(
                np.uint8, copy=False
            )
        return values

    def evaluate_counts(self, inputs: np.ndarray) -> np.ndarray:
        bits = self.evaluate_bits(inputs)
        return bits.reshape(len(bits), self.num_classes, -1).sum(
            axis=2, dtype=np.int64
        )

    def evaluate(self, inputs: np.ndarray) -> np.ndarray:
        return (self.evaluate_counts(inputs) + self.beta) / self.tau


@dataclass(frozen=True)
class CompactCircuit:
    """Reachable physical gates after exact Boolean simplification.

    References are topological node indices when non-negative. Negative values
    encode constants and primary inputs using the module-level constants.
    """

    input_width: int
    num_classes: int
    tau: float
    beta: float
    node_kinds: np.ndarray
    node_lut_ids: np.ndarray
    child_a: np.ndarray
    child_b: np.ndarray
    node_layers: np.ndarray
    outputs: np.ndarray
    raw_layer_widths: tuple[int, ...]

    def __post_init__(self) -> None:
        size = len(self.node_kinds)
        for name in ("node_lut_ids", "child_a", "child_b", "node_layers"):
            if len(getattr(self, name)) != size:
                raise ValueError(f"{name} has the wrong number of nodes.")
        if len(self.outputs) % self.num_classes:
            raise ValueError("Compact outputs cannot be divided into class groups.")

    @property
    def physical_gate_count(self) -> int:
        return int(len(self.node_kinds))

    @property
    def final_width(self) -> int:
        return int(len(self.outputs))

    def evaluate_bits(self, inputs: np.ndarray) -> np.ndarray:
        primary = _validate_boolean_inputs(inputs, self.input_width)
        values = np.empty(
            (len(primary), self.physical_gate_count), dtype=np.uint8
        )

        def resolve(ref: int) -> np.ndarray:
            if ref == CONST_ZERO:
                return np.zeros(len(primary), dtype=np.uint8)
            if ref == CONST_ONE:
                return np.ones(len(primary), dtype=np.uint8)
            if ref < 0:
                return primary[:, _input_index(ref)]
            return values[:, ref]

        for node in range(self.physical_gate_count):
            left = resolve(int(self.child_a[node]))
            if int(self.node_kinds[node]) == NOT_KIND:
                values[:, node] = 1 - left
                continue
            right = resolve(int(self.child_b[node]))
            shift = 3 - (2 * left + right)
            values[:, node] = (int(self.node_lut_ids[node]) >> shift) & 1

        output = np.empty((len(primary), self.final_width), dtype=np.uint8)
        refs = np.asarray(self.outputs)
        zero = refs == CONST_ZERO
        one = refs == CONST_ONE
        inputs_mask = refs <= -INPUT_OFFSET
        nodes_mask = refs >= 0
        output[:, zero] = 0
        output[:, one] = 1
        if np.any(inputs_mask):
            output[:, inputs_mask] = primary[
                :, np.asarray([_input_index(ref) for ref in refs[inputs_mask]])
            ]
        if np.any(nodes_mask):
            output[:, nodes_mask] = values[:, refs[nodes_mask]]
        return output

    def evaluate_counts(self, inputs: np.ndarray) -> np.ndarray:
        bits = self.evaluate_bits(inputs)
        return bits.reshape(len(bits), self.num_classes, -1).sum(
            axis=2, dtype=np.int64
        )

    def evaluate(self, inputs: np.ndarray) -> np.ndarray:
        return (self.evaluate_counts(inputs) + self.beta) / self.tau


def _validate_boolean_inputs(inputs: np.ndarray, width: int) -> np.ndarray:
    values = np.asarray(inputs)
    if values.ndim != 2 or values.shape[1] != width:
        raise ValueError(
            f"Expected Boolean inputs with shape (batch, {width}), got "
            f"{values.shape}."
        )
    if not np.all((values == 0) | (values == 1)):
        raise ValueError("Discrete circuit inputs must contain only 0 and 1.")
    return values.astype(np.uint8, copy=False)


def build_compact_circuit(circuit: DenseCircuit) -> CompactCircuit:
    """Build a functionally identical physical-gate netlist."""
    previous_refs = _input_refs(circuit.input_width)
    kinds: list[int] = []
    lut_ids: list[int] = []
    children: list[tuple[int, int]] = []
    node_layers: list[int] = []

    def add_node(
        kind: int, lut_id: int, left: int, right: int, layer_index: int
    ) -> int:
        kinds.append(kind)
        lut_ids.append(lut_id)
        children.append((int(left), int(right)))
        node_layers.append(layer_index)
        return len(kinds) - 1

    def invert(ref: int, layer_index: int) -> int:
        if ref == CONST_ZERO:
            return CONST_ONE
        if ref == CONST_ONE:
            return CONST_ZERO
        if ref >= 0 and kinds[ref] == NOT_KIND:
            return children[ref][0]
        return add_node(NOT_KIND, 12, ref, CONST_ZERO, layer_index)

    def unary(
        ref: int, output_at_zero: int, output_at_one: int, layer_index: int
    ) -> int:
        if output_at_zero == output_at_one:
            return CONST_ONE if output_at_zero else CONST_ZERO
        if output_at_zero == 0 and output_at_one == 1:
            return ref
        return invert(ref, layer_index)

    def apply_lut(lut_id: int, left: int, right: int, layer_index: int) -> int:
        # An inverter immediately feeding a LUT is folded into its truth table.
        if left >= 0 and kinds[left] == NOT_KIND:
            left = children[left][0]
            lut_id = _permute_input(lut_id, 0)
        if right >= 0 and kinds[right] == NOT_KIND:
            right = children[right][0]
            lut_id = _permute_input(lut_id, 1)

        if left in (CONST_ZERO, CONST_ONE) and right in (
            CONST_ZERO,
            CONST_ONE,
        ):
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

        return add_node(BINARY_KIND, lut_id, left, right, layer_index)

    for layer_index, layer in enumerate(circuit.layers):
        current_refs = np.empty(layer.width, dtype=np.int64)
        for neuron, lut_id in enumerate(layer.lut_ids):
            left = int(previous_refs[layer.connections[0, neuron]])
            right = int(previous_refs[layer.connections[1, neuron]])
            current_refs[neuron] = apply_lut(
                int(lut_id), left, right, layer_index
            )
        previous_refs = current_refs

    # GroupSum consumes every final bit, so each final reference is a root.
    stack = [int(ref) for ref in previous_refs if ref >= 0]
    reachable: set[int] = set()
    while stack:
        node = stack.pop()
        if node in reachable:
            continue
        reachable.add(node)
        stack.extend(child for child in children[node] if child >= 0)

    # Original node IDs are topological; filtering in ascending order preserves
    # that property and keeps evaluation simple.
    kept = sorted(reachable)
    remap = {old: new for new, old in enumerate(kept)}

    def remap_ref(ref: int) -> int:
        return remap[ref] if ref >= 0 else ref

    kept_children = [children[node] for node in kept]
    return CompactCircuit(
        input_width=circuit.input_width,
        num_classes=circuit.num_classes,
        tau=circuit.tau,
        beta=circuit.beta,
        node_kinds=np.asarray([kinds[node] for node in kept], dtype=np.uint8),
        node_lut_ids=np.asarray(
            [lut_ids[node] for node in kept], dtype=np.uint8
        ),
        child_a=np.asarray(
            [remap_ref(pair[0]) for pair in kept_children], dtype=np.int64
        ),
        child_b=np.asarray(
            [remap_ref(pair[1]) for pair in kept_children], dtype=np.int64
        ),
        node_layers=np.asarray(
            [node_layers[node] for node in kept], dtype=np.int32
        ),
        outputs=np.asarray(
            [remap_ref(int(ref)) for ref in previous_refs], dtype=np.int64
        ),
        raw_layer_widths=circuit.layer_widths,
    )


def _to_numpy_int(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.int64)


def _connection_indices(layer: Any) -> np.ndarray:
    connections = layer.connections
    indices = getattr(connections, "indices", None)
    if indices is None:
        raise TypeError("LogicDense connections do not expose `indices`.")
    indices_array = _to_numpy_int(indices)
    if indices_array.ndim == 2:
        return indices_array

    # Match da4ml-torch's discrete choice for learnable connections.
    weights = getattr(connections, "weights", None)
    if indices_array.ndim == 3 and weights is not None:
        choice = _to_numpy_int(weights.detach().cpu().argmax(dim=0))
        lut_axis = np.arange(int(layer.lut_rank))[:, None]
        out_axis = np.arange(int(layer.out_dim))[None, :]
        return indices_array[choice, lut_axis, out_axis]
    raise ValueError(
        "Only fixed connections or discretized learnable dense connections "
        f"are supported; got indices shape {indices_array.shape}."
    )


def extract_dense_circuit(model: Any) -> tuple[DenseCircuit, dict[str, Any]]:
    """Extract the hard rank-2 circuit from a loaded TorchLogix model."""
    original = getattr(model, "_orig_mod", model)
    layers_attr = getattr(original, "layers", None)
    if layers_attr is None:
        candidates = list(original.children())
    else:
        candidates = list(layers_attr)
    logic_layers = [
        layer for layer in candidates if type(layer).__name__ == "LogicDense"
    ]
    if not logic_layers:
        # Portable synthesis checkpoints are often torch.nn.Sequential.
        logic_layers = [
            layer
            for layer in original.modules()
            if layer is not original and type(layer).__name__ == "LogicDense"
        ]
    if not logic_layers:
        raise TypeError("No dense TorchLogix LogicDense layers were found.")

    netlist_layers = []
    previous_width = int(logic_layers[0].in_dim)
    for index, layer in enumerate(logic_layers):
        if int(getattr(layer, "lut_rank", -1)) != 2:
            raise ValueError("Only rank-2 LogicDense layers are supported.")
        if int(layer.in_dim) != previous_width:
            raise ValueError(
                f"Layer {index} expects {int(layer.in_dim)} inputs after a "
                f"{previous_width}-wide layer; the model is not sequential dense."
            )
        _, ids = layer.get_luts_and_ids()
        netlist_layers.append(
            LayerNetlist(_to_numpy_int(ids), _connection_indices(layer))
        )
        previous_width = int(layer.out_dim)

    group_sum = getattr(original, "group_sum", None)
    if group_sum is None:
        group_sum = next(
            (
                module
                for module in original.modules()
                if module is not original and type(module).__name__ == "GroupSum"
            ),
            None,
        )
    if group_sum is None:
        raise TypeError("No TorchLogix GroupSum output layer was found.")

    circuit = DenseCircuit(
        input_width=int(logic_layers[0].in_dim),
        layers=tuple(netlist_layers),
        num_classes=int(group_sum.k),
        tau=float(group_sum.tau),
        beta=float(group_sum.beta),
    )
    binarizer = getattr(original, "bin", None)
    if binarizer is None:
        binarizer = next(
            (
                module
                for module in original.modules()
                if "Binarization" in type(module).__name__
            ),
            None,
        )
    threshold_shape = None
    if binarizer is not None and hasattr(binarizer, "get_thresholds"):
        thresholds = binarizer.get_thresholds()
        if thresholds is not None:
            threshold_shape = list(thresholds.shape)

    return circuit, {
        "model_class": f"{type(original).__module__}.{type(original).__name__}",
        "threshold_shape": threshold_shape,
        "logic_layer_count": len(logic_layers),
    }


def require_s08(circuit: DenseCircuit) -> None:
    expected = (17_400, 17_400, 17_400, 11_600, 11_600, 11_600, 8_700)
    if circuit.input_width != 1_400 or circuit.layer_widths != expected:
        raise ValueError(
            "Checkpoint is not s08: expected 1,400 binary inputs and layer "
            f"widths {expected}, got {circuit.input_width} and "
            f"{circuit.layer_widths}."
        )
    if circuit.num_classes != 3 or circuit.tau != 40.0:
        raise ValueError(
            "Checkpoint is not the requested s08 tau-40 classifier: expected "
            f"3 classes and tau=40, got {circuit.num_classes} and "
            f"tau={circuit.tau}."
        )


def make_test_cases(
    input_width: int,
    *,
    random_cases: int = 32,
    seed: int = 808,
    feature_bits: tuple[int, int] | None = None,
) -> tuple[np.ndarray, dict[str, int]]:
    """Create edge, structured, and ordinary random Boolean inputs."""
    if random_cases < 1:
        raise ValueError("At least one random equivalence case is required.")
    edge = [
        np.zeros(input_width, dtype=np.uint8),
        np.ones(input_width, dtype=np.uint8),
    ]
    for index in sorted({0, input_width - 1}):
        one_hot = np.zeros(input_width, dtype=np.uint8)
        one_hot[index] = 1
        edge.append(one_hot)
        one_cold = np.ones(input_width, dtype=np.uint8)
        one_cold[index] = 0
        edge.append(one_cold)

    structured = [
        (np.arange(input_width) % 2).astype(np.uint8),
        ((np.arange(input_width) + 1) % 2).astype(np.uint8),
    ]
    if feature_bits is not None:
        features, bits = feature_bits
        if features * bits != input_width:
            raise ValueError(
                f"feature_bits={feature_bits} does not cover {input_width} inputs."
            )
        for level in sorted({0, 1, bits // 2, max(0, bits - 1), bits}):
            thermometer = np.zeros((features, bits), dtype=np.uint8)
            thermometer[:, :level] = 1
            structured.append(thermometer.reshape(-1))
        staircase = np.zeros((features, bits), dtype=np.uint8)
        for feature in range(features):
            level = round(bits * feature / max(1, features - 1))
            staircase[feature, :level] = 1
        structured.append(staircase.reshape(-1))

    rng = np.random.default_rng(seed)
    random = rng.integers(
        0, 2, size=(random_cases, input_width), dtype=np.uint8
    )
    all_cases = np.vstack((*edge, *structured, random))
    return all_cases, {
        "edge": len(edge),
        "structured": len(structured),
        "random": random_cases,
    }


def _move_plain_tensors_to_cpu(model: Any) -> None:
    """Move TorchLogix indices that older checkpoints did not register."""
    if hasattr(model, "cpu"):
        model.cpu()
    for child in model.modules():
        connections = getattr(child, "connections", None)
        indices = getattr(connections, "indices", None)
        if hasattr(indices, "detach"):
            connections.indices = indices.detach().cpu()
        if hasattr(child, "device"):
            child.device = "cpu"
        if connections is not None and hasattr(connections, "device"):
            connections.device = "cpu"


def checkpoint_discrete_outputs(model: Any, inputs: np.ndarray) -> np.ndarray:
    """Evaluate the checkpoint's own discrete layers after binarization."""
    import torch

    original = getattr(model, "_orig_mod", model)
    _move_plain_tensors_to_cpu(original)
    original.eval()
    layers_attr = getattr(original, "layers", None)
    candidates = list(layers_attr) if layers_attr is not None else list(
        original.children()
    )
    layers = [layer for layer in candidates if type(layer).__name__ == "LogicDense"]
    if not layers:
        layers = [
            layer
            for layer in original.modules()
            if layer is not original and type(layer).__name__ == "LogicDense"
        ]
    group_sum = getattr(original, "group_sum", None)
    if group_sum is None:
        group_sum = next(
            module
            for module in original.modules()
            if module is not original and type(module).__name__ == "GroupSum"
        )

    # Export mode makes the layer use its selected hard truth table directly.
    for layer in layers:
        if hasattr(layer, "set_export_mode"):
            layer.set_export_mode(True)

    with torch.no_grad():
        values = torch.from_numpy(np.asarray(inputs, dtype=np.bool_))
        for layer in layers:
            values = layer(values)
        return group_sum(values).detach().cpu().numpy()


def run_equivalence_suite(
    circuit: DenseCircuit,
    compact: CompactCircuit,
    *,
    model: Any | None = None,
    random_cases: int = 32,
    seed: int = 808,
    feature_bits: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Require exact regular-vs-compact equality on diverse test inputs."""
    inputs, categories = make_test_cases(
        circuit.input_width,
        random_cases=random_cases,
        seed=seed,
        feature_bits=feature_bits,
    )
    regular_counts = circuit.evaluate_counts(inputs)
    compact_counts = compact.evaluate_counts(inputs)
    mismatch_rows = np.flatnonzero(np.any(regular_counts != compact_counts, axis=1))
    if len(mismatch_rows):
        first = int(mismatch_rows[0])
        raise AssertionError(
            "Compacted circuit differs from the regular circuit at test case "
            f"{first}: regular={regular_counts[first].tolist()}, "
            f"compact={compact_counts[first].tolist()}."
        )

    checkpoint_match = None
    checkpoint_max_abs_error = None
    if model is not None:
        checkpoint_output = checkpoint_discrete_outputs(model, inputs)
        regular_output = circuit.evaluate(inputs)
        difference = np.abs(checkpoint_output - regular_output)
        checkpoint_max_abs_error = float(difference.max(initial=0.0))
        checkpoint_match = bool(np.allclose(checkpoint_output, regular_output, atol=1e-6))
        if not checkpoint_match:
            location = np.unravel_index(int(difference.argmax()), difference.shape)
            raise AssertionError(
                "Extracted regular netlist differs from the checkpoint's own "
                f"discrete forward pass at {location}; absolute error is "
                f"{difference[location]:.6g}."
            )

    return {
        "passed": True,
        "comparison": "exact GroupSum integer counts",
        "total_cases": int(len(inputs)),
        "case_counts": categories,
        "random_seed": int(seed),
        "regular_vs_compact_mismatched_cases": 0,
        "checkpoint_forward_checked": model is not None,
        "checkpoint_forward_match": checkpoint_match,
        "checkpoint_forward_max_abs_error": checkpoint_max_abs_error,
    }


def regular_summary(circuit: DenseCircuit) -> dict[str, Any]:
    layers = []
    previous_width = circuit.input_width
    for index, layer in enumerate(circuit.layers):
        fanout = np.bincount(layer.connections.reshape(-1), minlength=previous_width)
        gate_counts = np.bincount(layer.lut_ids, minlength=16)
        layers.append(
            {
                "index": index,
                "input_width": previous_width,
                "output_width": layer.width,
                "gate_counts": {
                    GATE_NAMES[i]: int(gate_counts[i]) for i in range(16)
                },
                "connection_count": int(layer.connections.size),
                "used_inputs": int(np.count_nonzero(fanout)),
                "unused_inputs": int(np.count_nonzero(fanout == 0)),
                "mean_fanout": float(fanout.mean()),
                "max_fanout": int(fanout.max(initial=0)),
                "same_input_gate_count": int(
                    np.count_nonzero(layer.connections[0] == layer.connections[1])
                ),
            }
        )
        previous_width = layer.width
    return {
        "input_width": circuit.input_width,
        "layer_widths": list(circuit.layer_widths),
        "raw_gate_count": circuit.raw_gate_count,
        "num_classes": circuit.num_classes,
        "final_width": circuit.final_width,
        "outputs_per_class": circuit.final_width // circuit.num_classes,
        "group_sum_tau": circuit.tau,
        "group_sum_beta": circuit.beta,
        "layers": layers,
    }


def compact_summary(compact: CompactCircuit) -> dict[str, Any]:
    layers = []
    for index, raw_width in enumerate(compact.raw_layer_widths):
        mask = compact.node_layers == index
        ids = compact.node_lut_ids[mask]
        gate_counts = np.bincount(ids, minlength=16)
        reachable = int(np.count_nonzero(mask))
        layers.append(
            {
                "index": index,
                "raw_gate_count": int(raw_width),
                "reachable_physical_gates": reachable,
                "removed_or_absorbed_gates": int(raw_width - reachable),
                "gate_counts": {
                    GATE_NAMES[i]: int(gate_counts[i]) for i in range(16)
                },
            }
        )
    raw_total = int(sum(compact.raw_layer_widths))
    physical = compact.physical_gate_count
    output_refs = np.asarray(compact.outputs)
    return {
        "physical_gate_count": physical,
        "raw_gate_count": raw_total,
        "removed_or_absorbed_gates": raw_total - physical,
        "reduction_fraction": (raw_total - physical) / raw_total,
        "reachable_binary_gates": int(
            np.count_nonzero(compact.node_kinds == BINARY_KIND)
        ),
        "reachable_inverters": int(
            np.count_nonzero(compact.node_kinds == NOT_KIND)
        ),
        "constant_final_outputs": int(
            np.count_nonzero(
                (output_refs == CONST_ZERO) | (output_refs == CONST_ONE)
            )
        ),
        "passthrough_primary_final_outputs": int(
            np.count_nonzero(output_refs <= -INPUT_OFFSET)
        ),
        "layers": layers,
    }


def save_regular_netlist(path: Path, circuit: DenseCircuit) -> None:
    arrays: dict[str, Any] = {
        "input_width": np.asarray(circuit.input_width),
        "num_classes": np.asarray(circuit.num_classes),
        "tau": np.asarray(circuit.tau),
        "beta": np.asarray(circuit.beta),
        "layer_count": np.asarray(len(circuit.layers)),
    }
    for index, layer in enumerate(circuit.layers):
        arrays[f"layer_{index}_lut_ids"] = layer.lut_ids
        arrays[f"layer_{index}_connections"] = layer.connections
    np.savez_compressed(path, **arrays)


def save_compact_netlist(path: Path, compact: CompactCircuit) -> None:
    np.savez_compressed(
        path,
        input_width=np.asarray(compact.input_width),
        num_classes=np.asarray(compact.num_classes),
        tau=np.asarray(compact.tau),
        beta=np.asarray(compact.beta),
        node_kinds=compact.node_kinds,
        node_lut_ids=compact.node_lut_ids,
        child_a=compact.child_a,
        child_b=compact.child_b,
        node_layers=compact.node_layers,
        outputs=compact.outputs,
        raw_layer_widths=np.asarray(compact.raw_layer_widths),
    )
