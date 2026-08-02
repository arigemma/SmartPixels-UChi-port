"""Inspect, compact, validate, and visualize discrete dense LGNs."""

from .circuit import (
    CompactCircuit,
    DenseCircuit,
    LayerNetlist,
    build_compact_circuit,
    extract_dense_circuit,
    run_equivalence_suite,
)

__all__ = (
    "CompactCircuit",
    "DenseCircuit",
    "LayerNetlist",
    "build_compact_circuit",
    "extract_dense_circuit",
    "run_equivalence_suite",
)
