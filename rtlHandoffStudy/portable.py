"""Portable deployment artifacts for dense and convolutional LGNs."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch


class PortableDenseLGN(torch.nn.Module):
    """Registry-independent dense LGN used by the synthesis handoff."""

    def __init__(self, binarizer, layers, group_sum, *, n_bits: int) -> None:
        super().__init__()
        self.bin = binarizer
        self.layers = torch.nn.ModuleList(layers)
        self.group_sum = group_sum
        self.n_bits = int(n_bits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value = self.bin(x)
        for layer in self.layers:
            value = layer(value)
        return self.group_sum(value)


class PortableConvLGN(torch.nn.Module):
    """Registry-independent four-convolution LGN used by synthesis."""

    def __init__(
        self,
        *,
        binarizer,
        conv1,
        conv2,
        or_pool1,
        conv3,
        conv4,
        fc1,
        fc2,
        fc3,
        reduction_fcs,
        fc4,
        group_sum,
        n_bits: int,
    ) -> None:
        super().__init__()
        self.bin = binarizer
        self.conv1 = conv1
        self.conv2 = conv2
        self.or_pool1 = or_pool1
        self.conv3 = conv3
        self.conv4 = conv4
        self.fc1 = fc1
        self.fc2 = fc2
        self.fc3 = fc3
        self.reduction_fcs = torch.nn.ModuleList(reduction_fcs)
        self.fc4 = fc4
        self.group_sum = group_sum
        self.n_bits = int(n_bits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value = self.bin(x)
        y0 = value[:, : self.n_bits]
        profile = value[:, self.n_bits :].reshape(-1, self.n_bits, 13, 8)
        value = self.conv1(profile)
        value = self.conv2(value)
        value = self.or_pool1(value)
        value = self.conv3(value)
        value = self.conv4(value)
        value = value.reshape(value.shape[0], -1)
        value = torch.cat((y0, value), dim=1)
        value = self.fc3(self.fc2(self.fc1(value)))
        for layer in self.reduction_fcs:
            value = layer(value)
        value = self.fc4(value)
        return self.group_sum(value)


def _move_plain_tensors_to_cpu(module: torch.nn.Module) -> None:
    """Move TorchLogix connection tensors, which are not registered buffers."""
    for child in module.modules():
        connections = getattr(child, "connections", None)
        indices = getattr(connections, "indices", None)
        if isinstance(indices, torch.Tensor):
            connections.indices = indices.detach().cpu()
        elif isinstance(indices, (list, tuple)):
            moved = _cpu_tree(indices)
            connections.indices = type(indices)(moved)
        if hasattr(child, "device"):
            child.device = "cpu"
        if connections is not None and hasattr(connections, "device"):
            connections.device = "cpu"


def _cpu_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    return value


def freeze_binarizer(model: torch.nn.Module) -> torch.nn.Module:
    """Deep-copy a model and replace its input binarizer with fixed thresholds."""
    original = getattr(model, "_orig_mod", model)
    if not hasattr(original, "bin") or not hasattr(original.bin, "get_thresholds"):
        raise TypeError("Expected an LGN with a top-level binarizer exposing get_thresholds().")
    from torchlogix.layers import FixedBinarization

    frozen = copy.deepcopy(original)
    thresholds = original.bin.get_thresholds().detach().clone()
    feature_dim = int(getattr(original.bin, "feature_dim", -2))
    frozen.bin = FixedBinarization(thresholds=thresholds, feature_dim=feature_dim)
    frozen.eval()
    return frozen


def make_portable_model(frozen_model: torch.nn.Module) -> torch.nn.Module:
    """Copy a fixed-threshold campaign model into a registry-independent class."""
    original = getattr(frozen_model, "_orig_mod", frozen_model)
    if type(original.bin).__name__ != "FixedBinarization":
        raise TypeError(
            "Deployment export requires a frozen FixedBinarization; found "
            f"{type(original.bin).__name__}."
        )
    if getattr(original, "family", None) == "dense":
        portable = PortableDenseLGN(
            copy.deepcopy(original.bin),
            [copy.deepcopy(layer) for layer in original.layers],
            copy.deepcopy(original.group_sum),
            n_bits=int(original.n_bits),
        )
    elif getattr(original, "family", None) == "conv":
        portable = PortableConvLGN(
            binarizer=copy.deepcopy(original.bin),
            conv1=copy.deepcopy(original.conv1),
            conv2=copy.deepcopy(original.conv2),
            or_pool1=copy.deepcopy(original.or_pool1),
            conv3=copy.deepcopy(original.conv3),
            conv4=copy.deepcopy(original.conv4),
            fc1=copy.deepcopy(original.fc1),
            fc2=copy.deepcopy(original.fc2),
            fc3=copy.deepcopy(original.fc3),
            reduction_fcs=[copy.deepcopy(layer) for layer in original.reduction_fcs],
            fc4=copy.deepcopy(original.fc4),
            group_sum=copy.deepcopy(original.group_sum),
            n_bits=int(original.n_bits),
        )
    else:
        raise TypeError(f"Unsupported RTL handoff family {getattr(original, 'family', None)!r}.")
    portable.cpu().eval()
    _move_plain_tensors_to_cpu(portable)
    return portable


def _extract_discrete_layer(module: torch.nn.Module) -> dict[str, Any] | None:
    connections = getattr(getattr(module, "connections", None), "indices", None)
    if not hasattr(module, "get_luts_and_ids") or connections is None:
        return None
    _luts, identifiers = module.get_luts_and_ids()
    attributes = {}
    for name in (
        "in_dim",
        "out_dim",
        "lut_rank",
        "channels",
        "num_kernels",
        "receptive_field_size",
        "tree_depth",
        "stride",
        "padding",
    ):
        if hasattr(module, name):
            attributes[name] = _cpu_tree(getattr(module, name))
    return {
        "type": type(module).__name__,
        "attributes": attributes,
        "lut_ids": _cpu_tree(identifiers),
        "connections": _cpu_tree(connections),
    }


def discrete_netlist(portable: torch.nn.Module, metadata: dict[str, Any]) -> dict[str, Any]:
    """Return a Python-class-independent integer description of trained logic."""
    layers = {}
    for name, module in portable.named_modules():
        if not name:
            continue
        extracted = _extract_discrete_layer(module)
        if extracted is not None:
            layers[name] = extracted
    thresholds = portable.bin.get_thresholds().detach().cpu()
    return {
        "format": "smartpixels_lgn_discrete_netlist_v1",
        "metadata": metadata,
        "thresholds": thresholds,
        "layers": layers,
        "group_sum": {
            "classes": int(portable.group_sum.k),
            "training_tau": float(portable.group_sum.tau),
            "training_beta": float(portable.group_sum.beta),
            "synthesis_tau": 1.0,
            "synthesis_beta": 0.0,
        },
    }


def _atomic_torch_save(value: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def export_deployment_artifacts(
    model: torch.nn.Module,
    output: Path,
    *,
    validation_metrics: dict[str, Any],
    training_step: int,
) -> dict[str, Any]:
    """Freeze thresholds and write portable/full plus discrete artifacts."""
    original = getattr(model, "_orig_mod", model)
    config = original.deployment_config()
    frozen = freeze_binarizer(original)
    portable = make_portable_model(frozen)

    metadata = {
        "format": "smartpixels_lgn_rtl_handoff_v1",
        "model_name": config["alias"],
        "model_set": "model2" if config["family"] == "dense" else "model3",
        "flow": "lgn",
        "input_shape": [1, int(config["raw_input_features"])],
        "model_kwargs": config,
        "input_contract": {
            "training_input": "continuous_scaled_features",
            "current_da4ml_input": "feature_major_threshold_minor_thermometer_bits",
            "proposed_eflx_input": "feature_major_unsigned_ordinal_codes",
            "ordinal_decode": "therm[f,t] = (code[f] > t)",
            "code_bits_per_feature": int(config["compact_code_bits_per_feature"]),
            "compact_input_bits": int(config["compact_input_bits"]),
            "thermometer_input_bits": int(config["thermometer_input_bits"]),
            "invalid_code_policy": "saturate_to_threshold_count",
        },
        "output_contract": {
            "classes": 3,
            "deployment_output": "three_integer_population_counts",
            "decision": "argmax_with_lowest-index_tie_break",
            "training_tau": float(config["tau"]),
            "synthesis_tau": 1.0,
            "synthesis_beta": 0.0,
        },
        "eflx4k_screen": {
            "accessible_user_inputs": 632,
            "lut_sites": 2520,
            "compact_input_margin": 632 - int(config["compact_input_bits"]),
            "fit_claim": "unknown_until_target_specific_synthesis",
        },
        "training": {
            "best_step": int(training_step),
            "validation_metrics": validation_metrics,
        },
    }

    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "best_model.importable.pth"
    netlist_path = output / "discrete_netlist.pth"
    threshold_path = output / "thresholds.pth"
    _atomic_torch_save(portable, checkpoint)
    _atomic_torch_save(discrete_netlist(portable, metadata), netlist_path)
    _atomic_torch_save(portable.bin.get_thresholds().detach().cpu(), threshold_path)
    (output / "model.json").write_text(json.dumps(metadata, indent=2) + "\n")
    checksummed = (checkpoint, netlist_path, threshold_path, output / "model.json")
    (output / "SHA256SUMS").write_text(
        "".join(f"{_sha256(path)}  {path.name}\n" for path in checksummed)
    )
    return metadata


def compare_frozen_outputs(
    model: torch.nn.Module,
    sample: torch.Tensor,
    *,
    atol: float = 0.0,
) -> float:
    """Require the trained eval model and frozen deployment model to agree."""
    original = getattr(model, "_orig_mod", model)
    original_mode = original.training
    original.eval()
    frozen = freeze_binarizer(original).to(sample.device).eval()
    with torch.no_grad():
        expected = original(sample)
        actual = frozen(sample)
    original.train(original_mode)
    maximum = float((expected - actual).abs().max().item())
    if maximum > atol:
        raise RuntimeError(
            f"Frozen-threshold deployment mismatch: max_abs_diff={maximum} > {atol}."
        )
    return maximum


def estimated_group_sum_luts(group_width: int) -> int:
    """Analytical Xilinx-style popcount diagnostic, not an EFLX fit claim."""
    from lgnStudies.resource_accounting import estimate_group_sum_luts

    return int(estimate_group_sum_luts(3 * int(group_width), 3))
