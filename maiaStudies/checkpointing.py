"""Portable checkpoint helpers for MAIA TorchLogix models.

Some TorchLogix releases keep fixed random connection indices outside the
ordinary ``state_dict``. Those indices are part of the trained model: loading
learned LUT weights onto newly randomized connections changes the network.
Store and restore them explicitly for correct cross-process evaluation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch


CHECKPOINT_FORMAT = "maia_lnn_state_dict_v3_explicit_connections"


def snapshot_to_cpu(value):
    """Recursively detach and clone tensors onto CPU."""

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {key: snapshot_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [snapshot_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(snapshot_to_cpu(item) for item in value)
    return value


def capture_fixed_connections(model: torch.nn.Module) -> dict:
    """Capture every TorchLogix layer's fixed connection indices."""

    result = {}
    for module_name, module in model.named_modules():
        connections = getattr(module, "connections", None)
        if connections is None or isinstance(connections, str):
            continue
        indices = getattr(connections, "indices", None)
        if indices is not None:
            result[module_name] = snapshot_to_cpu(indices)
    if not result:
        raise RuntimeError("No fixed TorchLogix connection indices were found")
    return result


def _copy_tensor_structure(target, source, path: str) -> None:
    if isinstance(target, torch.Tensor):
        if not isinstance(source, torch.Tensor):
            raise TypeError(f"Connection state type mismatch at {path}")
        if target.shape != source.shape:
            raise ValueError(
                f"Connection state shape mismatch at {path}: "
                f"{tuple(target.shape)} != {tuple(source.shape)}"
            )
        target.copy_(source.to(device=target.device, dtype=target.dtype))
        return
    if isinstance(target, Sequence) and not isinstance(target, (str, bytes)):
        if not isinstance(source, Sequence) or isinstance(source, (str, bytes)):
            raise TypeError(f"Connection state type mismatch at {path}")
        if len(target) != len(source):
            raise ValueError(
                f"Connection state length mismatch at {path}: "
                f"{len(target)} != {len(source)}"
            )
        for index, (target_item, source_item) in enumerate(zip(target, source)):
            _copy_tensor_structure(target_item, source_item, f"{path}[{index}]")
        return
    raise TypeError(f"Unsupported connection state at {path}: {type(target)!r}")


@torch.no_grad()
def restore_fixed_connections(model: torch.nn.Module, saved: Mapping) -> None:
    """Restore explicitly saved fixed connections into a rebuilt model."""

    modules = dict(model.named_modules())
    missing = sorted(set(saved) - set(modules))
    if missing:
        raise KeyError(f"Checkpoint connection modules are missing: {missing}")
    for module_name, source_indices in saved.items():
        connections = getattr(modules[module_name], "connections", None)
        target_indices = getattr(connections, "indices", None)
        if target_indices is None:
            raise RuntimeError(
                f"Model module {module_name!r} has no restorable connection indices"
            )
        _copy_tensor_structure(
            target_indices, source_indices, f"{module_name}.connections.indices"
        )


def load_model_checkpoint_state(model: torch.nn.Module, checkpoint: Mapping) -> None:
    """Load learned state and the fixed wiring that gives it meaning."""

    if checkpoint.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(
            f"Unexpected checkpoint format {checkpoint.get('format')!r}; "
            f"expected {CHECKPOINT_FORMAT!r}"
        )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    connection_state = checkpoint.get("fixed_connection_state")
    if not connection_state:
        raise ValueError("Checkpoint is missing explicit fixed connection state")
    restore_fixed_connections(model, connection_state)
