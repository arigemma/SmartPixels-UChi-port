"""Portable staged LGN checkpoints for the synthesis-SmartPixels harness."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import torch


def _move_plain_tensors_to_cpu(module: torch.nn.Module) -> None:
    """Move TorchLogix tensors that are attributes rather than buffers."""
    for child in module.modules():
        connections = getattr(child, "connections", None)
        indices = getattr(connections, "indices", None)
        if isinstance(indices, torch.Tensor):
            connections.indices = indices.detach().cpu()
        if hasattr(child, "device"):
            child.device = "cpu"
        if connections is not None and hasattr(connections, "device"):
            connections.device = "cpu"


def make_portable_dense_lgn(model: torch.nn.Module) -> torch.nn.Sequential:
    """Copy a classification.py dense LGN into an importable sequential module."""
    original = getattr(model, "_orig_mod", model)
    for attribute in ("bin", "layers", "group_sum"):
        if not hasattr(original, attribute):
            raise TypeError(
                f"Expected a dense model-2 LGN with attribute `{attribute}`."
            )

    children = [copy.deepcopy(original.bin)]
    children.extend(copy.deepcopy(layer) for layer in original.layers)
    children.append(copy.deepcopy(original.group_sum))
    portable = torch.nn.Sequential(*children).cpu().eval()
    _move_plain_tensors_to_cpu(portable)
    return portable


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def export_dense_lgn_stage(
    model: torch.nn.Module,
    export_root: Path,
    export_name: str,
    *,
    model_alias: str,
    registry_name: str,
    training_step: int,
    balanced_accuracy: float,
) -> Path:
    """Write one directory accepted directly by run_da4ml_safe.py."""
    target = export_root / "model2" / "LGN" / export_name
    target.mkdir(parents=True, exist_ok=True)
    checkpoint = target / "best_model.importable.pth"
    temporary = target / "best_model.importable.pth.tmp"

    portable = make_portable_dense_lgn(model)
    torch.save(portable, temporary)
    temporary.replace(checkpoint)

    original = getattr(model, "_orig_mod", model)
    hidden_dims = [
        int(getattr(layer, "out_dim")) for layer in original.layers
    ]
    tau = float(original.group_sum.tau)
    n_bits = int(getattr(original, "n_bits", 100))
    metadata = {
        "model_name": export_name,
        "model_set": "model2",
        "flow": "lgn",
        "input_shape": [1, 14],
        "model_kwargs": {
            "hidden_dims": hidden_dims,
            "tau": tau,
            "n_bits": n_bits,
        },
        "training_metadata": {
            "model_alias": model_alias,
            "registry_name": registry_name,
            "training_step": int(training_step),
            "balanced_accuracy": float(balanced_accuracy),
        },
    }
    (target / "model.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (target / "SHA256SUMS").write_text(
        f"{_sha256(checkpoint)}  {checkpoint.name}\n"
    )
    return target
