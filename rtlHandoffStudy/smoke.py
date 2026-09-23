#!/usr/bin/env python3
"""Fast construction, forward, freeze, and serialization smoke test."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import torch

from models.registry import REGISTRY
from models.rtl_handoff import RTL_HANDOFF_BY_ALIAS
from rtlHandoffStudy.portable import (
    compare_frozen_outputs,
    export_deployment_artifacts,
)


def main() -> None:
    alias = "dense_s03_t31_learn_g64"
    entry = RTL_HANDOFF_BY_ALIAS[alias]
    model_class = REGISTRY[entry["registry_name"]]["class"]
    thresholds = torch.linspace(-1.0, 1.0, 31).repeat(14, 1)
    model = model_class(thresholds=thresholds).cpu().eval()
    sample = torch.linspace(-0.9, 0.9, 28).reshape(2, 14)
    with torch.no_grad():
        expected = model(sample)
    difference = compare_frozen_outputs(model, sample)
    if difference != 0.0:
        raise RuntimeError(f"Frozen smoke-test output differs by {difference}.")

    with tempfile.TemporaryDirectory(prefix="rtl-handoff-smoke.") as temporary:
        output = Path(temporary)
        export_deployment_artifacts(
            model,
            output,
            validation_metrics={"balanced_accuracy": 0.0},
            training_step=0,
        )
        loaded = torch.load(
            output / "best_model.importable.pth",
            map_location="cpu",
            weights_only=False,
        ).eval()
        with torch.no_grad():
            actual = loaded(sample)
        if not torch.equal(expected, actual):
            raise RuntimeError("Serialized portable model changed the discrete output.")
        metadata = json.loads((output / "model.json").read_text())
        if metadata["input_contract"]["compact_input_bits"] != 70:
            raise RuntimeError("Unexpected dense T31 compact input width.")
        for name in (
            "best_model.importable.pth",
            "discrete_netlist.pth",
            "thresholds.pth",
            "model.json",
            "SHA256SUMS",
        ):
            if not (output / name).is_file():
                raise RuntimeError(f"Smoke export omitted {name}.")

    conv_alias = "conv_s01_t15_fixed_g100"
    conv_entry = RTL_HANDOFF_BY_ALIAS[conv_alias]
    conv_class = REGISTRY[conv_entry["registry_name"]]["class"]
    conv_thresholds = torch.linspace(-1.0, 1.0, 15).repeat(105, 1)
    conv = conv_class(thresholds=conv_thresholds).cpu().eval()
    conv_sample = torch.zeros(1, 105)
    with torch.no_grad():
        conv_expected = conv(conv_sample)
    with tempfile.TemporaryDirectory(prefix="rtl-handoff-conv-smoke.") as temporary:
        output = Path(temporary)
        export_deployment_artifacts(
            conv,
            output,
            validation_metrics={"balanced_accuracy": 0.0},
            training_step=0,
        )
        loaded = torch.load(
            output / "best_model.importable.pth",
            map_location="cpu",
            weights_only=False,
        ).eval()
        with torch.no_grad():
            conv_actual = loaded(conv_sample)
        if not torch.equal(conv_expected, conv_actual):
            raise RuntimeError("Serialized convolutional model changed the output.")
        metadata = json.loads((output / "model.json").read_text())
        if metadata["input_contract"]["compact_input_bits"] != 420:
            raise RuntimeError("Unexpected convolutional T15 compact input width.")
    print("[OK] RTL handoff model construction and portable export smoke test passed.")


if __name__ == "__main__":
    main()
