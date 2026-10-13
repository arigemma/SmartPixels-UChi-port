"""Fast environment and forward-shape check run before Slurm submission."""

from __future__ import annotations

import torch

from .campaign import (
    CAMPAIGN_CONFIGS,
    N_CONV_INPUT_FEATURES,
    N_DENSE_INPUT_FEATURES,
    N_POSITION_FEATURES,
)
from .models import build_model


def main() -> None:
    cases = (
        (
            next(row for row in CAMPAIGN_CONFIGS if row.family == "dense"),
            torch.zeros(2, N_DENSE_INPUT_FEATURES),
            {
                "dense": torch.linspace(-1, 1, 100).repeat(
                    N_DENSE_INPUT_FEATURES, 1
                )
            },
        ),
        (
            next(row for row in CAMPAIGN_CONFIGS if row.family == "conv"),
            torch.zeros(2, N_CONV_INPUT_FEATURES),
            {
                "cluster": torch.linspace(-1, 1, 100).reshape(1, 100),
                "position": torch.linspace(-1, 1, 100).repeat(
                    N_POSITION_FEATURES, 1
                ),
            },
        ),
    )
    for config, inputs, thresholds in cases:
        model = build_model(config, thresholds, "cpu")
        model.eval()
        with torch.no_grad():
            output = model(inputs)
        if tuple(output.shape) != (2, 2):
            raise RuntimeError(
                f"{config.alias}: expected binary output (2,2), got {tuple(output.shape)}"
            )
        print(f"[PREFLIGHT] {config.alias}: input={tuple(inputs.shape)} output={tuple(output.shape)}")


if __name__ == "__main__":
    main()
