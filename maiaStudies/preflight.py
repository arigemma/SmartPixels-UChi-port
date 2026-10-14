"""Fast environment and forward-shape check run before Slurm submission."""

from __future__ import annotations

import torch

from .campaign import (
    CAMPAIGN_CONFIGS,
    N_CONV_INPUT_FEATURES,
    N_DENSE_INPUT_FEATURES,
    N_POSITION_FEATURES,
)
from .checkpointing import (
    CHECKPOINT_FORMAT,
    capture_fixed_connections,
    load_model_checkpoint_state,
    snapshot_to_cpu,
)
from .models import build_model


def main() -> None:
    cases = (
        (
            next(row for row in CAMPAIGN_CONFIGS if row.family == "dense"),
            torch.linspace(-1, 1, 2 * N_DENSE_INPUT_FEATURES).reshape(
                2, N_DENSE_INPUT_FEATURES
            ),
            {
                "dense": torch.linspace(-1, 1, 100).repeat(
                    N_DENSE_INPUT_FEATURES, 1
                )
            },
        ),
        (
            next(row for row in CAMPAIGN_CONFIGS if row.family == "conv"),
            torch.linspace(-1, 1, 2 * N_CONV_INPUT_FEATURES).reshape(
                2, N_CONV_INPUT_FEATURES
            ),
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
        checkpoint = {
            "format": CHECKPOINT_FORMAT,
            "model_state_dict": snapshot_to_cpu(model.state_dict()),
            "fixed_connection_state": capture_fixed_connections(model),
        }
        # Force a different initialization before rebuilding; explicit
        # connection restoration must still reproduce the original output.
        torch.manual_seed(8675309)
        rebuilt = build_model(config, thresholds, "cpu")
        load_model_checkpoint_state(rebuilt, checkpoint)
        rebuilt.eval()
        with torch.no_grad():
            reloaded_output = rebuilt(inputs)
        torch.testing.assert_close(output, reloaded_output, rtol=0.0, atol=0.0)
        print(
            f"[PREFLIGHT] {config.alias}: input={tuple(inputs.shape)} "
            f"output={tuple(output.shape)} checkpoint_roundtrip=passed"
        )


if __name__ == "__main__":
    main()
