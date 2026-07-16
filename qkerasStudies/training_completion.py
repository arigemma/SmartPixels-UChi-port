"""Validate that a QKeras checkpoint satisfied campaign training requirements."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def validate_training_completion(
    checkpoint: Path,
    status_path: Path,
    minimum_hours: float,
    required_minimum_steps: int = 0,
) -> dict:
    if not checkpoint.exists():
        raise ValueError(f"Missing final checkpoint: {checkpoint}")
    if not status_path.is_file():
        raise ValueError(f"Missing training completion metadata: {status_path}")
    status = json.loads(status_path.read_text())
    if status.get("complete") is not True:
        raise ValueError(f"Training is not marked complete: {status_path}")

    requested_hours = float(status.get("minimum_training_hours", -1))
    elapsed_hours = float(status.get("elapsed_training_hours", -1))
    minimum_steps = int(status.get("minimum_steps", -1))
    completed_steps = int(status.get("completed_steps", -1))
    final_recorded_step = int(status.get("final_recorded_step", -1))
    if requested_hours < minimum_hours or elapsed_hours < minimum_hours:
        raise ValueError(
            f"Checkpoint has only {elapsed_hours:.3f} training hours with "
            f"{requested_hours:.3f} requested; require at least "
            f"{minimum_hours:.3f}."
        )
    if minimum_steps < required_minimum_steps:
        raise ValueError(
            f"Checkpoint requested only {minimum_steps} minimum steps; "
            f"require at least {required_minimum_steps}."
        )
    if completed_steps < minimum_steps or final_recorded_step != completed_steps:
        raise ValueError(
            "Checkpoint step accounting is incomplete: "
            f"minimum={minimum_steps}, completed={completed_steps}, "
            f"recorded={final_recorded_step}."
        )
    if status.get("stop_requested"):
        raise ValueError(f"Training completion contains a stop request: {status_path}")
    return status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--minimum-hours", type=float, default=3.0)
    parser.add_argument("--minimum-steps", type=int, default=6_000)
    args = parser.parse_args()
    try:
        status = validate_training_completion(
            args.checkpoint,
            args.status,
            args.minimum_hours,
            args.minimum_steps,
        )
    except (ValueError, json.JSONDecodeError) as error:
        raise SystemExit(str(error)) from error
    print(
        f"[TRAINING VALID] steps={status['completed_steps']} "
        f"duration={float(status['elapsed_training_hours']):.3f}h "
        f"balanced_accuracy={float(status['final_balanced_accuracy']):.6f}"
    )


if __name__ == "__main__":
    main()
