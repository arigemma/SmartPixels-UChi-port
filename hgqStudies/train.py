"""Train and fully evaluate one QKeras-matched HGQ2 Model-2 point."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "torch")

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataloader import create_dataloaders
from dataset import UnderSampler
from hgqStudies.campaign import config_for_alias, config_for_task
from hgqStudies.common import (
    balanced_accuracy_from_confusion,
    confusion_from_predictions,
    sha256_file,
    to_numpy,
)
from hgqStudies.models import build_model, load_model, model_ebops, quantizer_statistics


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def infinite_batches(loader):
    while True:
        for inputs, labels in loader:
            yield to_numpy(inputs).astype(np.float32), to_numpy(labels).astype(np.float32)


def configure_backend(device: str):
    import keras

    if keras.backend.backend() != "torch":
        raise RuntimeError(
            f"HGQ2 campaign requires KERAS_BACKEND=torch, got {keras.backend.backend()!r}."
        )
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested, but PyTorch cannot see a CUDA GPU.")
        device_name = "cuda"
    else:
        device_name = "cpu"

    # Keras' public device scope controls where Torch-backend variables and
    # batches live without changing PyTorch's global tensor default. Keeping
    # the scope entered for this one-model process also keeps Dataset tensors
    # on CPU until Keras transfers complete batches.
    device_scope = keras.device(device_name)
    device_scope.__enter__()
    return device_scope


def balanced_metric_class():
    import keras

    @keras.saving.register_keras_serializable(package="SmartPixels")
    class BalancedCategoricalAccuracy(keras.metrics.Metric):
        def __init__(self, name="balanced_accuracy", **kwargs):
            super().__init__(name=name, **kwargs)
            self.confusion = self.add_weight(
                name="confusion", shape=(3, 3), initializer="zeros"
            )

        def update_state(self, y_true, y_pred, sample_weight=None):
            truth = keras.ops.argmax(y_true, axis=-1)
            prediction = keras.ops.argmax(y_pred, axis=-1)
            truth_one_hot = keras.ops.one_hot(truth, 3)
            prediction_one_hot = keras.ops.one_hot(prediction, 3)
            confusion = keras.ops.matmul(
                keras.ops.transpose(truth_one_hot), prediction_one_hot
            )
            self.confusion.assign_add(keras.ops.cast(confusion, self.dtype))

        def result(self):
            totals = keras.ops.sum(self.confusion, axis=1)
            recalls = keras.ops.diagonal(self.confusion) / keras.ops.maximum(totals, 1.0)
            return keras.ops.mean(recalls)

        def reset_state(self):
            self.confusion.assign(keras.ops.zeros_like(self.confusion))

    return BalancedCategoricalAccuracy


def best_checkpoint_callback(output: Path):
    import keras

    class BestValidationCheckpoint(keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.best_cross_entropy = math.inf
            self.best_epoch = None
            self.best_validation_balanced_accuracy = None

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            cross_entropy = logs.get("val_cross_entropy")
            if cross_entropy is None or not math.isfinite(float(cross_entropy)):
                return
            if float(cross_entropy) >= self.best_cross_entropy:
                return
            self.best_cross_entropy = float(cross_entropy)
            self.best_epoch = int(epoch + 1)
            balanced_accuracy = logs.get("val_balanced_accuracy")
            self.best_validation_balanced_accuracy = (
                None if balanced_accuracy is None else float(balanced_accuracy)
            )
            self.model.save(output / "best_model.keras", overwrite=True)
            write_json(
                output / "best_checkpoint.json",
                {
                    "epoch": self.best_epoch,
                    "validation_cross_entropy": self.best_cross_entropy,
                    "validation_balanced_accuracy": self.best_validation_balanced_accuracy,
                    "validation_loss_with_hgq_regularization": float(logs["val_loss"]),
                    "training_ebops": float(logs.get("ebops", model_ebops(self.model))),
                },
            )
            print(
                f"[CHECKPOINT] epoch={self.best_epoch} "
                f"val_cross_entropy={self.best_cross_entropy:.6f} "
                f"val_balanced_accuracy={self.best_validation_balanced_accuracy:.6f}",
                flush=True,
            )

    return BestValidationCheckpoint()


def evaluate(model, loader) -> dict:
    confusion = np.zeros((3, 3), dtype=np.int64)
    cross_entropy_sum = 0.0
    sample_count = 0
    with torch.no_grad():
        for inputs, labels in loader:
            logits = model(to_numpy(inputs).astype(np.float32), training=False)
            logits = to_numpy(logits).astype(np.float64, copy=False)
            labels = to_numpy(labels)
            predictions = np.argmax(logits, axis=-1)
            truth = np.argmax(labels, axis=-1)
            confusion += confusion_from_predictions(truth, predictions)
            shifted = logits - logits.max(axis=-1, keepdims=True)
            log_normalizer = np.log(np.exp(shifted).sum(axis=-1))
            cross_entropy_sum += float(
                (-shifted[np.arange(len(truth)), truth] + log_normalizer).sum()
            )
            sample_count += len(truth)
    metrics = balanced_accuracy_from_confusion(confusion)
    metrics["categorical_accuracy"] = float(np.trace(confusion) / confusion.sum())
    metrics["cross_entropy"] = float(cross_entropy_sum / sample_count)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--task", type=int)
    selection.add_argument("--alias")
    parser.add_argument("--data-config", default="config/10783560/v3.yml")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--optimizer-steps", type=int, default=6000)
    parser.add_argument("--steps-per-epoch", type=int, default=50)
    parser.add_argument("--validation-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.optimizer_steps <= 0 or args.steps_per_epoch <= 0:
        raise ValueError("Optimizer and epoch step counts must be positive.")
    if args.optimizer_steps % args.steps_per_epoch:
        raise ValueError("--optimizer-steps must be divisible by --steps-per-epoch.")
    if args.validation_every <= 0:
        raise ValueError("--validation-every must be positive.")

    config = config_for_task(args.task) if args.task is not None else config_for_alias(args.alias)
    output = args.output_root.resolve() / config.alias
    if output.exists():
        raise FileExistsError(f"Refusing to reuse existing model directory: {output}")
    output.mkdir(parents=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    _device_scope = configure_backend(args.device)

    write_json(output / "model_config.json", config.to_dict())
    train_loader, val_loader, test_loader = create_dataloaders(
        config_path=args.data_config,
        batch_size=args.batch_size,
        shuffle=True,
        input_type=config.input_type,
        target_type="classification",
        balance_classes=False,
        val_size=0.01,
        label_format="one-hot",
    )
    train_sampler = UnderSampler(
        train_loader.dataset,
        target_class_ratios={0: 1 / 3, 1: 1 / 3, 2: 1 / 3},
        max_oversampling_rate=1.0,
    )
    balanced_train_loader = DataLoader(
        train_loader.dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=0,
    )
    if len(balanced_train_loader) == 0 or len(val_loader) == 0 or len(test_loader) == 0:
        raise RuntimeError("Train, validation, and test loaders must all be non-empty.")

    import keras
    import hgq
    from hgq.utils.sugar import FreeEBOPs

    model = build_model(config)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss=keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=[
            keras.metrics.CategoricalAccuracy(),
            balanced_metric_class()(),
            keras.metrics.CategoricalCrossentropy(
                from_logits=True, name="cross_entropy"
            ),
        ],
    )
    model.summary()

    epochs = args.optimizer_steps // args.steps_per_epoch
    checkpoint = best_checkpoint_callback(output)
    model.fit(
        infinite_batches(balanced_train_loader),
        steps_per_epoch=args.steps_per_epoch,
        epochs=epochs,
        validation_data=infinite_batches(val_loader),
        validation_steps=len(val_loader),
        validation_freq=args.validation_every,
        callbacks=[
            FreeEBOPs(),
            checkpoint,
            keras.callbacks.CSVLogger(output / "training_history.csv"),
            keras.callbacks.TerminateOnNaN(),
        ],
        shuffle=False,
        verbose=2,
    )
    model.save(output / "final_model.keras", overwrite=True)
    if not (output / "best_model.keras").is_file():
        raise RuntimeError("Training completed without a best validation checkpoint.")

    best_model = load_model(output / "best_model.keras")
    # Populate HGQ's operation accounting before reading EBOPs.
    first_inputs, _first_labels = next(iter(test_loader))
    best_model(to_numpy(first_inputs[:1]).astype(np.float32), training=False)
    metrics = evaluate(best_model, test_loader)
    metrics.update(
        {
            "alias": config.alias,
            "task_index": config.index,
            "input_type": config.input_type,
            "hidden_units": list(config.hidden_units),
            "beta0": config.beta0,
            "qkeras_peer": config.qkeras_peer,
            "reference_lut": config.reference_lut,
            "reference_ff": config.reference_ff,
            "reference_balanced_accuracy": config.reference_balanced_accuracy,
            "reference_resources_lut_plus_ff": config.reference_resources,
            "hgq_ebops": model_ebops(best_model),
            "keras_parameter_count": int(best_model.count_params()),
            "best_epoch": checkpoint.best_epoch,
            "best_validation_cross_entropy": checkpoint.best_cross_entropy,
            "best_validation_balanced_accuracy": (
                checkpoint.best_validation_balanced_accuracy
            ),
            "optimizer_steps": args.optimizer_steps,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "keras_version": keras.__version__,
            "hgq2_version": hgq.__version__,
            "keras_backend": keras.backend.backend(),
            "quantization": quantizer_statistics(best_model),
        }
    )
    write_json(output / "test_metrics.json", metrics)
    completion = {
        "schema_version": 1,
        "complete": True,
        "alias": config.alias,
        "task_index": config.index,
        "balanced_accuracy": metrics["balanced_accuracy"],
        "best_model": "best_model.keras",
        "best_model_sha256": sha256_file(output / "best_model.keras"),
        "test_metrics": "test_metrics.json",
    }
    write_json(output / "training_complete.json", completion)
    print(
        f"[DONE] {config.alias} balanced_accuracy={metrics['balanced_accuracy']:.6f} "
        f"EBOPs={metrics['hgq_ebops']:.1f} checkpoint={output / 'best_model.keras'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
