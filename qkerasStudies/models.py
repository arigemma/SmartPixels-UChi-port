"""Local QKeras model registry used by the robustness campaigns.

The definitions are copied from the `wip-mila-qkeras` branch and kept local to
this study so the LGN analysis branch does not need a global registry rewrite.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")


N_CLASSES = 3


@dataclass(frozen=True)
class QKerasModelConfig:
    alias: str
    registry_name: str
    hidden_units: tuple[int, ...]
    weight_bits: int
    activation_bits: int
    input_dim: int = 14
    input_type: str = "y-profile"
    task_type: str = "classification"

    @property
    def model_stage_widths(self) -> tuple[int, ...]:
        """Input stage followed by hidden activation stages."""
        return (self.input_dim, *self.hidden_units)


MODEL_CONFIGS: tuple[QKerasModelConfig, ...] = (
    QKerasModelConfig(
        alias="qkeras_y192_w5a10",
        registry_name="qkeras-model-2-yprofile-d192-w5a10",
        hidden_units=(192,),
        weight_bits=5,
        activation_bits=10,
    ),
    QKerasModelConfig(
        alias="qkeras_y256_64_w4a8",
        registry_name="qkeras-model-2-yprofile-d256-d64-w4a8",
        hidden_units=(256, 64),
        weight_bits=4,
        activation_bits=8,
    ),
    QKerasModelConfig(
        alias="qkeras_y768_256_128_w5a10",
        registry_name="qkeras-model-2-yprofile-d768-d256-d128-w5a10",
        hidden_units=(768, 256, 128),
        weight_bits=5,
        activation_bits=10,
    ),
)
CONFIG_BY_ALIAS = {config.alias: config for config in MODEL_CONFIGS}
CONFIG_BY_REGISTRY = {config.registry_name: config for config in MODEL_CONFIGS}


def _import_qkeras_layers():
    try:
        import qkeras
        from qkeras import QActivation, QBatchNormalization, QDense, quantized_bits
    except ImportError as error:
        raise ImportError(
            "Google QKeras 0.9.0 is required. The imported 'qkeras' package "
            "does not provide its API and is probably HGQ2's incompatible "
            "Keras-3 compatibility namespace. Run "
            "internalBitOverwriteStudy/prepare_qkeras_runtime.sh and make "
            "sure its PYTHONPATH overlay is active."
        ) from error

    qkeras_root = Path(qkeras.__file__).resolve().parent
    if getattr(qkeras, "__version__", None) != "0.9.0" or not (
        qkeras_root / "qlayers.py"
    ).is_file():
        raise ImportError(
            "Expected the isolated Google QKeras 0.9.0 package with a "
            f"qlayers.py module, but imported {qkeras.__file__}. Run "
            "internalBitOverwriteStudy/prepare_qkeras_runtime.sh."
        )

    return QActivation, QBatchNormalization, QDense, quantized_bits


def _qbits(bits: int):
    _qactivation, _qbatch_norm, _qdense, quantized_bits = _import_qkeras_layers()
    return quantized_bits(bits, 0, alpha=1)


def _qrelu(bits: int) -> str:
    return f"quantized_relu({bits},0)"


def _qdense(x, units: int, weight_bits: int, name: str):
    _qactivation, _qbatch_norm, QDense, _quantized_bits = _import_qkeras_layers()
    return QDense(
        units,
        kernel_quantizer=_qbits(weight_bits),
        bias_quantizer=_qbits(weight_bits),
        name=name,
    )(x)


def _qdense_batchnorm(x, units: int, weight_bits: int, name: str):
    _qactivation, QBatchNormalization, QDense, _quantized_bits = _import_qkeras_layers()
    x = QDense(
        units,
        kernel_quantizer=_qbits(weight_bits),
        bias_quantizer=_qbits(weight_bits),
        name=name,
    )(x)
    return QBatchNormalization(name=f"{name}_bn")(x)


def build_model(registry_name: str):
    """Build one of the selected y-profile QKeras models."""
    import tensorflow as tf

    QActivation, _qbatch_norm, _qdense_cls, _quantized_bits = _import_qkeras_layers()
    config = get_config(registry_name)
    x = x_in = tf.keras.layers.Input(shape=(config.input_dim,), name="input1")
    for index, units in enumerate(config.hidden_units, start=1):
        if index == 1:
            x = _qdense_batchnorm(x, units, config.weight_bits, f"dense{index}")
        else:
            x = _qdense(x, units, config.weight_bits, f"dense{index}")
        x = QActivation(_qrelu(config.activation_bits), name=f"relu{index}")(x)
    x = _qdense(
        x,
        N_CLASSES,
        config.weight_bits,
        f"dense{len(config.hidden_units) + 1}",
    )
    x = tf.keras.layers.Activation("linear", name="linear")(x)
    model = tf.keras.Model(
        inputs=x_in,
        outputs=x,
        name=registry_name.replace("-", "_"),
    )
    setattr(model, "_smartpixels_model_name", registry_name)
    return model


def get_config(name_or_alias: str) -> QKerasModelConfig:
    if name_or_alias in CONFIG_BY_ALIAS:
        return CONFIG_BY_ALIAS[name_or_alias]
    if name_or_alias in CONFIG_BY_REGISTRY:
        return CONFIG_BY_REGISTRY[name_or_alias]
    raise ValueError(
        f"Unknown QKeras model {name_or_alias!r}; "
        f"available aliases are {sorted(CONFIG_BY_ALIAS)}."
    )


def qkeras_custom_objects() -> dict[str, Any]:
    custom_objects: dict[str, Any] = {}
    try:
        from qkeras.utils import _add_supported_quantized_objects
    except ImportError:
        pass
    else:
        _add_supported_quantized_objects(custom_objects)

    try:
        _qactivation, QBatchNormalization, _qdense, _quantized_bits_cls = (
            _import_qkeras_layers()
        )
    except ImportError:
        pass
    else:
        custom_objects["QBatchNormalization"] = QBatchNormalization
    return custom_objects


def model_metadata_path(model_path: str | Path) -> Path:
    path = Path(model_path)
    if path.is_dir():
        return path / "model_config.json"
    return path.parent / "model_config.json"


def write_model_metadata(model_path: str | Path, metadata: dict[str, Any]) -> None:
    path = model_metadata_path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


def load_model_metadata(model_path: str | Path) -> dict[str, Any] | None:
    path = model_metadata_path(model_path)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def load_model(model_path: str | Path, registry_name: str | None = None):
    """Load a saved QKeras model and attach registry provenance."""
    import tensorflow as tf

    model = tf.keras.models.load_model(
        model_path,
        custom_objects=qkeras_custom_objects(),
        compile=False,
    )
    metadata = load_model_metadata(model_path) or {}
    resolved_name = registry_name or metadata.get("registry_name") or metadata.get("model")
    if resolved_name is not None:
        get_config(resolved_name)
        setattr(model, "_smartpixels_model_name", resolved_name)
    return model


def count_trainable_params(model) -> int:
    import numpy as np

    return int(np.sum([np.prod(variable.shape) for variable in model.trainable_weights]))
