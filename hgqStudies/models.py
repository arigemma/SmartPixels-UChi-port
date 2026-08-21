"""HGQ2 Model-2 classifier construction and quantizer introspection."""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from hgqStudies.campaign import HGQModelConfig


def _imports():
    os.environ.setdefault("KERAS_BACKEND", "torch")
    import keras
    from hgq.config import QuantizerConfig, QuantizerConfigScope
    from hgq.layers import QBatchNormDense, QDense

    return keras, QuantizerConfig, QuantizerConfigScope, QBatchNormDense, QDense


def build_model(config: HGQModelConfig):
    keras, QuantizerConfig, QuantizerConfigScope, QBatchNormDense, QDense = _imports()

    # Begin at the most permissive QKeras comparator (approximately W5/A10)
    # and let HGQ's beta regularizer learn lower bit widths where useful.
    weight_scope = QuantizerConfigScope(
        place="all",
        k0=1,
        b0=4,
        i0=0,
        default_q_type="kbi",
        overflow_mode="sat_sym",
    )
    data_scope = QuantizerConfigScope(
        place="datalane",
        k0=0,
        i0=5,
        f0=5,
        default_q_type="kif",
        overflow_mode="wrap",
    )

    with weight_scope, data_scope:
        signed_input = QuantizerConfig(place="datalane", k0=1)
        layers = [keras.layers.Input(shape=(14,), name="features")]
        layers.append(
            QBatchNormDense(
                config.hidden_units[0],
                beta0=config.beta0,
                iq_conf=signed_input,
                activation="relu",
                name="dense_0_bn",
            )
        )
        for layer_index, width in enumerate(config.hidden_units[1:], start=1):
            layers.append(
                QDense(
                    width,
                    beta0=config.beta0,
                    activation="relu",
                    name=f"dense_{layer_index}",
                )
            )
        layers.append(
            QDense(
                3,
                beta0=config.beta0,
                enable_oq=True,
                name="logits",
            )
        )

    return keras.Sequential(layers, name=f"hgq2_model2_{config.alias}")


def load_model(path):
    keras, *_unused = _imports()
    # Importing HGQ above registers all custom serializable layers.
    return keras.models.load_model(path, compile=False)


def model_ebops(model) -> float:
    total = 0.0
    for layer in model._flatten_layers(include_self=False, recursive=True):
        ebops = getattr(layer, "ebops", None)
        if ebops is None:
            continue
        try:
            value = np.asarray(ebops.detach().cpu().numpy())
        except AttributeError:
            try:
                value = np.asarray(ebops.numpy())
            except AttributeError:
                value = np.asarray(ebops)
        total += float(value.sum())
    return total


def quantizer_statistics(model) -> dict[str, Any]:
    keras, *_unused = _imports()
    records = []
    seen = set()
    for layer in model._flatten_layers(include_self=False, recursive=True):
        for quantizer_name in ("iq", "kq", "bq", "oq"):
            quantizer = getattr(layer, quantizer_name, None)
            if quantizer is None or id(quantizer) in seen:
                continue
            seen.add(id(quantizer))
            bits = getattr(quantizer, "bits", None)
            if bits is None:
                continue
            values = np.asarray(keras.ops.convert_to_numpy(bits), dtype=np.float64)
            if values.size == 0:
                continue
            records.append(
                {
                    "layer": layer.name,
                    "quantizer": quantizer_name,
                    "count": int(values.size),
                    "minimum_bits": float(values.min()),
                    "maximum_bits": float(values.max()),
                    "mean_bits": float(values.mean()),
                }
            )
    return {"quantizers": records, "quantizer_count": len(records)}
