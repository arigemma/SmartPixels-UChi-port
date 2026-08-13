"""Activation-bit fault injection for the selected QKeras dense models."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from qkerasStudies.models import QKerasModelConfig


FAULT_SCOPES = ("all", "input")


@dataclass(frozen=True)
class QuantizationRange:
    minimum: float
    maximum: float

    def to_dict(self) -> dict[str, float]:
        return {"minimum": self.minimum, "maximum": self.maximum}


def validate_fault_scope(fault_scope: str) -> str:
    if fault_scope not in FAULT_SCOPES:
        raise ValueError(
            f"Fault scope must be one of {FAULT_SCOPES}, got {fault_scope!r}."
        )
    return fault_scope


def validate_fault_probability(probability: float) -> float:
    if isinstance(probability, bool):
        raise ValueError(f"Fault probability must be numeric, got {probability!r}.")
    probability = float(probability)
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError(
            f"Fault probability must lie in [0, 1], got {probability!r}."
        )
    return probability


def validate_flip_count(flip_count: int, total_sites: int) -> int:
    if isinstance(flip_count, bool) or int(flip_count) != flip_count:
        raise ValueError(f"Flip count must be an integer, got {flip_count!r}.")
    flip_count = int(flip_count)
    if flip_count < 0 or flip_count > total_sites:
        raise ValueError(
            f"Flip count must lie in [0, {total_sites}], got {flip_count}."
        )
    return flip_count


def eligible_stage_indices(number_stages: int, fault_scope: str) -> tuple[int, ...]:
    fault_scope = validate_fault_scope(fault_scope)
    if fault_scope == "input":
        return (0,)
    return tuple(range(number_stages))


def stage_widths_for_scope(
    config: QKerasModelConfig,
    fault_scope: str,
) -> tuple[int, ...]:
    indices = eligible_stage_indices(len(config.model_stage_widths), fault_scope)
    return tuple(config.model_stage_widths[index] for index in indices)


def total_fault_sites(config: QKerasModelConfig, fault_scope: str) -> int:
    return sum(stage_widths_for_scope(config, fault_scope)) * config.activation_bits


def sample_unique_fault_indices(
    batch_size: int,
    total_sites: int,
    flip_count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Uniformly sample exactly K unique bit sites for every event.

    This mirrors the LGN sampler strategy: draw with replacement and redraw
    duplicates. The campaign only uses K <= 5.3% of the fault space, so
    duplicate rejection remains cheap even for the largest QKeras model.
    """
    flip_count = validate_flip_count(flip_count, total_sites)
    if batch_size < 0:
        raise ValueError(f"Batch size must be non-negative, got {batch_size}.")
    if flip_count == 0 or batch_size == 0:
        return np.empty((batch_size, flip_count), dtype=np.int64)
    if flip_count == total_sites:
        return np.broadcast_to(
            np.arange(total_sites, dtype=np.int64),
            (batch_size, total_sites),
        ).copy()

    # At high fault fractions, duplicate rejection becomes both slow and
    # unreliable.  Random-key partitioning samples a uniform K-subset for
    # every event and remains well behaved through the 60% input-fault point.
    if flip_count * 10 >= total_sites:
        random_keys = rng.random((batch_size, total_sites))
        return np.argpartition(random_keys, kth=flip_count - 1, axis=1)[
            :, :flip_count
        ].astype(np.int64, copy=False)

    indices = rng.integers(
        0,
        total_sites,
        size=(batch_size, flip_count),
        dtype=np.int64,
    )
    for _ in range(100):
        sorted_indices = np.sort(indices, axis=1)
        duplicates_sorted = np.zeros_like(sorted_indices, dtype=bool)
        duplicates_sorted[:, 1:] = sorted_indices[:, 1:] == sorted_indices[:, :-1]
        if not duplicates_sorted.any():
            return indices

        duplicate_values = sorted_indices[duplicates_sorted]
        duplicate_original = np.zeros_like(indices, dtype=bool)
        for value in np.unique(duplicate_values):
            matches = indices == value
            duplicate_original |= matches & (np.cumsum(matches, axis=1) > 1)
        indices[duplicate_original] = rng.integers(
            0,
            total_sites,
            size=int(duplicate_original.sum()),
            dtype=np.int64,
        )

    raise RuntimeError("Could not sample unique fault sites after 100 redraws.")


def _stage_ranges(
    config: QKerasModelConfig,
    input_quantization_range: QuantizationRange,
) -> tuple[QuantizationRange, ...]:
    return (
        input_quantization_range,
        *[
            QuantizationRange(0.0, 1.0)
            for _ in config.hidden_units
        ],
    )


def _call_layer(layer, values, training: bool):
    try:
        return layer(values, training=training)
    except TypeError:
        return layer(values)


def _quantize_to_uint(values, bits: int, value_range: QuantizationRange):
    import tensorflow as tf

    levels = float((1 << bits) - 1)
    span = value_range.maximum - value_range.minimum
    if span <= 0:
        raise ValueError(f"Invalid quantization range: {value_range}")
    normalized = (values - value_range.minimum) / span
    clipped = tf.clip_by_value(normalized, 0.0, 1.0)
    return tf.cast(tf.round(clipped * levels), tf.int32)


def _dequantize_from_uint(integers, bits: int, value_range: QuantizationRange):
    import tensorflow as tf

    levels = tf.cast((1 << bits) - 1, tf.float32)
    span = tf.cast(value_range.maximum - value_range.minimum, tf.float32)
    minimum = tf.cast(value_range.minimum, tf.float32)
    return tf.cast(integers, tf.float32) / levels * span + minimum


def _apply_bernoulli_bit_faults(
    values,
    probability: float,
    bits: int,
    value_range: QuantizationRange,
):
    import tensorflow as tf

    probability = validate_fault_probability(probability)
    if probability == 0.0:
        return values

    integers = _quantize_to_uint(values, bits, value_range)
    bit_values = tf.constant([1 << bit for bit in range(bits)], dtype=tf.int32)
    selected = tf.random.uniform(
        tf.concat([tf.shape(integers), [bits]], axis=0),
        dtype=tf.float32,
    ) < probability
    masks = tf.reduce_sum(tf.cast(selected, tf.int32) * bit_values, axis=-1)
    faulted = tf.bitwise.bitwise_xor(integers, masks)
    dequantized = _dequantize_from_uint(faulted, bits, value_range)
    return values + tf.stop_gradient(dequantized - values)


def _apply_exact_stage_bit_faults(
    values,
    fault_indices,
    stage_offset: int,
    bits: int,
    value_range: QuantizationRange,
):
    import tensorflow as tf

    if fault_indices is None:
        return values
    integers = _quantize_to_uint(values, bits, value_range)
    stage_bit_width = tf.cast(tf.shape(integers)[1] * bits, tf.int64)
    local = fault_indices - stage_offset
    in_stage = (local >= 0) & (local < stage_bit_width)
    if not bool(tf.reduce_any(in_stage).numpy()):
        return values

    batch_indices = tf.tile(
        tf.reshape(tf.range(tf.shape(integers)[0], dtype=tf.int64), (-1, 1)),
        (1, tf.shape(fault_indices)[1]),
    )
    selected_batch = tf.boolean_mask(batch_indices, in_stage)
    selected_local = tf.boolean_mask(local, in_stage)
    selected_value = selected_local // bits
    selected_bit = selected_local % bits
    scatter_indices = tf.stack(
        [selected_batch, selected_value],
        axis=1,
    )
    bit_masks = tf.bitwise.left_shift(
        tf.ones_like(selected_bit, dtype=tf.int32),
        tf.cast(selected_bit, tf.int32),
    )
    masks = tf.scatter_nd(
        scatter_indices,
        bit_masks,
        tf.cast(tf.shape(integers), tf.int64),
    )
    faulted = tf.bitwise.bitwise_xor(integers, masks)
    return _dequantize_from_uint(faulted, bits, value_range)


def forward_with_faults(
    model,
    inputs,
    config: QKerasModelConfig,
    *,
    training: bool,
    fault_scope: str,
    input_quantization_range: QuantizationRange,
    training_fault_probability: float = 0.0,
    exact_fault_indices=None,
):
    """Run a selected dense QKeras model with optional transient bit faults."""
    import tensorflow as tf

    validate_fault_scope(fault_scope)
    training_fault_probability = validate_fault_probability(
        training_fault_probability
    )
    if exact_fault_indices is not None:
        exact_fault_indices = tf.convert_to_tensor(exact_fault_indices, dtype=tf.int64)

    eligible = eligible_stage_indices(len(config.model_stage_widths), fault_scope)
    ranges = _stage_ranges(config, input_quantization_range)
    stage_offsets: dict[int, int] = {}
    offset = 0
    for model_stage_index in eligible:
        stage_offsets[model_stage_index] = offset
        offset += config.model_stage_widths[model_stage_index] * config.activation_bits

    def apply_faults(values, model_stage_index: int):
        if model_stage_index not in stage_offsets:
            return values
        if exact_fault_indices is not None:
            return _apply_exact_stage_bit_faults(
                values,
                exact_fault_indices,
                stage_offsets[model_stage_index],
                config.activation_bits,
                ranges[model_stage_index],
            )
        if training:
            return _apply_bernoulli_bit_faults(
                values,
                training_fault_probability,
                config.activation_bits,
                ranges[model_stage_index],
            )
        return values

    values = tf.convert_to_tensor(inputs, dtype=tf.float32)
    values = apply_faults(values, 0)
    activation_stage_index = 0
    for layer in model.layers:
        if layer.__class__.__name__ == "InputLayer":
            continue
        values = _call_layer(layer, values, training=training)
        if layer.name.startswith("relu"):
            activation_stage_index += 1
            values = apply_faults(values, activation_stage_index)
    return values


def stage_metadata(
    config: QKerasModelConfig,
    fault_scope: str,
) -> dict:
    indices = eligible_stage_indices(len(config.model_stage_widths), fault_scope)
    widths = tuple(config.model_stage_widths[index] for index in indices)
    bit_widths = tuple(width * config.activation_bits for width in widths)
    offsets = [0]
    for width in bit_widths[:-1]:
        offsets.append(offsets[-1] + width)
    return {
        "activation_bits": config.activation_bits,
        "model_stage_widths": config.model_stage_widths,
        "eligible_model_stage_indices": indices,
        "stage_widths": widths,
        "stage_bit_widths": bit_widths,
        "stage_offsets": tuple(offsets),
        "total_fault_sites_per_event": sum(bit_widths),
    }
