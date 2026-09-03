"""Differentiable transient activation faults used during LGN training."""

from __future__ import annotations

import math

import torch

from lgnStudies.bitFlipStudy.fault_injection import (
    FAULT_OPERATIONS,
    FAULT_SCOPES,
    _eligible_stage_indices,
    validate_fault_operation,
    validate_fault_scope,
)


def validate_fault_probability(probability: float) -> float:
    """Validate an independent per-site flip probability."""
    if isinstance(probability, bool):
        raise ValueError(f"Fault probability must be numeric, got {probability!r}.")
    probability = float(probability)
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError(
            f"Fault probability must lie in [0, 1], got {probability!r}."
        )
    return probability


def apply_bernoulli_faults(
    values: torch.Tensor,
    probability: float,
    fault_operation: str = "flip",
) -> torch.Tensor:
    """Independently corrupt each activation with probability ``p``.

    ``torch.where`` keeps the operation out of place and differentiable.
    LogicDense activations are relaxed values in [0, 1] during training, so the
    differentiable analogue of a Boolean bit flip is ``x -> 1 - x``.
    """
    probability = validate_fault_probability(probability)
    fault_operation = validate_fault_operation(fault_operation)
    if probability == 0.0:
        return values
    if probability == 1.0 and fault_operation == "flip":
        return 1 - values
    selected = torch.rand(values.shape, device=values.device) < probability
    if fault_operation == "flip":
        return torch.where(selected, 1 - values, values)

    targets = torch.randint(
        0,
        2,
        values.shape,
        dtype=torch.int64,
        device=values.device,
    ).to(values.dtype)
    faulted = torch.where(selected, targets, values)
    # Exact random 0/1 values in the forward pass with an identity
    # straight-through gradient for the relaxed training activations.
    return values + (faulted - values).detach()


class FaultAwareTrainingModel(torch.nn.Module):
    """Train a dense model-2 LGN with transient activation faults.

    Faults are enabled only while this wrapper is in training mode. By
    default, every output of FixedBinarization and every LogicDense layer is
    independently selected with the configured probability on each forward
    pass. With ``fault_scope="input"``, only the FixedBinarization output is
    eligible. GroupSum and model parameters are not faulted.
    """

    def __init__(
        self,
        base_model: torch.nn.Module,
        fault_probability: float,
        fault_scope: str = "all",
        fault_operation: str = "flip",
    ):
        super().__init__()
        for attribute in ("bin", "layers", "group_sum"):
            if not hasattr(base_model, attribute):
                raise TypeError(
                    f"Expected a dense model-2 LGN with attribute `{attribute}`."
                )
        if len(base_model.layers) == 0:
            raise TypeError("Expected at least one LogicDense layer.")

        try:
            input_width = int(base_model.layers[0].in_dim)
            layer_widths = [int(layer.out_dim) for layer in base_model.layers]
        except AttributeError as error:
            raise TypeError(
                "Every model-2 logic layer must expose in_dim and out_dim."
            ) from error

        self.base_model = base_model
        self.fault_probability = validate_fault_probability(fault_probability)
        self.model_stage_widths = (input_width, *layer_widths)
        self.fault_scope = validate_fault_scope(fault_scope)
        self.fault_operation = validate_fault_operation(fault_operation)
        self._eligible_stage_indices = _eligible_stage_indices(
            len(self.model_stage_widths), self.fault_scope
        )
        self.stage_widths = tuple(
            self.model_stage_widths[index] for index in self._eligible_stage_indices
        )
        self.total_fault_sites = sum(self.stage_widths)

    @property
    def expected_flips_per_event(self) -> float:
        return self.expected_changed_sites_per_event

    @property
    def expected_selected_sites_per_event(self) -> float:
        return self.fault_probability * self.total_fault_sites

    @property
    def expected_changed_sites_per_event(self) -> float:
        multiplier = 0.5 if self.fault_operation == "random_set" else 1.0
        return multiplier * self.expected_selected_sites_per_event

    @property
    def fault_count_standard_deviation(self) -> float:
        p = self.fault_probability
        changed_probability = p / 2 if self.fault_operation == "random_set" else p
        return math.sqrt(
            self.total_fault_sites
            * changed_probability
            * (1 - changed_probability)
        )

    @property
    def selected_site_count_standard_deviation(self) -> float:
        p = self.fault_probability
        return math.sqrt(self.total_fault_sites * p * (1 - p))

    def _apply_training_faults(self, values: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return values
        return apply_bernoulli_faults(
            values,
            self.fault_probability,
            self.fault_operation,
        )

    def _should_fault_stage(self, model_stage_index: int) -> bool:
        return model_stage_index in self._eligible_stage_indices

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        values = self.base_model.bin(inputs)
        if self._should_fault_stage(0):
            values = self._apply_training_faults(values)
        for layer_index, layer in enumerate(self.base_model.layers, start=1):
            values = layer(values)
            if self._should_fault_stage(layer_index):
                values = self._apply_training_faults(values)
        return self.base_model.group_sum(values)
