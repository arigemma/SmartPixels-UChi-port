"""Exact-count transient activation-bit faults for dense model-2 LGNs."""

from __future__ import annotations

import torch


FAULT_SCOPES = ("all", "input", "internal")
FAULT_OPERATIONS = ("flip", "random_set")


def validate_fault_scope(fault_scope: str) -> str:
    """Validate which activation stages are eligible for fault injection."""
    if fault_scope not in FAULT_SCOPES:
        raise ValueError(
            f"Fault scope must be one of {FAULT_SCOPES}, got {fault_scope!r}."
        )
    return fault_scope


def validate_fault_operation(fault_operation: str) -> str:
    """Validate how a selected Boolean site is corrupted."""
    if fault_operation not in FAULT_OPERATIONS:
        raise ValueError(
            f"Fault operation must be one of {FAULT_OPERATIONS}, "
            f"got {fault_operation!r}."
        )
    return fault_operation


def validate_flip_count(flip_count: int, total_sites: int) -> int:
    """Validate an exact number of unique faults per event."""
    if isinstance(flip_count, bool) or int(flip_count) != flip_count:
        raise ValueError(f"Flip count must be an integer, got {flip_count!r}.")
    flip_count = int(flip_count)
    if flip_count < 0 or flip_count > total_sites:
        raise ValueError(
            f"Flip count must lie in [0, {total_sites}], got {flip_count}."
        )
    return flip_count


def sample_unique_fault_indices(
    batch_size: int,
    total_sites: int,
    flip_count: int,
    device: torch.device | str,
) -> torch.Tensor:
    """Uniformly sample exactly ``flip_count`` distinct sites for every event.

    Each row is a simple random sample without replacement from
    ``range(total_sites)``. Rejection of duplicate draws avoids constructing a
    dense random tensor over every fault site.
    """
    flip_count = validate_flip_count(flip_count, total_sites)
    if batch_size < 0:
        raise ValueError(f"Batch size must be non-negative, got {batch_size}.")
    if flip_count == 0 or batch_size == 0:
        return torch.empty(
            (batch_size, flip_count), dtype=torch.int64, device=device
        )
    if flip_count == total_sites:
        return torch.arange(
            total_sites, dtype=torch.int64, device=device
        ).expand(batch_size, -1)

    # Duplicate rejection is efficient for the low-density campaigns.  For
    # high-density input-fault studies (for example K=60% of all input bits),
    # assign every site an independent random key and take the K smallest.
    # Continuous i.i.d. keys induce a uniform sample without replacement and
    # avoid a rejection loop whose collision rate explodes at large K/N.
    if flip_count * 10 >= total_sites:
        random_keys = torch.rand(
            (batch_size, total_sites), device=device, dtype=torch.float64
        )
        return torch.topk(
            random_keys,
            k=flip_count,
            dim=1,
            largest=False,
            sorted=False,
        ).indices

    indices = torch.randint(
        total_sites, (batch_size, flip_count), dtype=torch.int64, device=device
    )
    for _ in range(100):
        sorted_indices, order = indices.sort(dim=1)
        duplicate_sorted = torch.zeros_like(sorted_indices, dtype=torch.bool)
        duplicate_sorted[:, 1:] = (
            sorted_indices[:, 1:] == sorted_indices[:, :-1]
        )
        if not bool(duplicate_sorted.any()):
            return indices

        duplicate_original = torch.zeros_like(duplicate_sorted)
        duplicate_original.scatter_(1, order, duplicate_sorted)
        number_to_redraw = int(duplicate_original.sum())
        indices[duplicate_original] = torch.randint(
            total_sites,
            (number_to_redraw,),
            dtype=torch.int64,
            device=device,
        )

    raise RuntimeError("Could not sample unique fault sites after 100 redraws.")


def apply_stage_faults(
    values: torch.Tensor,
    fault_indices: torch.Tensor,
    stage_offset: int,
    *,
    fault_operation: str = "flip",
    fault_values: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply the planned global faults that fall inside one signal stage."""
    fault_operation = validate_fault_operation(fault_operation)
    if values.ndim != 2:
        raise ValueError(
            f"Expected a 2-D Boolean signal tensor, got shape {tuple(values.shape)}."
        )
    if fault_indices.ndim != 2 or fault_indices.shape[0] != values.shape[0]:
        raise ValueError("Fault plan batch dimension does not match activations.")
    if fault_operation == "random_set":
        if fault_values is None or fault_values.shape != fault_indices.shape:
            raise ValueError(
                "random_set requires one 0/1 target for every selected site."
            )
        if bool(((fault_values != 0) & (fault_values != 1)).any()):
            raise ValueError("Random-set targets must contain only 0 and 1.")
    stage_width = values.shape[1]
    in_stage = (fault_indices >= stage_offset) & (
        fault_indices < stage_offset + stage_width
    )
    batch_indices = (
        torch.arange(values.shape[0], device=values.device)
        .unsqueeze(1)
        .expand_as(fault_indices)
    )
    selected_batch = batch_indices[in_stage]
    selected_signal = fault_indices[in_stage] - stage_offset
    if fault_operation == "flip":
        values[selected_batch, selected_signal] = (
            1 - values[selected_batch, selected_signal]
        )
    else:
        values[selected_batch, selected_signal] = fault_values[in_stage].to(
            dtype=values.dtype
        )
    return values


def _eligible_stage_indices(number_stages: int, fault_scope: str) -> tuple[int, ...]:
    fault_scope = validate_fault_scope(fault_scope)
    if fault_scope == "input":
        return (0,)
    if fault_scope == "internal":
        return tuple(range(1, number_stages))
    return tuple(range(number_stages))


class ExactCountBitFlipModel(torch.nn.Module):
    """Existing model-2 LGN with exactly K transient signal faults per event.

    The global fault-site space is the concatenation of the Boolean outputs of
    FixedBinarization and every LogicDense layer. GroupSum is intentionally
    excluded: it is an arithmetic class accumulator, not a Boolean activation
    stage. Faults are transient and resampled independently for each forward
    pass; model parameters and TorchLogix connection indices are unchanged.
    """

    def __init__(
        self,
        base_model: torch.nn.Module,
        flip_count: int = 0,
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
        self.model_stage_widths = (input_width, *layer_widths)
        self.fault_scope = validate_fault_scope(fault_scope)
        self.fault_operation = validate_fault_operation(fault_operation)
        self._eligible_stage_indices = _eligible_stage_indices(
            len(self.model_stage_widths),
            self.fault_scope,
        )
        self.stage_widths = tuple(
            self.model_stage_widths[index] for index in self._eligible_stage_indices
        )
        offsets = [0]
        for width in self.stage_widths[:-1]:
            offsets.append(offsets[-1] + width)
        self.stage_offsets = tuple(offsets)
        self.total_fault_sites = sum(self.stage_widths)
        self.flip_count = validate_flip_count(flip_count, self.total_fault_sites)

    @property
    def fault_stages(self) -> int:
        return len(self.stage_widths)

    def _eligible_stage_index(self, model_stage_index: int) -> int | None:
        try:
            return self._eligible_stage_indices.index(model_stage_index)
        except ValueError:
            return None

    def _apply_faults_for_model_stage(
        self,
        values: torch.Tensor,
        fault_indices: torch.Tensor,
        fault_values: torch.Tensor | None,
        model_stage_index: int,
    ) -> torch.Tensor:
        eligible_index = self._eligible_stage_index(model_stage_index)
        if eligible_index is None:
            return values
        return apply_stage_faults(
            values,
            fault_indices,
            self.stage_offsets[eligible_index],
            fault_operation=self.fault_operation,
            fault_values=fault_values,
        )

    def set_flip_count(self, flip_count: int) -> None:
        self.flip_count = validate_flip_count(
            flip_count, self.total_fault_sites
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.flip_count == 0:
            return self.base_model(inputs)

        fault_indices = sample_unique_fault_indices(
            inputs.shape[0],
            self.total_fault_sites,
            self.flip_count,
            inputs.device,
        )
        fault_values = (
            torch.randint(
                0,
                2,
                fault_indices.shape,
                dtype=torch.int64,
                device=inputs.device,
            )
            if self.fault_operation == "random_set"
            else None
        )

        values = self.base_model.bin(inputs)
        values = self._apply_faults_for_model_stage(
            values, fault_indices, fault_values, 0
        )
        for layer_index, layer in enumerate(self.base_model.layers, start=1):
            values = layer(values)
            values = self._apply_faults_for_model_stage(
                values,
                fault_indices,
                fault_values,
                layer_index,
            )
        return self.base_model.group_sum(values)
