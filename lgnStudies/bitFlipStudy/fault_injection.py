"""Exact-count transient activation-bit faults for dense model-2 LGNs."""

from __future__ import annotations

import torch


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
) -> torch.Tensor:
    """Apply the planned global faults that fall inside one signal stage."""
    if values.ndim != 2:
        raise ValueError(
            f"Expected a 2-D Boolean signal tensor, got shape {tuple(values.shape)}."
        )
    if fault_indices.ndim != 2 or fault_indices.shape[0] != values.shape[0]:
        raise ValueError("Fault plan batch dimension does not match activations.")
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
    values[selected_batch, selected_signal] = (
        1 - values[selected_batch, selected_signal]
    )
    return values


class ExactCountBitFlipModel(torch.nn.Module):
    """Existing model-2 LGN with exactly K transient signal faults per event.

    The global fault-site space is the concatenation of the Boolean outputs of
    FixedBinarization and every LogicDense layer. GroupSum is intentionally
    excluded: it is an arithmetic class accumulator, not a Boolean activation
    stage. Faults are transient and resampled independently for each forward
    pass; model parameters and TorchLogix connection indices are unchanged.
    """

    def __init__(self, base_model: torch.nn.Module, flip_count: int = 0):
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
        self.stage_widths = (input_width, *layer_widths)
        offsets = [0]
        for width in self.stage_widths[:-1]:
            offsets.append(offsets[-1] + width)
        self.stage_offsets = tuple(offsets)
        self.total_fault_sites = sum(self.stage_widths)
        self.flip_count = validate_flip_count(flip_count, self.total_fault_sites)

    @property
    def fault_stages(self) -> int:
        return len(self.stage_widths)

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

        values = self.base_model.bin(inputs)
        values = apply_stage_faults(
            values, fault_indices, self.stage_offsets[0]
        )
        for layer, offset in zip(
            self.base_model.layers, self.stage_offsets[1:]
        ):
            values = layer(values)
            values = apply_stage_faults(values, fault_indices, offset)
        return self.base_model.group_sum(values)
