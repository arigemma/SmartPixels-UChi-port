"""Differentiable random bit-flip injection for dense model-2 LGNs."""

from __future__ import annotations

import torch


def validate_flip_probability(probability: float) -> float:
    probability = float(probability)
    if not 0.0 <= probability <= 1.0:
        raise ValueError(
            f"Bit-flip probability must lie in [0, 1], got {probability}."
        )
    return probability


def random_bit_flip(values: torch.Tensor, probability: float) -> torch.Tensor:
    """Independently replace each value x by 1-x with probability p.

    Discrete inference values are exactly flipped between zero and one. During
    relaxed training the same operation mirrors a soft Boolean activation
    around 0.5 and remains differentiable.
    """
    probability = validate_flip_probability(probability)
    if probability == 0.0:
        return values
    if probability == 1.0:
        return 1 - values
    mask = torch.rand(values.shape, device=values.device) < probability
    return torch.where(mask, 1 - values, values)


class ActivationBitFlipModel(torch.nn.Module):
    """Model-2 forward path with faults after every binary signal stage.

    Faults are injected after FixedBinarization and after every LogicDense
    layer. GroupSum is not flipped because its outputs are integer-valued class
    accumulators rather than Boolean activation bits.
    """

    def __init__(self, base_model: torch.nn.Module, probability: float = 0.0):
        super().__init__()
        for attribute in ("bin", "layers", "group_sum"):
            if not hasattr(base_model, attribute):
                raise TypeError(
                    f"Expected a dense model-2 LGN with attribute `{attribute}`."
                )
        self.base_model = base_model
        self.probability = validate_flip_probability(probability)

    @property
    def fault_locations(self) -> int:
        return 1 + len(self.base_model.layers)

    def set_flip_probability(self, probability: float) -> None:
        self.probability = validate_flip_probability(probability)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        values = self.base_model.bin(inputs)
        values = random_bit_flip(values, self.probability)
        for layer in self.base_model.layers:
            values = layer(values)
            values = random_bit_flip(values, self.probability)
        return self.base_model.group_sum(values)
