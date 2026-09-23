"""EFLX-oriented LGN models for the RTL handoff campaign.

The campaign deliberately changes only three deployment-relevant quantities:
the number of ordered input thresholds, whether those thresholds are trained,
and the number of final Boolean values accumulated per class.  The stem
architectures are taken directly from the existing compact Model-2 and
Model-3 sweeps.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from .registry import register

try:
    from torchlogix.layers import (
        FixedBinarization,
        GroupSum,
        LogicConv2d,
        LogicDense,
        OrPooling2d,
    )
except ImportError:  # Keep registry discovery importable without torchlogix.
    FixedBinarization = GroupSum = LogicConv2d = LogicDense = OrPooling2d = None


def _merge_feature_and_threshold_dims(x: torch.Tensor, feature_dim: int) -> torch.Tensor:
    """Match TorchLogix's feature-major, threshold-minor flattening."""
    ndim = x.ndim
    if feature_dim < 0:
        feature_dim += ndim
    if not 0 <= feature_dim < ndim - 1:
        raise ValueError(
            f"feature_dim must select a non-final dimension; got {feature_dim} "
            f"for a rank-{ndim} tensor."
        )
    last = ndim - 1
    permutation = list(range(ndim))
    permutation.pop(last)
    permutation.insert(feature_dim + 1, last)
    merged = x.permute(*permutation)
    shape = list(merged.shape)
    shape[feature_dim] *= shape[feature_dim + 1]
    shape.pop(feature_dim + 1)
    return merged.reshape(*shape)


class OrderedLearnableBinarization(torch.nn.Module):
    """Learn ordered thresholds with identical train/eval reconstruction.

    TorchLogix's current ``LearnableBinarization`` reconstructs positive gaps
    differently in training and evaluation.  This campaign-local layer uses a
    single ordered parameterization in both modes.  Training emits soft bits;
    evaluation emits the exact hard thermometer bits used by deployment.
    """

    def __init__(
        self,
        thresholds: torch.Tensor,
        *,
        feature_dim: int = -2,
        temperature: float = 0.1,
        minimum_gap: float = 1e-6,
        max_grad_norm: float = 1e-3,
    ) -> None:
        super().__init__()
        values = torch.as_tensor(thresholds, dtype=torch.float32)
        if values.ndim != 2 or values.shape[-1] < 1:
            raise ValueError(
                "OrderedLearnableBinarization expects thresholds shaped "
                f"(features, thresholds_per_feature); got {tuple(values.shape)}."
            )
        if temperature <= 0 or minimum_gap <= 0 or max_grad_norm <= 0:
            raise ValueError("temperature, minimum_gap, and max_grad_norm must be positive.")

        values = torch.sort(values, dim=-1).values
        first = values[..., :1].clone()
        gaps = torch.diff(values, dim=-1).clamp_min(minimum_gap)
        scaled = gaps / temperature
        # Stable inverse softplus: x + log(1 - exp(-x)).
        raw_gaps = temperature * (scaled + torch.log(-torch.expm1(-scaled)))

        self.first_threshold = torch.nn.Parameter(first)
        self.raw_gaps = torch.nn.Parameter(raw_gaps)
        self.feature_dim = int(feature_dim)
        self.temperature = float(temperature)
        self.minimum_gap = float(minimum_gap)
        self.max_grad_norm = float(max_grad_norm)
        self.first_threshold.register_hook(self._clip_gradient)
        self.raw_gaps.register_hook(self._clip_gradient)

    def _clip_gradient(self, gradient: torch.Tensor) -> torch.Tensor:
        norm = gradient.norm()
        if torch.isfinite(norm) and norm > self.max_grad_norm:
            return gradient * (self.max_grad_norm / (norm + 1e-12))
        return gradient

    def get_thresholds(self) -> torch.Tensor:
        if self.raw_gaps.shape[-1] == 0:
            return self.first_threshold
        gaps = self.minimum_gap + self.temperature * torch.nn.functional.softplus(
            self.raw_gaps / self.temperature
        )
        return torch.cat(
            (self.first_threshold, self.first_threshold + torch.cumsum(gaps, dim=-1)),
            dim=-1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        thresholds = self.get_thresholds()
        expanded = x.unsqueeze(-1)
        if self.training:
            bits = torch.sigmoid((expanded - thresholds) / self.temperature)
        else:
            bits = (expanded > thresholds).to(dtype=x.dtype)
        return _merge_feature_and_threshold_dims(bits, self.feature_dim)


def _logic_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _reduction_widths(in_dim: int, out_dim: int) -> list[int]:
    """Return the shortest rank-2 coverage-safe path to ``out_dim``.

    Fixed TorchLogix connections require ``2 * out_dim >= in_dim`` so every
    input can be used.  A narrow GroupSum therefore needs one or more gradual
    reduction layers rather than a single invalid jump.
    """
    if in_dim < 2 or out_dim < 2:
        raise ValueError("Logic reduction dimensions must be at least two.")
    widths = []
    current = int(in_dim)
    while 2 * out_dim < current:
        current = math.ceil(current / 2)
        widths.append(current)
    widths.append(int(out_dim))
    return widths


class _RtlHandoffMixin:
    alias = ""
    family = ""
    base_architecture = ""
    n_bits = 63
    binarization_kind = "fixed"
    group_width = 128
    tau = 1.0
    n_input_features = 0

    def _make_binarizer(self, thresholds: torch.Tensor) -> torch.nn.Module:
        if FixedBinarization is None:
            raise ImportError("torchlogix is required to instantiate an RTL handoff LGN.")
        expected = (self.n_input_features, self.n_bits)
        if tuple(thresholds.shape) != expected:
            raise ValueError(
                f"{self.alias} expects thresholds shaped {expected}; "
                f"got {tuple(thresholds.shape)}."
            )
        if self.binarization_kind == "fixed":
            return FixedBinarization(thresholds=thresholds)
        if self.binarization_kind == "learnable":
            return OrderedLearnableBinarization(thresholds=thresholds)
        raise ValueError(f"Unknown binarization kind {self.binarization_kind!r}.")

    @classmethod
    def deployment_config(cls) -> dict[str, Any]:
        code_width = math.ceil(math.log2(cls.n_bits + 1))
        return {
            "alias": cls.alias,
            "family": cls.family,
            "base_architecture": cls.base_architecture,
            "thresholds_per_feature": int(cls.n_bits),
            "compact_code_bits_per_feature": int(code_width),
            "raw_input_features": int(cls.n_input_features),
            "thermometer_input_bits": int(cls.n_input_features * cls.n_bits),
            "compact_input_bits": int(cls.n_input_features * code_width),
            "binarization_training": cls.binarization_kind,
            "deployment_binarization": "fixed",
            "group_width_per_class": int(cls.group_width),
            "final_logic_width": int(3 * cls.group_width),
            "tau": float(cls.tau),
            "lut_rank": 2,
        }


class RtlHandoffDenseLGN(_RtlHandoffMixin, torch.nn.Module):
    family = "dense"
    n_input_features = 14
    stem_width = 1269

    def __init__(self, thresholds: torch.Tensor) -> None:
        super().__init__()
        device = _logic_device()
        kwargs = {"weight_init": "residual"}
        self.bin = self._make_binarizer(thresholds)
        widths = [self.stem_width, self.stem_width]
        widths.extend(_reduction_widths(self.stem_width, 3 * self.group_width))
        layers = []
        in_dim = self.n_input_features * self.n_bits
        for out_dim in widths:
            layers.append(
                LogicDense(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    device=device,
                    parametrization_kwargs=kwargs,
                    lut_rank=2,
                )
            )
            in_dim = out_dim
        self.layers = torch.nn.ModuleList(layers)
        self.group_sum = GroupSum(3, tau=float(self.tau), device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[-1] != self.n_input_features:
            raise ValueError(
                f"{self.alias} expects (batch, {self.n_input_features}); got {tuple(x.shape)}."
            )
        value = self.bin(x)
        for layer in self.layers:
            value = layer(value)
        return self.group_sum(value)

    @classmethod
    def deployment_config(cls) -> dict[str, Any]:
        result = super().deployment_config()
        result["stem_widths"] = [int(cls.stem_width), int(cls.stem_width)]
        result["pre_group_reduction_widths"] = _reduction_widths(
            int(cls.stem_width), 3 * int(cls.group_width)
        )
        result["dense_widths"] = (
            result["stem_widths"] + result["pre_group_reduction_widths"]
        )
        return result


class RtlHandoffConvLGN(_RtlHandoffMixin, torch.nn.Module):
    family = "conv"
    n_input_features = 105
    kernels = (16, 32, 64, 60)
    fc_base = 300

    def __init__(self, thresholds: torch.Tensor) -> None:
        super().__init__()
        device = _logic_device()
        kwargs = {"weight_init": "residual"}
        k1, k2, k3, k4 = self.kernels
        self.bin = self._make_binarizer(thresholds)
        self.conv1 = LogicConv2d(
            in_dim=(13, 8), channels=self.n_bits, num_kernels=k1,
            receptive_field_size=3, tree_depth=3, stride=1, padding=1,
            device=device, parametrization_kwargs=kwargs, lut_rank=2,
        )
        self.conv2 = LogicConv2d(
            in_dim=(13, 8), channels=k1, num_kernels=k2,
            receptive_field_size=3, tree_depth=3, stride=1, padding=1,
            device=device, parametrization_kwargs=kwargs, lut_rank=2,
        )
        self.or_pool1 = OrPooling2d(kernel_size=2, stride=2, padding=0)
        self.conv3 = LogicConv2d(
            in_dim=(6, 4), channels=k2, num_kernels=k3,
            receptive_field_size=3, tree_depth=3, stride=1, padding=1,
            device=device, parametrization_kwargs=kwargs, lut_rank=2,
        )
        self.conv4 = LogicConv2d(
            in_dim=(6, 4), channels=k3, num_kernels=k4,
            receptive_field_size=3, tree_depth=3, stride=2, padding=1,
            device=device, parametrization_kwargs=kwargs, lut_rank=2,
        )
        dense_input = self.n_bits + k4 * 3 * 2
        hidden1 = 4 * self.fc_base
        hidden2 = 2 * self.fc_base
        self.fc1 = LogicDense(
            in_dim=dense_input, out_dim=hidden1, device=device,
            parametrization_kwargs=kwargs, lut_rank=2,
        )
        self.fc2 = LogicDense(
            in_dim=hidden1, out_dim=hidden1, device=device,
            parametrization_kwargs=kwargs, lut_rank=2,
        )
        self.fc3 = LogicDense(
            in_dim=hidden1, out_dim=hidden2, device=device,
            parametrization_kwargs=kwargs, lut_rank=2,
        )
        reduction_widths = _reduction_widths(hidden2, 3 * self.group_width)
        reduction_layers = []
        reduction_input = hidden2
        for reduction_output in reduction_widths[:-1]:
            reduction_layers.append(
                LogicDense(
                    in_dim=reduction_input,
                    out_dim=reduction_output,
                    device=device,
                    parametrization_kwargs=kwargs,
                    lut_rank=2,
                )
            )
            reduction_input = reduction_output
        self.reduction_fcs = torch.nn.ModuleList(reduction_layers)
        self.fc4 = LogicDense(
            in_dim=reduction_input, out_dim=3 * self.group_width, device=device,
            parametrization_kwargs=kwargs, lut_rank=2,
        )
        self.group_sum = GroupSum(3, tau=float(self.tau), device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[-1] != self.n_input_features:
            raise ValueError(
                f"{self.alias} expects (batch, {self.n_input_features}); got {tuple(x.shape)}."
            )
        value = self.bin(x)
        y0 = value[:, : self.n_bits]
        profile = value[:, self.n_bits :].reshape(-1, self.n_bits, 13, 8)
        value = self.conv1(profile)
        value = self.conv2(value)
        value = self.or_pool1(value)
        value = self.conv3(value)
        value = self.conv4(value)
        value = value.reshape(value.shape[0], -1)
        value = torch.cat((y0, value), dim=1)
        value = self.fc3(self.fc2(self.fc1(value)))
        for layer in self.reduction_fcs:
            value = layer(value)
        value = self.fc4(value)
        return self.group_sum(value)

    @classmethod
    def deployment_config(cls) -> dict[str, Any]:
        result = super().deployment_config()
        result["conv_kernels"] = [int(value) for value in cls.kernels]
        result["fc_base"] = int(cls.fc_base)
        reductions = _reduction_widths(
            2 * int(cls.fc_base), 3 * int(cls.group_width)
        )
        result["pre_group_reduction_widths"] = reductions
        result["dense_widths"] = [
            4 * int(cls.fc_base),
            4 * int(cls.fc_base),
            2 * int(cls.fc_base),
        ] + reductions
        return result


# Ten dense and ten convolutional screening points.  ``tau`` is reduced with
# GroupSum width so the training-logit scale remains close to its established
# parent model.  Synthesis later emits unscaled integer class counts.
RTL_HANDOFF_CONFIGS: tuple[dict[str, Any], ...] = (
    # Dense s01: established best tau=20 and original group width 423.
    dict(alias="dense_s01_t100_fixed_g423", family="dense", base="s01_p005M", stem=1269, bits=100, bin="fixed", group=423, tau=20.0),
    dict(alias="dense_s01_t63_fixed_g423", family="dense", base="s01_p005M", stem=1269, bits=63, bin="fixed", group=423, tau=20.0),
    dict(alias="dense_s01_t31_fixed_g423", family="dense", base="s01_p005M", stem=1269, bits=31, bin="fixed", group=423, tau=20.0),
    dict(alias="dense_s01_t63_learn_g423", family="dense", base="s01_p005M", stem=1269, bits=63, bin="learnable", group=423, tau=20.0),
    dict(alias="dense_s01_t100_fixed_g128", family="dense", base="s01_p005M", stem=1269, bits=100, bin="fixed", group=128, tau=6.0),
    dict(alias="dense_s01_t63_fixed_g128", family="dense", base="s01_p005M", stem=1269, bits=63, bin="fixed", group=128, tau=6.0),
    dict(alias="dense_s01_t63_learn_g128", family="dense", base="s01_p005M", stem=1269, bits=63, bin="learnable", group=128, tau=6.0),
    dict(alias="dense_s02_t63_learn_g128", family="dense", base="s02_p008M", stem=1617, bits=63, bin="learnable", group=128, tau=5.0),
    dict(alias="dense_s03_t63_learn_g128", family="dense", base="s03_p010M", stem=1914, bits=63, bin="learnable", group=128, tau=2.0),
    dict(alias="dense_s03_t31_learn_g64", family="dense", base="s03_p010M", stem=1914, bits=31, bin="learnable", group=64, tau=1.0),
    # Compact Model-3 families.  Fifteen thresholds approximately match the
    # existing dense 100-threshold total input dimensionality.
    dict(alias="conv_s01_t100_fixed_g100", family="conv", base="s01_p003M", kernels=(16, 32, 64, 60), fc=300, bits=100, bin="fixed", group=100, tau=10.0),
    dict(alias="conv_s01_t63_fixed_g100", family="conv", base="s01_p003M", kernels=(16, 32, 64, 60), fc=300, bits=63, bin="fixed", group=100, tau=10.0),
    dict(alias="conv_s01_t31_fixed_g100", family="conv", base="s01_p003M", kernels=(16, 32, 64, 60), fc=300, bits=31, bin="fixed", group=100, tau=10.0),
    dict(alias="conv_s01_t15_fixed_g100", family="conv", base="s01_p003M", kernels=(16, 32, 64, 60), fc=300, bits=15, bin="fixed", group=100, tau=10.0),
    dict(alias="conv_s01_t15_learn_g100", family="conv", base="s01_p003M", kernels=(16, 32, 64, 60), fc=300, bits=15, bin="learnable", group=100, tau=10.0),
    dict(alias="conv_s01_t15_learn_g32", family="conv", base="s01_p003M", kernels=(16, 32, 64, 60), fc=300, bits=15, bin="learnable", group=32, tau=3.0),
    dict(alias="conv_s02_t15_fixed_g64", family="conv", base="s02_p007M", kernels=(24, 48, 96, 90), fc=450, bits=15, bin="fixed", group=64, tau=4.0),
    dict(alias="conv_s02_t15_learn_g64", family="conv", base="s02_p007M", kernels=(24, 48, 96, 90), fc=450, bits=15, bin="learnable", group=64, tau=4.0),
    dict(alias="conv_s02_t31_learn_g64", family="conv", base="s02_p007M", kernels=(24, 48, 96, 90), fc=450, bits=31, bin="learnable", group=64, tau=4.0),
    dict(alias="conv_s03_t15_learn_g64", family="conv", base="s03_p012M", kernels=(32, 64, 128, 120), fc=600, bits=15, bin="learnable", group=64, tau=3.0),
)


RTL_HANDOFF_BY_ALIAS: dict[str, dict[str, Any]] = {}


def _class_name(alias: str) -> str:
    return "RtlHandoff" + "".join(part.capitalize() for part in alias.split("_"))


for _config in RTL_HANDOFF_CONFIGS:
    _alias = str(_config["alias"])
    _family = str(_config["family"])
    _registry_name = f"lgn-rtl-handoff-{_alias.replace('_', '-')}"
    _attributes: dict[str, Any] = {
        "__module__": __name__,
        "__qualname__": _class_name(_alias),
        "alias": _alias,
        "base_architecture": str(_config["base"]),
        "n_bits": int(_config["bits"]),
        "binarization_kind": str(_config["bin"]),
        "group_width": int(_config["group"]),
        "tau": float(_config["tau"]),
    }
    if _family == "dense":
        _base_class = RtlHandoffDenseLGN
        _attributes["stem_width"] = int(_config["stem"])
        _input_type = "y-profile"
    else:
        _base_class = RtlHandoffConvLGN
        _attributes["kernels"] = tuple(int(value) for value in _config["kernels"])
        _attributes["fc_base"] = int(_config["fc"])
        _input_type = "y-profile-timing"
    _generated = type(_class_name(_alias), (_base_class,), _attributes)
    _generated = register(
        _registry_name,
        input_type=_input_type,
        task_type="classification",
    )(_generated)
    globals()[_generated.__name__] = _generated
    _entry = dict(_config)
    _entry["registry_name"] = _registry_name
    _entry["class_name"] = _generated.__name__
    RTL_HANDOFF_BY_ALIAS[_alias] = _entry


def get_rtl_handoff_config(alias: str) -> dict[str, Any]:
    try:
        return dict(RTL_HANDOFF_BY_ALIAS[alias])
    except KeyError as exc:
        raise ValueError(
            f"Unknown RTL handoff alias {alias!r}; expected one of "
            f"{sorted(RTL_HANDOFF_BY_ALIAS)}."
        ) from exc
