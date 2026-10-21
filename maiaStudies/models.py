"""Binary fixed-threshold LNNs for the MAIA signal/BIB task."""

from __future__ import annotations

import math

import torch

from .campaign import (
    MaiaConfig,
    N_CONV_INPUT_FEATURES,
    N_DENSE_INPUT_FEATURES,
    N_POSITION_FEATURES,
    N_SLICES,
    N_Y,
)

try:
    from torchlogix.layers import (
        FixedBinarization,
        GroupSum,
        LogicConv2d,
        LogicDense,
        OrPooling2d,
    )
except ImportError as exc:  # pragma: no cover - exercised by cluster preflight
    FixedBinarization = GroupSum = LogicConv2d = LogicDense = OrPooling2d = None
    _TORCHLOGIX_IMPORT_ERROR = exc
else:
    _TORCHLOGIX_IMPORT_ERROR = None


def _require_torchlogix() -> None:
    if _TORCHLOGIX_IMPORT_ERROR is not None:
        raise ImportError(
            "torchlogix is required. Activate smart_pixels and ensure the local "
            "torchlogix checkout is installed or on PYTHONPATH."
        ) from _TORCHLOGIX_IMPORT_ERROR


def _device_name(device: torch.device | str) -> str:
    return torch.device(device).type


def _logic_dense(in_dim: int, out_dim: int, device: str) -> torch.nn.Module:
    return LogicDense(
        in_dim=in_dim,
        out_dim=out_dim,
        device=device,
        lut_rank=2,
        parametrization_kwargs={"weight_init": "residual"},
    )


class MaiaDenseLNN(torch.nn.Module):
    """Model-2 LNN over y0 plus the final-frame 13-bin y profile."""

    def __init__(
        self, config: MaiaConfig, thresholds: dict[str, torch.Tensor], device: str
    ):
        super().__init__()
        _require_torchlogix()
        if config.family != "dense":
            raise ValueError(f"Expected dense configuration, got {config.family}")
        if set(thresholds) != {"dense"}:
            raise ValueError(f"Dense thresholds require only 'dense', got {set(thresholds)}")
        dense_thresholds = thresholds["dense"]
        expected = (N_DENSE_INPUT_FEATURES, config.thresholds_per_feature)
        if tuple(dense_thresholds.shape) != expected:
            raise ValueError(f"Dense thresholds must have shape {expected}")
        logic_device = _device_name(device)
        self.config = config
        self.binarization = FixedBinarization(thresholds=dense_thresholds)
        layers = []
        in_dim = N_DENSE_INPUT_FEATURES * config.thresholds_per_feature
        for out_dim in config.dense_widths:
            layers.append(_logic_dense(in_dim, out_dim, logic_device))
            in_dim = out_dim
        self.layers = torch.nn.ModuleList(layers)
        self.group_sum = GroupSum(2, tau=config.tau, device=logic_device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != N_DENSE_INPUT_FEATURES:
            raise ValueError(
                f"Dense MAIA input must be (batch,{N_DENSE_INPUT_FEATURES}), "
                f"got {tuple(x.shape)}"
            )
        value = self.binarization(x)
        for layer in self.layers:
            value = layer(value)
        return self.group_sum(value)


class MaiaConvLNN(torch.nn.Module):
    """Model-3 LNN over y0 plus the 13x8 time-resolved y profile."""

    def __init__(
        self, config: MaiaConfig, thresholds: dict[str, torch.Tensor], device: str
    ):
        super().__init__()
        _require_torchlogix()
        if config.family != "conv":
            raise ValueError(f"Expected conv configuration, got {config.family}")
        if set(thresholds) != {"conv"}:
            raise ValueError(f"Conv thresholds require only 'conv', got {set(thresholds)}")
        conv_thresholds = thresholds["conv"]
        expected = (N_CONV_INPUT_FEATURES, config.thresholds_per_feature)
        if tuple(conv_thresholds.shape) != expected:
            raise ValueError(f"Conv thresholds must have shape {expected}")
        logic_device = _device_name(device)
        self.config = config
        kwargs = {
            "device": logic_device,
            "lut_rank": 2,
            "parametrization_kwargs": {"weight_init": "residual"},
        }
        k1, k2, k3, k4 = config.conv_kernels
        self.binarization = FixedBinarization(thresholds=conv_thresholds)
        self.conv1 = LogicConv2d(
            in_dim=(N_Y, N_SLICES), channels=config.thresholds_per_feature,
            num_kernels=k1, receptive_field_size=3, tree_depth=3,
            stride=1, padding=1, **kwargs,
        )
        self.conv2 = LogicConv2d(
            in_dim=(N_Y, N_SLICES), channels=k1, num_kernels=k2,
            receptive_field_size=3, tree_depth=3, stride=1, padding=1, **kwargs,
        )
        self.pool = OrPooling2d(kernel_size=2, stride=2, padding=0)
        self.conv3 = LogicConv2d(
            in_dim=(6, 4), channels=k2, num_kernels=k3,
            receptive_field_size=3, tree_depth=3, stride=1, padding=1, **kwargs,
        )
        self.conv4 = LogicConv2d(
            in_dim=(6, 4), channels=k3, num_kernels=k4,
            receptive_field_size=3, tree_depth=3, stride=2, padding=1, **kwargs,
        )
        layers = []
        in_dim = k4 * 3 * 2 + N_POSITION_FEATURES * config.thresholds_per_feature
        for out_dim in config.dense_widths:
            layers.append(_logic_dense(in_dim, out_dim, logic_device))
            in_dim = out_dim
        self.layers = torch.nn.ModuleList(layers)
        self.group_sum = GroupSum(2, tau=config.tau, device=logic_device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != N_CONV_INPUT_FEATURES:
            raise ValueError(
                f"Convolutional MAIA input must be (batch,{N_CONV_INPUT_FEATURES}), "
                f"got {tuple(x.shape)}"
            )
        bits = self.binarization(x)
        y0_bits = bits[:, : self.config.thresholds_per_feature]
        value = bits[:, self.config.thresholds_per_feature :].reshape(
            -1, self.config.thresholds_per_feature, N_Y, N_SLICES
        )
        value = self.conv1(value)
        value = self.conv2(value)
        value = self.pool(value)
        value = self.conv3(value)
        value = self.conv4(value)
        value = value.reshape(value.shape[0], -1)
        value = torch.cat((y0_bits, value), dim=1)
        for layer in self.layers:
            value = layer(value)
        return self.group_sum(value)


def build_model(
    config: MaiaConfig,
    thresholds: dict[str, torch.Tensor],
    device: torch.device | str,
) -> torch.nn.Module:
    config.validate()
    if config.family == "dense":
        return MaiaDenseLNN(config, thresholds, str(device))
    return MaiaConvLNN(config, thresholds, str(device))


def resource_proxies(config: MaiaConfig) -> dict[str, int]:
    """Return architecture counts, explicitly not post-synthesis LUT/FF values."""
    dense_nodes = sum(config.dense_widths)
    if config.family == "dense":
        logic_nodes = dense_nodes
    else:
        # Each depth-3 binary tree has seven rank-2 logic nodes at every
        # output position. Padded convs preserve dimensions except conv4.
        k1, k2, k3, k4 = config.conv_kernels
        conv_nodes = 7 * (
            k1 * N_Y * N_SLICES
            + k2 * N_Y * N_SLICES
            + k3 * 6 * 4
            + k4 * 3 * 2
        )
        # A four-input OR pool can be represented by three rank-2 OR nodes.
        pool_nodes = 3 * k2 * 6 * 4
        logic_nodes = conv_nodes + pool_nodes + dense_nodes
    return {
        "logic_node_proxy": int(logic_nodes),
        "group_sum_inputs": int(config.final_logic_width),
        "thermometer_input_bits": int(
            config.raw_input_features * config.thresholds_per_feature
        ),
        "compact_ordinal_input_bits": int(
            config.raw_input_features
            * math.ceil(math.log2(config.thresholds_per_feature + 1))
        ),
    }
