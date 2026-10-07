"""Binary fixed-threshold LNNs for the MAIA signal/BIB task."""

from __future__ import annotations

import math

import torch

from .campaign import MaiaConfig

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
    """Model-2-style LNN over a 13-bin time-integrated charge profile."""

    def __init__(self, config: MaiaConfig, thresholds: torch.Tensor, device: str):
        super().__init__()
        _require_torchlogix()
        if config.family != "dense":
            raise ValueError(f"Expected dense configuration, got {config.family}")
        expected = (13, config.thresholds_per_feature)
        if tuple(thresholds.shape) != expected:
            raise ValueError(f"Dense thresholds must have shape {expected}")
        logic_device = _device_name(device)
        self.config = config
        self.binarization = FixedBinarization(thresholds=thresholds)
        layers = []
        in_dim = 13 * config.thresholds_per_feature
        for out_dim in config.dense_widths:
            layers.append(_logic_dense(in_dim, out_dim, logic_device))
            in_dim = out_dim
        self.layers = torch.nn.ModuleList(layers)
        self.group_sum = GroupSum(2, tau=config.tau, device=logic_device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != 13:
            raise ValueError(f"Dense MAIA input must be (batch,13), got {tuple(x.shape)}")
        value = self.binarization(x)
        for layer in self.layers:
            value = layer(value)
        return self.group_sum(value)


class MaiaConvLNN(torch.nn.Module):
    """Model-3-style LNN over the complete 13x20 charge time-profile."""

    def __init__(self, config: MaiaConfig, thresholds: torch.Tensor, device: str):
        super().__init__()
        _require_torchlogix()
        if config.family != "conv":
            raise ValueError(f"Expected conv configuration, got {config.family}")
        expected = (1, config.thresholds_per_feature)
        if tuple(thresholds.shape) != expected:
            raise ValueError(f"Conv thresholds must have shape {expected}")
        logic_device = _device_name(device)
        kwargs = {
            "device": logic_device,
            "lut_rank": 2,
            "parametrization_kwargs": {"weight_init": "residual"},
        }
        k1, k2, k3, k4 = config.conv_kernels
        # feature_dim=1 folds the threshold axis into the single input channel:
        # (B,1,13,20,100) -> (B,100,13,20).
        self.binarization = FixedBinarization(thresholds=thresholds, feature_dim=1)
        self.conv1 = LogicConv2d(
            in_dim=(13, 20), channels=config.thresholds_per_feature,
            num_kernels=k1, receptive_field_size=3, tree_depth=3,
            stride=1, padding=1, **kwargs,
        )
        self.conv2 = LogicConv2d(
            in_dim=(13, 20), channels=k1, num_kernels=k2,
            receptive_field_size=3, tree_depth=3, stride=1, padding=1, **kwargs,
        )
        self.pool = OrPooling2d(kernel_size=2, stride=2, padding=0)
        self.conv3 = LogicConv2d(
            in_dim=(6, 10), channels=k2, num_kernels=k3,
            receptive_field_size=3, tree_depth=3, stride=1, padding=1, **kwargs,
        )
        self.conv4 = LogicConv2d(
            in_dim=(6, 10), channels=k3, num_kernels=k4,
            receptive_field_size=3, tree_depth=3, stride=2, padding=1, **kwargs,
        )
        layers = []
        in_dim = k4 * 3 * 5
        for out_dim in config.dense_widths:
            layers.append(_logic_dense(in_dim, out_dim, logic_device))
            in_dim = out_dim
        self.layers = torch.nn.ModuleList(layers)
        self.group_sum = GroupSum(2, tau=config.tau, device=logic_device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or tuple(x.shape[1:]) != (13, 20):
            raise ValueError(
                f"Convolutional MAIA input must be (batch,13,20), got {tuple(x.shape)}"
            )
        value = self.binarization(x.unsqueeze(1))
        value = self.conv1(value)
        value = self.conv2(value)
        value = self.pool(value)
        value = self.conv3(value)
        value = self.conv4(value)
        value = value.reshape(value.shape[0], -1)
        for layer in self.layers:
            value = layer(value)
        return self.group_sum(value)


def build_model(
    config: MaiaConfig, thresholds: torch.Tensor, device: torch.device | str
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
            k1 * 13 * 20 + k2 * 13 * 20 + k3 * 6 * 10 + k4 * 3 * 5
        )
        # A four-input OR pool can be represented by three rank-2 OR nodes.
        pool_nodes = 3 * k2 * 6 * 10
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
