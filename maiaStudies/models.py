"""Binary fixed-threshold LNNs for the MAIA signal/BIB task."""

from __future__ import annotations

import math

import torch

from .campaign import (
    MaiaConfig,
    N_CONV_CLUSTER_FEATURES,
    N_CONV_INPUT_FEATURES,
    N_DENSE_INPUT_FEATURES,
    N_POSITION_FEATURES,
    N_X,
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
    """Model-2 LNN over x/y charge profiles and detector position."""

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
    """Model-3 LNN over a 13x21 spatial cluster plus position branch."""

    def __init__(
        self, config: MaiaConfig, thresholds: dict[str, torch.Tensor], device: str
    ):
        super().__init__()
        _require_torchlogix()
        if config.family != "conv":
            raise ValueError(f"Expected conv configuration, got {config.family}")
        if set(thresholds) != {"cluster", "position"}:
            raise ValueError(
                "Conv thresholds require 'cluster' and 'position', got "
                f"{set(thresholds)}"
            )
        cluster_thresholds = thresholds["cluster"]
        position_thresholds = thresholds["position"]
        cluster_expected = (1, config.thresholds_per_feature)
        position_expected = (N_POSITION_FEATURES, config.thresholds_per_feature)
        if tuple(cluster_thresholds.shape) != cluster_expected:
            raise ValueError(f"Conv cluster thresholds must have shape {cluster_expected}")
        if tuple(position_thresholds.shape) != position_expected:
            raise ValueError(
                f"Conv position thresholds must have shape {position_expected}"
            )
        logic_device = _device_name(device)
        kwargs = {
            "device": logic_device,
            "lut_rank": 2,
            "parametrization_kwargs": {"weight_init": "residual"},
        }
        k1, k2, k3, k4 = config.conv_kernels
        # feature_dim=1 folds the threshold axis into the single input channel:
        # (B,1,13,21,100) -> (B,100,13,21).
        self.cluster_binarization = FixedBinarization(
            thresholds=cluster_thresholds, feature_dim=1
        )
        self.position_binarization = FixedBinarization(
            thresholds=position_thresholds
        )
        self.conv1 = LogicConv2d(
            in_dim=(N_Y, N_X), channels=config.thresholds_per_feature,
            num_kernels=k1, receptive_field_size=3, tree_depth=3,
            stride=1, padding=1, **kwargs,
        )
        self.conv2 = LogicConv2d(
            in_dim=(N_Y, N_X), channels=k1, num_kernels=k2,
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
        in_dim = k4 * 3 * 5 + N_POSITION_FEATURES * config.thresholds_per_feature
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
        cluster = x[:, :N_CONV_CLUSTER_FEATURES].reshape(-1, N_Y, N_X)
        position = x[:, N_CONV_CLUSTER_FEATURES:]
        value = self.cluster_binarization(cluster.unsqueeze(1))
        value = self.conv1(value)
        value = self.conv2(value)
        value = self.pool(value)
        value = self.conv3(value)
        value = self.conv4(value)
        value = value.reshape(value.shape[0], -1)
        position_bits = self.position_binarization(position)
        value = torch.cat((value, position_bits), dim=1)
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
            k1 * N_Y * N_X + k2 * N_Y * N_X + k3 * 6 * 10 + k4 * 3 * 5
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
