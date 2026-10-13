"""Canonical 30-point MAIA LNN architecture and temperature sweep.

The input schemas follow the MAIA dataset paper rather than the older CMS
SmartPixels profile convention:

* Model 2: 21-bin x profile, 13-bin y profile, and three position features;
* Model 3: 13x21 spatial charge cluster and the same three position features.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from typing import Literal


Family = Literal["dense", "conv"]

N_Y = 13
N_X = 21
N_POSITION_FEATURES = 3
N_DENSE_PROFILE_FEATURES = N_X + N_Y
N_DENSE_INPUT_FEATURES = N_DENSE_PROFILE_FEATURES + N_POSITION_FEATURES
N_CONV_CLUSTER_FEATURES = N_Y * N_X
N_CONV_INPUT_FEATURES = N_CONV_CLUSTER_FEATURES + N_POSITION_FEATURES


@dataclass(frozen=True)
class MaiaConfig:
    task_index: int
    alias: str
    family: Family
    size: str
    tau: float
    thresholds_per_feature: int
    dense_widths: tuple[int, ...]
    conv_kernels: tuple[int, ...] = ()
    tree_depth: int = 0
    lut_rank: int = 2

    @property
    def final_logic_width(self) -> int:
        return self.dense_widths[-1]

    @property
    def group_width_per_class(self) -> int:
        if self.final_logic_width % 2:
            raise ValueError(
                f"{self.alias}: final width {self.final_logic_width} is not divisible by 2"
            )
        return self.final_logic_width // 2

    @property
    def raw_input_features(self) -> int:
        return (
            N_DENSE_INPUT_FEATURES
            if self.family == "dense"
            else N_CONV_INPUT_FEATURES
        )

    @property
    def position_features(self) -> int:
        return N_POSITION_FEATURES

    def validate(self) -> None:
        if self.lut_rank != 2:
            raise ValueError(f"{self.alias}: only lut_rank=2 is permitted")
        if self.thresholds_per_feature < 1:
            raise ValueError(f"{self.alias}: thresholds must be positive")
        if not self.dense_widths or min(self.dense_widths) < 2:
            raise ValueError(f"{self.alias}: invalid dense widths")
        _ = self.group_width_per_class
        if self.family == "dense":
            previous = self.raw_input_features * self.thresholds_per_feature
        else:
            if len(self.conv_kernels) != 4 or self.tree_depth < 1:
                raise ValueError(f"{self.alias}: invalid convolutional configuration")
            # Four padded convolutions with one 2x2 pool and a stride-2 final
            # convolution map (13,21) -> (6,10) -> (3,5). Position features
            # are thermometer encoded and concatenated after convolution.
            previous = (
                self.conv_kernels[-1] * 3 * 5
                + N_POSITION_FEATURES * self.thresholds_per_feature
            )
        for width in self.dense_widths:
            if width * self.lut_rank < previous:
                raise ValueError(
                    f"{self.alias}: width {width} cannot cover {previous} rank-2 inputs"
                )
            previous = width

    def to_dict(self) -> dict:
        result = asdict(self)
        result["final_logic_width"] = self.final_logic_width
        result["group_width_per_class"] = self.group_width_per_class
        result["raw_input_features"] = self.raw_input_features
        result["thermometer_input_bits"] = (
            self.raw_input_features * self.thresholds_per_feature
        )
        return result


# Five increasing stems in each family. Every transition obeys the fixed
# rank-2 connection coverage constraint: 2*out_dim >= in_dim. These widths are
# intentionally larger than the v1 campaign because Model 2 now has the
# correct 37-feature input (3,700 thermometer bits), and Model 3 appends 300
# position bits after its convolutional trunk.
_DENSE_SIZES = (
    ("s01", (1900, 1900, 1900)),
    ("s02", (2600, 2600, 2600)),
    ("s03", (3600, 3600, 3600)),
    ("s04", (5000, 5000, 5000)),
    ("s05", (7000, 7000, 7000)),
)

_CONV_SIZES = (
    ("s01", (16, 32, 64, 64), (640, 640, 320)),
    ("s02", (24, 48, 96, 96), (896, 896, 448)),
    ("s03", (32, 64, 128, 128), (1152, 1152, 576)),
    ("s04", (48, 96, 192, 192), (1664, 1664, 832)),
    ("s05", (64, 128, 256, 256), (2112, 2112, 1056)),
)

TAUS = (10.0, 20.0, 40.0)
N_THRESHOLDS = 100


def _build_configs() -> tuple[MaiaConfig, ...]:
    rows: list[MaiaConfig] = []
    for size, widths in _DENSE_SIZES:
        for tau in TAUS:
            rows.append(
                MaiaConfig(
                    task_index=len(rows),
                    alias=f"dense_{size}_tau{int(tau)}",
                    family="dense",
                    size=size,
                    tau=tau,
                    thresholds_per_feature=N_THRESHOLDS,
                    dense_widths=widths,
                )
            )
    for size, kernels, widths in _CONV_SIZES:
        for tau in TAUS:
            rows.append(
                MaiaConfig(
                    task_index=len(rows),
                    alias=f"conv_{size}_tau{int(tau)}",
                    family="conv",
                    size=size,
                    tau=tau,
                    thresholds_per_feature=N_THRESHOLDS,
                    dense_widths=widths,
                    conv_kernels=kernels,
                    tree_depth=3,
                )
            )
    for row in rows:
        row.validate()
    if len(rows) != 30 or [row.task_index for row in rows] != list(range(30)):
        raise AssertionError("The MAIA campaign must contain exactly 30 ordered tasks")
    return tuple(rows)


CAMPAIGN_CONFIGS = _build_configs()
CAMPAIGN_BY_ALIAS = {row.alias: row for row in CAMPAIGN_CONFIGS}


def get_config(task_index: int) -> MaiaConfig:
    if not 0 <= task_index < len(CAMPAIGN_CONFIGS):
        raise ValueError(f"task index must be in [0, {len(CAMPAIGN_CONFIGS) - 1}]")
    return CAMPAIGN_CONFIGS[task_index]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-index", type=int)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if args.task_index is not None:
        row = get_config(args.task_index)
        print(json.dumps(row.to_dict(), indent=2) if args.json else row.alias)
        return

    for row in CAMPAIGN_CONFIGS:
        row.validate()
    if args.validate:
        print("Validated 30 MAIA tasks: 5 dense + 5 conv architectures, 3 taus each.")
        return

    print(
        "task\talias\tfamily\tsize\ttau\tthresholds\tconv_kernels\tdense_widths\t"
        "group_width"
    )
    for row in CAMPAIGN_CONFIGS:
        kernels = ",".join(map(str, row.conv_kernels)) or "-"
        widths = ",".join(map(str, row.dense_widths))
        print(
            f"{row.task_index}\t{row.alias}\t{row.family}\t{row.size}\t"
            f"{row.tau:g}\t{row.thresholds_per_feature}\t{kernels}\t{widths}\t"
            f"{row.group_width_per_class}"
        )


if __name__ == "__main__":
    main()
