"""Canonical 30-point MAIA LNN architecture and temperature sweep."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from typing import Literal


Family = Literal["dense", "conv"]


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
        return 13 if self.family == "dense" else 13 * 20

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
            # convolution map (13,20) -> (6,10) -> (3,5).
            previous = self.conv_kernels[-1] * 3 * 5
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


# Five increasing stems in each family.  Every transition obeys the fixed
# rank-2 connection coverage constraint: 2*out_dim >= in_dim.
_DENSE_SIZES = (
    ("s01", (650, 650, 650)),
    ("s02", (900, 900, 900)),
    ("s03", (1300, 1300, 1300)),
    ("s04", (1800, 1800, 1800)),
    ("s05", (2600, 2600, 2600)),
)

_CONV_SIZES = (
    ("s01", (16, 32, 64, 64), (512, 512, 256)),
    ("s02", (24, 48, 96, 96), (768, 768, 384)),
    ("s03", (32, 64, 128, 128), (1024, 1024, 512)),
    ("s04", (48, 96, 192, 192), (1536, 1536, 768)),
    ("s05", (64, 128, 256, 256), (2048, 2048, 1024)),
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
