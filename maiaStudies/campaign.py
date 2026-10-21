"""Canonical SmartPixels Model-2/Model-3 sweeps for the MAIA binary task.

The detector task changes from signed-pT classification to signal-versus-BIB,
but the deployed model inputs remain the established SmartPixels schemas:

* Model 2: y-local plus the 13-bin y profile from the final time slice;
* Model 3: y-local plus the 13x8 time-resolved y profile from slices 0--7.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace
from typing import Literal


Family = Literal["dense", "conv"]

N_Y = 13
N_X = 21
N_TIME = 20
LAST_FRAME = N_TIME - 1
N_SLICES = 8
N_POSITION_FEATURES = 1
N_DENSE_PROFILE_FEATURES = N_Y
N_DENSE_INPUT_FEATURES = N_POSITION_FEATURES + N_DENSE_PROFILE_FEATURES
N_CONV_PROFILE_FEATURES = N_Y * N_SLICES
N_CONV_INPUT_FEATURES = N_POSITION_FEATURES + N_CONV_PROFILE_FEATURES


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
            # Canonical Model 3: four padded convolutions, one 2x2 pool, and a
            # stride-2 final convolution: (13,8) -> (6,4) -> (3,2). The 100
            # thermometer bits for y-local bypass the convolutional trunk.
            previous = (
                self.conv_kernels[-1] * 3 * 2
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


# First five points of the established compact Model-2 left extension and
# Model-3 frontier sweeps. The final widths of odd-width three-class Model-2
# points are increased by one so the binary GroupSum receives equal groups.
# Every transition obeys 2*out_dim >= in_dim.
_DENSE_SIZES = (
    ("s01", (1269, 1269, 1270)),
    ("s02", (1617, 1617, 1618)),
    ("s03", (1914, 1914, 1914)),
    ("s04", (2175, 2175, 2176)),
    ("s05", (2412, 2412, 2412)),
)

_CONV_SIZES = (
    ("s01", (16, 32, 64, 60), (1200, 1200, 600, 300)),
    ("s02", (24, 48, 96, 90), (1800, 1800, 900, 450)),
    ("s03", (32, 64, 128, 120), (2400, 2400, 1200, 600)),
    ("s04", (40, 80, 160, 150), (3000, 3000, 1500, 750)),
    ("s05", (48, 96, 192, 180), (3600, 3600, 1800, 900)),
)

TAUS = (10.0, 20.0, 40.0)
TAU_EXTENSION = (1.0, 3.0)
N_THRESHOLDS = 100


def _build_configs(taus: tuple[float, ...]) -> tuple[MaiaConfig, ...]:
    rows: list[MaiaConfig] = []
    for size, widths in _DENSE_SIZES:
        for tau in taus:
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
        for tau in taus:
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
    if [row.task_index for row in rows] != list(range(len(rows))):
        raise AssertionError("MAIA campaign task indices must be contiguous")
    return tuple(rows)


CAMPAIGN_CONFIGS = _build_configs(TAUS)
TAU_EXTENSION_CONFIGS = _build_configs(TAU_EXTENSION)
ALL_CONFIGS = tuple(
    replace(row, task_index=index)
    for index, row in enumerate(
        sorted(
            CAMPAIGN_CONFIGS + TAU_EXTENSION_CONFIGS,
            key=lambda row: (0 if row.family == "dense" else 1, row.size, row.tau),
        )
    )
)
CAMPAIGNS = {
    "main": CAMPAIGN_CONFIGS,
    "tau_extension": TAU_EXTENSION_CONFIGS,
    "all": ALL_CONFIGS,
}
CAMPAIGN_BY_ALIAS = {row.alias: row for row in CAMPAIGN_CONFIGS}


def get_campaign_configs(campaign: str = "main") -> tuple[MaiaConfig, ...]:
    try:
        return CAMPAIGNS[campaign]
    except KeyError as exc:
        raise ValueError(
            f"Unknown MAIA campaign {campaign!r}; choose from {sorted(CAMPAIGNS)}"
        ) from exc


def get_config(task_index: int, campaign: str = "main") -> MaiaConfig:
    configs = get_campaign_configs(campaign)
    if not 0 <= task_index < len(configs):
        raise ValueError(f"task index must be in [0, {len(configs) - 1}]")
    return configs[task_index]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-index", type=int)
    parser.add_argument("--campaign", choices=tuple(CAMPAIGNS), default="main")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if args.task_index is not None:
        row = get_config(args.task_index, args.campaign)
        print(json.dumps(row.to_dict(), indent=2) if args.json else row.alias)
        return

    configs = get_campaign_configs(args.campaign)
    for row in configs:
        row.validate()
    if args.validate:
        taus = sorted({row.tau for row in configs})
        print(
            f"Validated {len(configs)} MAIA tasks for {args.campaign}: "
            f"taus={','.join(f'{tau:g}' for tau in taus)}."
        )
        return

    print(
        "task\talias\tfamily\tsize\ttau\tthresholds\tconv_kernels\tdense_widths\t"
        "group_width"
    )
    for row in configs:
        kernels = ",".join(map(str, row.conv_kernels)) or "-"
        widths = ",".join(map(str, row.dense_widths))
        print(
            f"{row.task_index}\t{row.alias}\t{row.family}\t{row.size}\t"
            f"{row.tau:g}\t{row.thresholds_per_feature}\t{kernels}\t{widths}\t"
            f"{row.group_width_per_class}"
        )


if __name__ == "__main__":
    main()
