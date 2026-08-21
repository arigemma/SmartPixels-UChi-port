"""The fixed, QKeras-matched HGQ2 Model-2 campaign registry."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class HGQModelConfig:
    index: int
    alias: str
    hidden_units: tuple[int, ...]
    input_type: str
    beta0: float
    qkeras_peer: str
    reference_lut: int
    reference_ff: int
    reference_balanced_accuracy: float

    @property
    def reference_resources(self) -> int:
        return self.reference_lut + self.reference_ff

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["hidden_units"] = list(self.hidden_units)
        payload["reference_resources_lut_plus_ff"] = self.reference_resources
        return payload


# These beta values provide five monotonically weaker HGQ bit-cost penalties.
# They are paired with the QKeras W/A labels for bookkeeping only: HGQ learns
# its bit widths and does not impose those fixed QKeras precisions.
BETA_BY_PRECISION = {
    "w2a4": 2.0e-5,
    "w2a6": 1.0e-5,
    "w3a6": 5.0e-6,
    "w4a8": 2.0e-6,
    "w5a10": 1.0e-6,
}


def _make_family(
    start_index: int,
    family_alias: str,
    qkeras_template: str,
    hidden_units: tuple[int, ...],
    input_type: str,
    references: tuple[tuple[int, int, float], ...],
) -> list[HGQModelConfig]:
    precisions = tuple(BETA_BY_PRECISION)
    if len(references) != len(precisions):
        raise ValueError("Each five-point family must define five QKeras references.")
    return [
        HGQModelConfig(
            index=start_index + offset,
            alias=f"{family_alias}_{precision}_b{BETA_BY_PRECISION[precision]:.0e}".replace("-", "m"),
            hidden_units=hidden_units,
            input_type=input_type,
            beta0=BETA_BY_PRECISION[precision],
            qkeras_peer=qkeras_template.format(precision=precision),
            reference_lut=references[offset][0],
            reference_ff=references[offset][1],
            reference_balanced_accuracy=references[offset][2],
        )
        for offset, precision in enumerate(precisions)
    ]


MODEL_CONFIGS = tuple(
    _make_family(
        0,
        "adc2a_d128",
        "qkeras-model-2-{precision}-adc2a",
        (128,),
        "y-profile-adc2a",
        (
            (14250, 5255, 0.720),
            (14216, 5229, 0.718),
            (25354, 10482, 0.741),
            (34391, 14824, 0.750),
            (48864, 21596, 0.752),
        ),
    )
    + _make_family(
        5,
        "yprofile_d128",
        "qkeras-model-2-yprofile-d128-{precision}",
        (128,),
        "y-profile",
        (
            (30489, 12890, 0.731),
            (31466, 14049, 0.730),
            (45907, 20215, 0.771),
            (53237, 25140, 0.781),
            (69183, 30304, 0.786),
        ),
    )
    + _make_family(
        10,
        "yprofile_d192",
        "qkeras-model-2-yprofile-d192-{precision}",
        (192,),
        "y-profile",
        (
            (50198, 25503, 0.744),
            (45246, 21001, 0.732),
            (64593, 28658, 0.774),
            (76097, 35144, 0.785),
            (96399, 44720, 0.790),
        ),
    )
    + _make_family(
        15,
        "yprofile_d256_d64",
        "qkeras-model-2-yprofile-d256-d64-{precision}",
        (256, 64),
        "y-profile",
        (
            (100710, 36257, 0.768),
            (108707, 37981, 0.764),
            (140759, 49732, 0.786),
            (201428, 74387, 0.798),
            (269992, 108107, 0.797),
        ),
    )
    + [
        HGQModelConfig(
            index=20,
            alias="yprofile_d768_d256_d128_w5a10_b1em06",
            hidden_units=(768, 256, 128),
            input_type="y-profile",
            beta0=BETA_BY_PRECISION["w5a10"],
            qkeras_peer="qkeras-model-2-yprofile-d768-d256-d128-w5a10",
            reference_lut=441737,
            reference_ff=384730,
            reference_balanced_accuracy=0.811,
        )
    ]
)


if len(MODEL_CONFIGS) != 21:
    raise RuntimeError(f"The HGQ2 comparison campaign must contain 21 jobs, got {len(MODEL_CONFIGS)}.")
if tuple(config.index for config in MODEL_CONFIGS) != tuple(range(len(MODEL_CONFIGS))):
    raise RuntimeError("HGQ2 campaign indices must be contiguous and zero-based.")
if len({config.alias for config in MODEL_CONFIGS}) != len(MODEL_CONFIGS):
    raise RuntimeError("HGQ2 campaign aliases must be unique.")


def config_for_task(task_id: int) -> HGQModelConfig:
    if task_id < 0 or task_id >= len(MODEL_CONFIGS):
        raise ValueError(f"Task {task_id} is outside 0..{len(MODEL_CONFIGS) - 1}.")
    return MODEL_CONFIGS[task_id]


def config_for_alias(alias: str) -> HGQModelConfig:
    for config in MODEL_CONFIGS:
        if config.alias == alias:
            return config
    raise ValueError(f"Unknown HGQ2 campaign alias: {alias}")


def manifest() -> dict:
    return {
        "schema_version": 1,
        "campaign": "hgq2_model2_pareto_qkeras_matched",
        "comparison_only": True,
        "model_count": len(MODEL_CONFIGS),
        "resource_note": (
            "QKeras LUT/FF values are reference targets only. HGQ EBOPs are a training proxy; "
            "the plot x coordinate must come from synthesis under the shared hardware flow."
        ),
        "models": [config.to_dict() for config in MODEL_CONFIGS],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--print-task-count", action="store_true")
    group.add_argument("--task", type=int)
    group.add_argument("--write-manifest", type=Path)
    args = parser.parse_args()

    if args.print_task_count:
        print(len(MODEL_CONFIGS))
    elif args.task is not None:
        print(json.dumps(config_for_task(args.task).to_dict(), indent=2, sort_keys=True))
    else:
        args.write_manifest.parent.mkdir(parents=True, exist_ok=True)
        args.write_manifest.write_text(json.dumps(manifest(), indent=2, sort_keys=True) + "\n")
        print(args.write_manifest)


if __name__ == "__main__":
    main()
