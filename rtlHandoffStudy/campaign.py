#!/usr/bin/env python3
"""Inspect and validate the 20-model EFLX RTL handoff campaign."""

from __future__ import annotations

import argparse
import math

from models.registry import REGISTRY
from models.rtl_handoff import RTL_HANDOFF_CONFIGS, RTL_HANDOFF_BY_ALIAS
from rtlHandoffStudy.portable import estimated_group_sum_luts


def validate_campaign() -> None:
    aliases = [str(config["alias"]) for config in RTL_HANDOFF_CONFIGS]
    if len(aliases) != 20 or len(set(aliases)) != 20:
        raise RuntimeError("RTL handoff campaign must contain exactly 20 unique aliases.")
    families = [str(config["family"]) for config in RTL_HANDOFF_CONFIGS]
    if families.count("dense") != 10 or families.count("conv") != 10:
        raise RuntimeError("RTL handoff campaign must contain ten dense and ten conv models.")
    if set(aliases) != set(RTL_HANDOFF_BY_ALIAS):
        raise RuntimeError("Campaign list and registry lookup disagree.")
    for config in RTL_HANDOFF_BY_ALIAS.values():
        registry_name = str(config["registry_name"])
        if registry_name not in REGISTRY:
            raise RuntimeError(f"Missing registered model {registry_name}.")
        model_class = REGISTRY[registry_name]["class"]
        deployed = model_class.deployment_config()
        if deployed["final_logic_width"] != 3 * deployed["group_width_per_class"]:
            raise RuntimeError(f"{config['alias']} has an invalid GroupSum width.")
        expected_code = math.ceil(math.log2(deployed["thresholds_per_feature"] + 1))
        if deployed["compact_code_bits_per_feature"] != expected_code:
            raise RuntimeError(f"{config['alias']} has an invalid compact-code width.")
        if deployed["tau"] <= 0:
            raise RuntimeError(f"{config['alias']} has non-positive tau.")


def print_table() -> None:
    header = (
        "task\talias\tfamily\tbase\tthresholds\tbinarizer\tgroup_width\ttau\t"
        "therm_bits\tcompact_bits\tinput_margin\tgroup_sum_luts_diag"
    )
    print(header)
    for index, config in enumerate(RTL_HANDOFF_CONFIGS):
        entry = RTL_HANDOFF_BY_ALIAS[str(config["alias"])]
        model = REGISTRY[str(entry["registry_name"])]["class"]
        deployed = model.deployment_config()
        print(
            f"{index}\t{deployed['alias']}\t{deployed['family']}\t"
            f"{deployed['base_architecture']}\t{deployed['thresholds_per_feature']}\t"
            f"{deployed['binarization_training']}\t{deployed['group_width_per_class']}\t"
            f"{deployed['tau']:g}\t{deployed['thermometer_input_bits']}\t"
            f"{deployed['compact_input_bits']}\t{632-deployed['compact_input_bits']}\t"
            f"{estimated_group_sum_luts(deployed['group_width_per_class'])}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--print-table", action="store_true")
    args = parser.parse_args()
    validate_campaign()
    if args.print_table or not args.validate:
        print_table()
    print("[OK] RTL handoff campaign contains 10 dense + 10 conv configurations.")


if __name__ == "__main__":
    main()
