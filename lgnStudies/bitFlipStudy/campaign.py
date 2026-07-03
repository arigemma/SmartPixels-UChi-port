"""Canonical model and fault-count layout for the chunked Slurm campaign."""

from __future__ import annotations

import argparse


MODELS = (
    (
        "s06_p760M_tau40",
        "lgn-dense-2-dense-100_128_-model2lgnFull_s06_p760M_40",
    ),
    (
        "s08_p1202M_tau40",
        "lgn-dense-2-dense-100_128_-model2lgnFull_s08_p1202M_40",
    ),
    (
        "s09_p1460M_tau40",
        "lgn-dense-2-dense-100_128_-model2lgnFull_s09_p1460M_40",
    ),
)

FLIP_COUNT_CHUNKS = (
    (1, 2, 4),
    (8, 16, 24),
    (32, 48, 64),
    (96, 128, 192),
    (256, 384, 512),
    (768, 1024, 1536),
    (2048, 3072, 4096),
)

EXPECTED_FLIP_COUNTS = (
    0,
    *(count for chunk in FLIP_COUNT_CHUNKS for count in chunk),
)
REPEATS = 5
NUMBER_TASKS = len(MODELS) * len(FLIP_COUNT_CHUNKS)


def task_configuration(task_id: int) -> tuple[str, str, int, tuple[int, ...]]:
    if task_id < 0 or task_id >= NUMBER_TASKS:
        raise ValueError(f"Task ID must lie in [0, {NUMBER_TASKS - 1}].")
    model_index, chunk_index = divmod(task_id, len(FLIP_COUNT_CHUNKS))
    model_alias, registry_name = MODELS[model_index]
    return (
        model_alias,
        registry_name,
        chunk_index,
        FLIP_COUNT_CHUNKS[chunk_index],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", type=int, required=True)
    args = parser.parse_args()
    try:
        model_alias, registry_name, chunk_index, counts = task_configuration(
            args.task_id
        )
    except ValueError as error:
        parser.error(str(error))
    print(
        "\t".join(
            (
                model_alias,
                registry_name,
                f"{chunk_index:02d}",
                " ".join(map(str, counts)),
            )
        )
    )


if __name__ == "__main__":
    main()
