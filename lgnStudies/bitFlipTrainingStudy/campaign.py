"""Model and task layout for the fault-aware training campaign."""

from __future__ import annotations

import argparse

from lgnStudies.bitFlipStudy.campaign import FLIP_COUNT_CHUNKS, MODELS


NUMBER_TRAINING_TASKS = len(MODELS)
NUMBER_EVALUATION_TASKS = len(MODELS) * len(FLIP_COUNT_CHUNKS)


def training_configuration(task_id: int) -> tuple[str, str]:
    if task_id < 0 or task_id >= NUMBER_TRAINING_TASKS:
        raise ValueError(
            f"Training task ID must lie in [0, {NUMBER_TRAINING_TASKS - 1}]."
        )
    return MODELS[task_id]


def evaluation_configuration(
    task_id: int,
) -> tuple[str, str, int, tuple[int, ...]]:
    if task_id < 0 or task_id >= NUMBER_EVALUATION_TASKS:
        raise ValueError(
            f"Evaluation task ID must lie in [0, {NUMBER_EVALUATION_TASKS - 1}]."
        )
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
    task_group = parser.add_mutually_exclusive_group(required=True)
    task_group.add_argument("--training-task-id", type=int)
    task_group.add_argument("--evaluation-task-id", type=int)
    args = parser.parse_args()

    try:
        if args.training_task_id is not None:
            print("\t".join(training_configuration(args.training_task_id)))
        else:
            model, registry, chunk, counts = evaluation_configuration(
                args.evaluation_task_id
            )
            print(
                "\t".join(
                    (
                        model,
                        registry,
                        f"{chunk:02d}",
                        " ".join(map(str, counts)),
                    )
                )
            )
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
