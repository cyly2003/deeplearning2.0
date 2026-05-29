"""Experiment listing, summary, and launch CLI helpers."""

from __future__ import annotations

import argparse
import json

from deeplearning2.config.loader import (
    load_experiment_launch_spec,
    load_experiment_records,
    summarize_experiment_families,
)
from deeplearning2.data.tasks import build_task_inventory, fetch_normalized_task_records
from deeplearning2.data.splits import get_split_protocol
from deeplearning2.models.baseline.runner import run_baseline_experiment
from deeplearning2.models.components.contracts import (
    RunnerExecutionConfig,
    SplitDependencyContract,
    TaskContract,
)
from deeplearning2.models.components.targets import build_target_contract

BASELINE_MODEL_ALIASES = {
    "randomforest": "random_forest",
}
DEFAULT_PLACEHOLDER_TASK = {
    "species_id": "placeholder_species",
    "effect_type": "EC",
    "endpoint_observation": "mortality",
}


def _build_list_payload() -> list[dict[str, str | None]]:
    return [record.to_dict() for record in load_experiment_records()]


def _build_summary_payload() -> dict[str, dict[str, int]]:
    return summarize_experiment_families(load_experiment_records())


def handle_runs_list(args: argparse.Namespace) -> None:
    """Print experiment configs as JSON or text rows."""

    records = load_experiment_records()
    if args.format == "json":
        print(json.dumps([record.to_dict() for record in records], ensure_ascii=False, indent=2))
        return

    for record in records:
        parts = [
            f"id={record.experiment_id}",
            f"family={record.family}",
            f"config={record.config_path.as_posix()}",
        ]
        if record.medium_scope:
            parts.append(f"medium_scope={record.medium_scope}")
        if record.split:
            parts.append(f"split={record.split}")
        if record.summary:
            parts.append(f"summary={record.summary}")
        print(" | ".join(parts))


def handle_runs_summary(args: argparse.Namespace) -> None:
    """Print family-level experiment counts."""

    payload = _build_summary_payload()
    if args.format == "json":
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    for family, stats in payload.items():
        print(
            " | ".join(
                [
                    f"family={family}",
                    f"count={stats['count']}",
                    f"with_split={stats['with_split']}",
                    f"with_medium_scope={stats['with_medium_scope']}",
                ]
            )
        )


def handle_runs_launch(args: argparse.Namespace) -> None:
    """Launch a placeholder experiment runner using the shared contracts."""

    spec = load_experiment_launch_spec(args.config)
    if spec.family != args.family:
        raise ValueError(
            f"Launch family mismatch: CLI family={args.family!r}, config family={spec.family!r}."
        )

    if spec.family != "baseline":
        raise ValueError("Current launch skeleton only supports family='baseline'.")

    models = spec.body.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError(f"Baseline experiment at {args.config} must declare a non-empty models list.")

    split_name = spec.split or "scaffold_holdout"
    split_protocol = get_split_protocol(split_name)
    task_rows = _resolve_launch_tasks(
        db_path=args.db_path,
        view_name=args.view_name,
        medium_scope=spec.medium_scope,
    )
    run_reports = []
    for task_row in task_rows:
        for model_name in models:
            normalized_model_name = BASELINE_MODEL_ALIASES.get(str(model_name), str(model_name))
            report = run_baseline_experiment(
                RunnerExecutionConfig(
                    runner_family="baseline",
                    run_name=f"{spec.experiment_id}__{task_row.task_id}__{normalized_model_name}",
                    task=TaskContract(
                        species_id=task_row.species_id,
                        effect_type=task_row.effect_type,
                        endpoint_observation=task_row.endpoint_observation,
                    ),
                    split=SplitDependencyContract(
                        split_name=split_name,
                        split_group=split_protocol.split_group,
                    ),
                    target=build_target_contract(
                        task_row.effect_type,
                        notes="baseline_launch_bound_to_task_inventory",
                    ),
                    medium_scope=spec.medium_scope or "water",
                    model_name=normalized_model_name,
                    extra={
                        "config_path": str(spec.config_path),
                        "experiment_id": spec.experiment_id,
                        "launch_mode": "baseline_task_inventory_launch",
                        "declared_model_name": str(model_name),
                        "task_id": task_row.task_id,
                        "target_family": task_row.target_family,
                        "task_sample_count": task_row.sample_count,
                        "task_distinct_smiles_count": task_row.distinct_smiles_count,
                        "task_mediums": list(task_row.mediums),
                        "split_purpose": split_protocol.purpose,
                        "split_is_primary": split_protocol.is_primary,
                    },
                )
            )
            run_reports.append(report.to_dict())

    print(json.dumps(run_reports, ensure_ascii=False, indent=2))


def _resolve_launch_tasks(
    *,
    db_path: str | None,
    view_name: str | None,
    medium_scope: str | None,
) -> list:
    try:
        records = fetch_normalized_task_records(db_path, view_name=view_name)
    except Exception:  # noqa: BLE001
        return [
            type(
                "PlaceholderTaskRow",
                (),
                {
                    "task_id": "placeholder_species__EC_mortality",
                    "species_id": DEFAULT_PLACEHOLDER_TASK["species_id"],
                    "effect_type": DEFAULT_PLACEHOLDER_TASK["effect_type"],
                    "endpoint_observation": DEFAULT_PLACEHOLDER_TASK["endpoint_observation"],
                    "target_family": "EC_LC_ICx",
                    "sample_count": 0,
                    "distinct_smiles_count": 0,
                    "mediums": ((medium_scope or "water"),),
                },
            )()
        ]

    inventory = build_task_inventory(records)
    if medium_scope:
        inventory = [row for row in inventory if medium_scope in row.mediums]
    return inventory or [
        type(
            "PlaceholderTaskRow",
            (),
            {
                "task_id": "placeholder_species__EC_mortality",
                "species_id": DEFAULT_PLACEHOLDER_TASK["species_id"],
                "effect_type": DEFAULT_PLACEHOLDER_TASK["effect_type"],
                "endpoint_observation": DEFAULT_PLACEHOLDER_TASK["endpoint_observation"],
                "target_family": "EC_LC_ICx",
                "sample_count": 0,
                "distinct_smiles_count": 0,
                "mediums": ((medium_scope or "water"),),
            },
        )()
    ]


def register_runs_subcommand(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register the runs CLI namespace."""

    parser = subparsers.add_parser(
        "runs",
        help="Inspect experiment configs for baseline/deep/transfer/ablation skeletons.",
    )
    runs_subparsers = parser.add_subparsers(dest="runs_command", required=True)

    list_parser = runs_subparsers.add_parser("list", help="List discovered experiment configs.")
    list_parser.add_argument("--format", choices=("text", "json"), default="text")
    list_parser.set_defaults(handler=handle_runs_list)

    summary_parser = runs_subparsers.add_parser(
        "summary",
        help="Summarize discovered experiment configs by family.",
    )
    summary_parser.add_argument("--format", choices=("text", "json"), default="text")
    summary_parser.set_defaults(handler=handle_runs_summary)

    launch_parser = runs_subparsers.add_parser(
        "launch",
        help="Launch a placeholder experiment runner from a config file.",
    )
    launch_parser.add_argument("--family", required=True, choices=("baseline", "deep", "transfer"))
    launch_parser.add_argument("--config", required=True, type=str)
    launch_parser.add_argument("--db-path", default=None)
    launch_parser.add_argument("--view-name", default=None)
    launch_parser.set_defaults(handler=handle_runs_launch)
