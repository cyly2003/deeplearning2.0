"""Experiment listing, summary, and launch CLI helpers."""

from __future__ import annotations

import argparse
import json

from deeplearning2.config.loader import (
    load_experiment_launch_spec,
    load_experiment_records,
    summarize_experiment_families,
)
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
    run_reports = []
    for model_name in models:
        normalized_model_name = BASELINE_MODEL_ALIASES.get(str(model_name), str(model_name))
        report = run_baseline_experiment(
            RunnerExecutionConfig(
                runner_family="baseline",
                run_name=f"{spec.experiment_id}__{normalized_model_name}",
                task=TaskContract(
                    species_id="placeholder_species",
                    effect_type="EC",
                    endpoint_observation="mortality",
                ),
                split=SplitDependencyContract(
                    split_name=split_name,
                    split_group="primary_research_split",
                ),
                target=build_target_contract("EC", notes="baseline_launch_skeleton"),
                medium_scope=spec.medium_scope or "water",
                model_name=normalized_model_name,
                extra={
                    "config_path": str(spec.config_path),
                    "experiment_id": spec.experiment_id,
                    "launch_mode": "placeholder_baseline_launch",
                    "declared_model_name": str(model_name),
                },
            )
        )
        run_reports.append(report.to_dict())

    print(json.dumps(run_reports, ensure_ascii=False, indent=2))


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
    launch_parser.set_defaults(handler=handle_runs_launch)
