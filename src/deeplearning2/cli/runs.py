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
from deeplearning2.models.deep.runner import run_deep_experiment
from deeplearning2.models.components.contracts import (
    RunnerExecutionConfig,
    SplitDependencyContract,
    TaskContract,
)
from deeplearning2.models.components.targets import build_target_contract
from deeplearning2.models.transfer.protocols import FREEZE_MODES, TRANSFER_STAGES
from deeplearning2.models.transfer.runner import run_transfer_experiment

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

    split_name = spec.split or "scaffold_holdout"
    split_protocol = get_split_protocol(split_name)
    task_rows = _resolve_launch_tasks(
        db_path=args.db_path,
        view_name=args.view_name,
        medium_scope=spec.medium_scope,
    )
    if spec.family == "baseline":
        run_reports = _launch_baseline_family(spec, split_name, split_protocol, task_rows)
    elif spec.family == "deep":
        run_reports = _launch_deep_family(spec, split_name, split_protocol, task_rows)
    elif spec.family == "transfer":
        run_reports = _launch_transfer_family(spec, split_name, split_protocol, task_rows)
    else:
        raise ValueError(f"Unsupported launch family={spec.family!r}.")

    print(json.dumps(run_reports, ensure_ascii=False, indent=2))


def _launch_baseline_family(spec, split_name: str, split_protocol, task_rows: list) -> list[dict[str, object]]:
    models = spec.body.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError(f"Baseline experiment at {spec.config_path} must declare a non-empty models list.")

    run_reports: list[dict[str, object]] = []
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
                    extra=_build_launch_extra(
                        spec=spec,
                        task_row=task_row,
                        split_protocol=split_protocol,
                        launch_mode="baseline_task_inventory_launch",
                        declared_model_name=str(model_name),
                    ),
                )
            )
            run_reports.append(report.to_dict())
    return run_reports


def _launch_deep_family(spec, split_name: str, split_protocol, task_rows: list) -> list[dict[str, object]]:
    architecture = spec.body.get("architecture")
    if not isinstance(architecture, dict) or not architecture:
        raise ValueError(f"Deep experiment at {spec.config_path} must declare an architecture mapping.")

    run_reports: list[dict[str, object]] = []
    for task_row in task_rows:
        report = run_deep_experiment(
            RunnerExecutionConfig(
                runner_family="deep",
                run_name=f"{spec.experiment_id}__{task_row.task_id}",
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
                    notes="deep_launch_bound_to_task_inventory",
                ),
                medium_scope=spec.medium_scope or "water",
                extra=_build_launch_extra(
                    spec=spec,
                    task_row=task_row,
                    split_protocol=split_protocol,
                    launch_mode="deep_task_inventory_launch",
                    architecture=architecture,
                ),
            )
        )
        run_reports.append(report.to_dict())
    return run_reports


def _launch_transfer_family(spec, split_name: str, split_protocol, task_rows: list) -> list[dict[str, object]]:
    stages = spec.body.get("stages")
    freeze_modes = spec.body.get("freeze_modes")
    if not isinstance(stages, list) or not stages:
        raise ValueError(f"Transfer experiment at {spec.config_path} must declare a non-empty stages list.")
    if not isinstance(freeze_modes, list) or not freeze_modes:
        raise ValueError(
            f"Transfer experiment at {spec.config_path} must declare a non-empty freeze_modes list."
        )

    transfer_stage_name = _resolve_transfer_stage_name(spec.body)
    freeze_mode_name = _resolve_freeze_mode_name(freeze_modes)
    medium_scope = _resolve_transfer_medium_scope(spec.body)

    run_reports: list[dict[str, object]] = []
    for task_row in task_rows:
        report = run_transfer_experiment(
            RunnerExecutionConfig(
                runner_family="transfer",
                run_name=f"{spec.experiment_id}__{task_row.task_id}__{transfer_stage_name}",
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
                    notes="transfer_launch_bound_to_task_inventory",
                ),
                medium_scope=medium_scope,
                transfer_stage=transfer_stage_name,
                freeze_mode=freeze_mode_name,
                extra=_build_launch_extra(
                    spec=spec,
                    task_row=task_row,
                    split_protocol=split_protocol,
                    launch_mode="transfer_task_inventory_launch",
                    declared_transfer_stage=transfer_stage_name,
                    declared_freeze_mode=freeze_mode_name,
                    pretrain_scope=spec.body.get("pretrain_scope"),
                    finetune_scope=spec.body.get("finetune_scope"),
                ),
            )
        )
        run_reports.append(report.to_dict())
    return run_reports


def _build_launch_extra(
    *,
    spec,
    task_row,
    split_protocol,
    launch_mode: str,
    declared_model_name: str | None = None,
    architecture: dict | None = None,
    declared_transfer_stage: str | None = None,
    declared_freeze_mode: str | None = None,
    pretrain_scope: str | None = None,
    finetune_scope: str | None = None,
) -> dict[str, object]:
    extra = {
        "config_path": str(spec.config_path),
        "experiment_id": spec.experiment_id,
        "launch_mode": launch_mode,
        "task_id": task_row.task_id,
        "target_family": task_row.target_family,
        "task_sample_count": task_row.sample_count,
        "task_distinct_smiles_count": task_row.distinct_smiles_count,
        "task_mediums": list(task_row.mediums),
        "split_purpose": split_protocol.purpose,
        "split_is_primary": split_protocol.is_primary,
    }
    if declared_model_name is not None:
        extra["declared_model_name"] = declared_model_name
    if architecture is not None:
        extra["architecture"] = architecture
    if declared_transfer_stage is not None:
        extra["declared_transfer_stage"] = declared_transfer_stage
    if declared_freeze_mode is not None:
        extra["declared_freeze_mode"] = declared_freeze_mode
    if pretrain_scope is not None:
        extra["pretrain_scope"] = pretrain_scope
    if finetune_scope is not None:
        extra["finetune_scope"] = finetune_scope
    return extra


def _resolve_transfer_stage_name(body: dict[str, object]) -> str:
    pretrain_scope = str(body.get("pretrain_scope") or "")
    finetune_scope = str(body.get("finetune_scope") or "")
    if pretrain_scope == "water" and finetune_scope == "soil":
        return "finetune_soil"
    if pretrain_scope == "water":
        return "pretrain_water"
    if pretrain_scope == "water_sediment":
        return "pretrain_water_sediment"
    fallback = "finetune_soil"
    if fallback not in TRANSFER_STAGES:
        raise ValueError("Expected finetune_soil in TRANSFER_STAGES.")
    return fallback


def _resolve_freeze_mode_name(freeze_modes: list[object]) -> str:
    normalized = [str(item) for item in freeze_modes]
    for preferred in ("chemical_encoder_partial", "none", "chemical_encoder_full"):
        if preferred in normalized:
            return preferred
    raise ValueError(f"Could not map freeze_modes {normalized!r} onto supported modes {FREEZE_MODES}.")


def _resolve_transfer_medium_scope(body: dict[str, object]) -> str:
    pretrain_scope = str(body.get("pretrain_scope") or "")
    finetune_scope = str(body.get("finetune_scope") or "")
    if pretrain_scope == "water" and finetune_scope == "soil":
        return "water_sediment_soil"
    if pretrain_scope == "water":
        return "water"
    if pretrain_scope == "water_sediment":
        return "water_sediment"
    return "water_sediment_soil"


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
