"""Shared execution contracts for baseline, deep, and transfer runners."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from deeplearning2.models.baseline.registry import BASELINE_MODELS
from deeplearning2.models.components.tasks import TASK_ID_DEFINITION
from deeplearning2.models.transfer.protocols import FREEZE_MODES, TRANSFER_STAGES


RUNNER_FAMILIES = ("baseline", "deep", "transfer")
SUPPORTED_SPLIT_NAMES = (
    "scaffold_holdout",
    "chemical_id_holdout",
    "medium_transfer_split",
)
SUPPORTED_MEDIUM_SCOPES = (
    "water",
    "water_sediment",
    "water_sediment_soil",
)
REPORT_REQUIRED_SECTIONS = (
    "task",
    "split",
    "target",
    "artifacts",
    "summary",
)


@dataclass(frozen=True)
class TaskContract:
    """Task identity shared across all model families."""

    species_id: str
    effect_type: str
    endpoint_observation: str
    task_id_definition: str = TASK_ID_DEFINITION
    effect_level_as_input: bool = True
    effect_level_in_task_id: bool = False
    excludes_nr: bool = True


@dataclass(frozen=True)
class SplitDependencyContract:
    """Split protocol dependency required for comparable research lines."""

    split_name: str
    split_group: str
    requires_curated_view: str = "ecotox_toxicity_joined_curated"
    supports_scaffold_holdout: bool = True
    supports_chemical_id_holdout: bool = True
    supports_medium_transfer_split: bool = True

    def __post_init__(self) -> None:
        if self.split_name not in SUPPORTED_SPLIT_NAMES:
            raise ValueError(
                f"Unsupported split_name={self.split_name!r}. "
                f"Expected one of {SUPPORTED_SPLIT_NAMES}."
            )


@dataclass(frozen=True)
class TargetContract:
    """Target-space placeholder without implementing transformations."""

    family: str
    training_space: str
    output_space: str
    uses_effect_level_features: bool
    notes: str = ""


@dataclass(frozen=True)
class ExecutionArtifacts:
    """Reserved artifact outputs for every runner family."""

    run_dir: str
    config_path: str
    manifest_path: str
    report_path: str
    checkpoint_path: str | None = None
    predictions_path: str | None = None


@dataclass(frozen=True)
class RunnerExecutionConfig:
    """Unified execution input contract."""

    runner_family: str
    run_name: str
    task: TaskContract
    split: SplitDependencyContract
    target: TargetContract
    medium_scope: str
    dataset_view: str = "ecotox_toxicity_joined_curated"
    seed: int = 42
    context_features: tuple[str, ...] = (
        "species_id",
        "taxon_group_l1",
        "taxon_group_l2",
        "taxon_group_l3",
        "genus",
        "family",
        "organism_lifestage",
        "primary_medium",
        "effect_type",
        "endpoint_observation",
        "is_lethal",
        "is_chronic",
        "is_threshold_endpoint",
        "is_bioaccumulation",
        "duration_h",
    )
    effect_level_features: tuple[str, ...] = (
        "level_fraction",
        "logit_level_fraction",
        "probit_level_fraction",
    )
    descriptor_mode: str = "rdkit_grouped_plus_morgan"
    model_name: str | None = None
    transfer_stage: str | None = None
    freeze_mode: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.runner_family not in RUNNER_FAMILIES:
            raise ValueError(
                f"Unsupported runner_family={self.runner_family!r}. "
                f"Expected one of {RUNNER_FAMILIES}."
            )
        if self.medium_scope not in SUPPORTED_MEDIUM_SCOPES:
            raise ValueError(
                f"Unsupported medium_scope={self.medium_scope!r}. "
                f"Expected one of {SUPPORTED_MEDIUM_SCOPES}."
            )
        if self.runner_family == "baseline":
            if self.model_name not in BASELINE_MODELS:
                raise ValueError(
                    "Baseline runner requires model_name from BASELINE_MODELS: "
                    f"{BASELINE_MODELS}."
                )
        else:
            if self.model_name is not None:
                raise ValueError("Only baseline runner may set model_name.")
        if self.runner_family == "transfer":
            if self.transfer_stage not in TRANSFER_STAGES:
                raise ValueError(
                    "Transfer runner requires transfer_stage from TRANSFER_STAGES: "
                    f"{TRANSFER_STAGES}."
                )
            if self.freeze_mode not in FREEZE_MODES:
                raise ValueError(
                    "Transfer runner requires freeze_mode from FREEZE_MODES: "
                    f"{FREEZE_MODES}."
                )
        else:
            if self.transfer_stage is not None or self.freeze_mode is not None:
                raise ValueError("Only transfer runner may set transfer_stage or freeze_mode.")


@dataclass(frozen=True)
class ExecutionSummary:
    """Status-only training placeholder summary."""

    status: str
    executed_training: bool
    comparability_group: str
    notes: tuple[str, ...]


@dataclass(frozen=True)
class ExecutionReportContract:
    """Unified report schema shared by all runners."""

    runner_family: str
    config: RunnerExecutionConfig
    artifacts: ExecutionArtifacts
    summary: ExecutionSummary
    schema_version: str = "0.1.0"
    report_sections: tuple[str, ...] = REPORT_REQUIRED_SECTIONS

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_placeholder_artifacts(runner_family: str, run_name: str) -> ExecutionArtifacts:
    """Create deterministic placeholder artifact paths."""

    root = f"artifacts/{runner_family}/{run_name}"
    return ExecutionArtifacts(
        run_dir=root,
        config_path=f"{root}/config.json",
        manifest_path=f"{root}/manifest.json",
        report_path=f"{root}/report.json",
        checkpoint_path=f"{root}/checkpoint.bin",
        predictions_path=f"{root}/predictions.csv",
    )


def build_execution_summary(runner_family: str) -> ExecutionSummary:
    """Create a non-training placeholder summary."""

    notes = [
        "placeholder_only",
        "no_real_training_executed",
        "shared_task_and_split_contract_enforced",
    ]
    if runner_family == "baseline":
        notes.append("baseline_comparable_to_unified_task_system")
    elif runner_family == "deep":
        notes.append("deep_reserved_for_unified_multitask_main_model")
    else:
        notes.append("transfer_reserved_for_pretrain_finetune_research_line")

    return ExecutionSummary(
        status="not_started",
        executed_training=False,
        comparability_group="unified_multitask_qsar",
        notes=tuple(notes),
    )


def build_execution_report(config: RunnerExecutionConfig) -> ExecutionReportContract:
    """Build a unified placeholder report for any runner family."""

    return ExecutionReportContract(
        runner_family=config.runner_family,
        config=config,
        artifacts=build_placeholder_artifacts(config.runner_family, config.run_name),
        summary=build_execution_summary(config.runner_family),
    )
