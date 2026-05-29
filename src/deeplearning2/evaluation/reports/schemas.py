"""Execution report schemas shared by model runners and result aggregators."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from deeplearning2.config.loader import load_yaml_document
from deeplearning2.config.registry import CONFIG_FILES
from deeplearning2.models.components.contracts import (
    ExecutionArtifacts,
    ExecutionReportContract,
    ExecutionSummary,
    REPORT_REQUIRED_SECTIONS,
    RunnerExecutionConfig,
    SplitDependencyContract,
    TargetContract,
    TaskContract,
)

REPORT_SCHEMA_NAME = "model_execution_report"
REPORT_SCHEMA_VERSION = "0.1.0"
RESULT_BUNDLE_SCHEMA_NAME = "unified_result_bundle"
RESULT_BUNDLE_SCHEMA_VERSION = "0.1.0"


def load_metric_names() -> tuple[str, ...]:
    """Load the authoritative evaluation metrics used by result rows."""

    payload = load_yaml_document(CONFIG_FILES["evaluation"])
    body = payload.get("evaluation")
    if not isinstance(body, dict):
        raise ValueError("Expected top-level 'evaluation' mapping in configs/evaluation/evaluation.yaml.")
    return tuple(body.get("metrics", ()))


SUPPORTED_METRIC_NAMES = load_metric_names()


@dataclass(frozen=True)
class RunMetadata:
    """Minimal metadata shared by task/split/species result outputs."""

    run_name: str
    runner_family: str
    medium_scope: str
    split_name: str
    dataset_view: str
    seed: int
    comparability_group: str = "unified_multitask_qsar"
    task_id_definition: str = "task = species + endpoint semantics"


@dataclass(frozen=True)
class TaskMetricRow:
    """Task-level performance record."""

    task_id: str
    species_id: str
    effect_type: str
    endpoint_observation: str
    split_name: str
    sample_count: int
    metrics: dict[str, float]

    def __post_init__(self) -> None:
        _validate_metrics(self.metrics)


@dataclass(frozen=True)
class SplitMetricRow:
    """Split-level aggregate performance record."""

    split_name: str
    runner_family: str
    medium_scope: str
    task_count: int
    metrics: dict[str, float]

    def __post_init__(self) -> None:
        _validate_metrics(self.metrics)


@dataclass(frozen=True)
class SpeciesMetricRow:
    """Species-level aggregate performance record."""

    species_id: str
    split_name: str
    task_count: int
    metrics: dict[str, float]

    def __post_init__(self) -> None:
        _validate_metrics(self.metrics)


@dataclass(frozen=True)
class UnifiedResultBundle:
    """Result bundle that downstream AD/SSD/reporting workflows can share."""

    metadata: RunMetadata
    task_rows: tuple[TaskMetricRow, ...] = field(default_factory=tuple)
    split_rows: tuple[SplitMetricRow, ...] = field(default_factory=tuple)
    species_rows: tuple[SpeciesMetricRow, ...] = field(default_factory=tuple)
    schema_name: str = RESULT_BUNDLE_SCHEMA_NAME
    schema_version: str = RESULT_BUNDLE_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Serialize the bundle for JSON output."""

        return asdict(self)


def _validate_metrics(metrics: dict[str, float]) -> None:
    unknown = set(metrics) - set(SUPPORTED_METRIC_NAMES)
    if unknown:
        raise ValueError(
            f"Unsupported metrics {sorted(unknown)}. Expected subset of {SUPPORTED_METRIC_NAMES}."
        )


def build_result_bundle(
    metadata: RunMetadata,
    *,
    task_rows: tuple[TaskMetricRow, ...] = (),
    split_rows: tuple[SplitMetricRow, ...] = (),
    species_rows: tuple[SpeciesMetricRow, ...] = (),
) -> UnifiedResultBundle:
    """Build a standardized result bundle for downstream workflows."""

    return UnifiedResultBundle(
        metadata=metadata,
        task_rows=task_rows,
        split_rows=split_rows,
        species_rows=species_rows,
    )

__all__ = [
    "ExecutionArtifacts",
    "ExecutionReportContract",
    "ExecutionSummary",
    "REPORT_REQUIRED_SECTIONS",
    "REPORT_SCHEMA_NAME",
    "REPORT_SCHEMA_VERSION",
    "RESULT_BUNDLE_SCHEMA_NAME",
    "RESULT_BUNDLE_SCHEMA_VERSION",
    "RunnerExecutionConfig",
    "RunMetadata",
    "SplitDependencyContract",
    "SplitMetricRow",
    "SpeciesMetricRow",
    "SUPPORTED_METRIC_NAMES",
    "TaskMetricRow",
    "TargetContract",
    "TaskContract",
    "UnifiedResultBundle",
    "build_result_bundle",
]
