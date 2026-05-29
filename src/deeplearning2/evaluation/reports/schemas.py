"""Execution report schemas shared by model runners."""

from __future__ import annotations

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

__all__ = [
    "ExecutionArtifacts",
    "ExecutionReportContract",
    "ExecutionSummary",
    "REPORT_REQUIRED_SECTIONS",
    "REPORT_SCHEMA_NAME",
    "REPORT_SCHEMA_VERSION",
    "RunnerExecutionConfig",
    "SplitDependencyContract",
    "TargetContract",
    "TaskContract",
]

