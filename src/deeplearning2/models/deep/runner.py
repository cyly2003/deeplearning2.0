"""Placeholder runner for the unified deep multitask model."""

from __future__ import annotations

from deeplearning2.models.components.contracts import (
    ExecutionReportContract,
    RunnerExecutionConfig,
    build_execution_report,
)


def run_deep_experiment(config: RunnerExecutionConfig) -> ExecutionReportContract:
    """Return the shared execution contract for the deep research line."""

    if config.runner_family != "deep":
        raise ValueError("run_deep_experiment requires runner_family='deep'.")
    return build_execution_report(config)

