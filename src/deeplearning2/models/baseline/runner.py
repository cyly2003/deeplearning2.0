"""Placeholder runner for baseline model experiments."""

from __future__ import annotations

from deeplearning2.models.components.contracts import (
    ExecutionReportContract,
    RunnerExecutionConfig,
    build_execution_report,
    materialize_execution_report,
)


def run_baseline_experiment(config: RunnerExecutionConfig) -> ExecutionReportContract:
    """Return the shared execution contract for the baseline research line."""

    if config.runner_family != "baseline":
        raise ValueError("run_baseline_experiment requires runner_family='baseline'.")
    return materialize_execution_report(build_execution_report(config))
