"""Placeholder runner for transfer-learning experiments."""

from __future__ import annotations

from deeplearning2.models.components.contracts import (
    ExecutionReportContract,
    RunnerExecutionConfig,
    build_execution_report,
    materialize_execution_report,
)


def run_transfer_experiment(config: RunnerExecutionConfig) -> ExecutionReportContract:
    """Return the shared execution contract for the transfer research line."""

    if config.runner_family != "transfer":
        raise ValueError("run_transfer_experiment requires runner_family='transfer'.")
    return materialize_execution_report(build_execution_report(config))
