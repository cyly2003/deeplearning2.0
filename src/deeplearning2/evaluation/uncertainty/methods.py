"""Uncertainty estimation method contracts required in v1."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


UNCERTAINTY_METHODS = (
    "deep_ensemble",
    "mc_dropout",
)

RELIABILITY_OUTPUTS = (
    "predictive_mean",
    "predictive_std",
    "interval_lower",
    "interval_upper",
)


@dataclass(frozen=True)
class UncertaintyMethodSpec:
    """Authoritative uncertainty method descriptor."""

    name: str
    requires_repeated_predictions: bool
    reliability_outputs: tuple[str, ...] = RELIABILITY_OUTPUTS

    def __post_init__(self) -> None:
        if self.name not in UNCERTAINTY_METHODS:
            raise ValueError(
                f"Unsupported uncertainty method '{self.name}'. Expected one of {UNCERTAINTY_METHODS}."
            )
        invalid_outputs = tuple(
            output_name for output_name in self.reliability_outputs if output_name not in RELIABILITY_OUTPUTS
        )
        if invalid_outputs:
            raise ValueError(
                f"Unsupported reliability outputs {invalid_outputs}. Expected subset of {RELIABILITY_OUTPUTS}."
            )


UNCERTAINTY_METHOD_SPECS = (
    UncertaintyMethodSpec(name="deep_ensemble", requires_repeated_predictions=True),
    UncertaintyMethodSpec(name="mc_dropout", requires_repeated_predictions=True),
)

UNCERTAINTY_METHOD_SPEC_BY_NAME = {spec.name: spec for spec in UNCERTAINTY_METHOD_SPECS}


@dataclass(frozen=True)
class ReliabilityEstimate:
    """Comparable reliability payload shared by baseline and deep predictors."""

    predictive_mean: float
    predictive_std: float
    interval_lower: float
    interval_upper: float
    coverage_level: float = 0.95
    sample_count: int | None = None

    def __post_init__(self) -> None:
        if self.predictive_std < 0:
            raise ValueError("predictive_std must be non-negative.")
        if not 0 < self.coverage_level < 1:
            raise ValueError("coverage_level must be between 0 and 1.")
        if self.interval_lower > self.interval_upper:
            raise ValueError("interval_lower must be <= interval_upper.")
        if self.sample_count is not None and self.sample_count <= 0:
            raise ValueError("sample_count must be positive when provided.")


@dataclass(frozen=True)
class UncertaintyEstimate:
    """Per-item uncertainty estimate shared by evaluation and prediction flows."""

    sample_id: str
    task_id: str
    method: str
    reliability: ReliabilityEstimate
    member_predictions: tuple[float, ...] = ()
    epistemic_std: float | None = None
    aleatoric_std: float | None = None

    def __post_init__(self) -> None:
        if not self.sample_id:
            raise ValueError("sample_id must not be empty.")
        if not self.task_id:
            raise ValueError("task_id must not be empty.")
        if self.method not in UNCERTAINTY_METHODS:
            raise ValueError(
                f"Unsupported uncertainty method '{self.method}'. Expected one of {UNCERTAINTY_METHODS}."
            )
        spec = get_uncertainty_method_spec(self.method)
        if spec.requires_repeated_predictions and len(self.member_predictions) < 2:
            raise ValueError(
                f"{self.method} requires at least two member_predictions to support repeated inference outputs."
            )
        if self.epistemic_std is not None and self.epistemic_std < 0:
            raise ValueError("epistemic_std must be non-negative when provided.")
        if self.aleatoric_std is not None and self.aleatoric_std < 0:
            raise ValueError("aleatoric_std must be non-negative when provided.")

    def to_dict(self) -> dict[str, Any]:
        """Serialize the estimate for JSON output."""

        return asdict(self)


def get_uncertainty_method_spec(name: str) -> UncertaintyMethodSpec:
    """Return the authoritative uncertainty method spec."""

    try:
        return UNCERTAINTY_METHOD_SPEC_BY_NAME[name]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported uncertainty method '{name}'. Expected one of {UNCERTAINTY_METHODS}."
        ) from exc


def build_uncertainty_estimate(
    *,
    sample_id: str,
    task_id: str,
    method: str,
    reliability: ReliabilityEstimate,
    member_predictions: tuple[float, ...],
    epistemic_std: float | None = None,
    aleatoric_std: float | None = None,
) -> UncertaintyEstimate:
    """Build a validated uncertainty estimate row."""

    return UncertaintyEstimate(
        sample_id=sample_id,
        task_id=task_id,
        method=method,
        reliability=reliability,
        member_predictions=member_predictions,
        epistemic_std=epistemic_std,
        aleatoric_std=aleatoric_std,
    )


__all__ = [
    "RELIABILITY_OUTPUTS",
    "ReliabilityEstimate",
    "UNCERTAINTY_METHOD_SPEC_BY_NAME",
    "UNCERTAINTY_METHOD_SPECS",
    "UNCERTAINTY_METHODS",
    "UncertaintyEstimate",
    "UncertaintyMethodSpec",
    "build_uncertainty_estimate",
    "get_uncertainty_method_spec",
]
