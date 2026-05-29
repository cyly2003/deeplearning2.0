"""Structured SSD workflow contracts without implementing real fitting."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from deeplearning2.predict.schemas import (
    BIOACCUMULATION_EFFECT_TYPES,
    BatchPredictionRequest,
    PredictionResult,
    effect_level_is_forbidden,
    effect_level_is_required,
    resolve_endpoint_family,
)


SSD_REQUIRED_OUTPUTS = (
    "hc5",
    "hc10",
    "uncertainty",
)
SSD_SCHEMA_NAME = "ssd_contract"
SSD_SCHEMA_VERSION = "0.1.0"
SSD_ALLOWED_ENDPOINT_FAMILIES = ("EC_LC_ICx", "NOEC_LOEC", "BCF_BAF")


@dataclass(frozen=True)
class EndpointFamilyFilter:
    """Explicit endpoint-family gate to prevent mixing incompatible SSD families."""

    endpoint_family: str
    allowed_effect_types: tuple[str, ...]
    endpoint_observation: str

    def __post_init__(self) -> None:
        if self.endpoint_family not in SSD_ALLOWED_ENDPOINT_FAMILIES:
            raise ValueError(
                "endpoint_family must be one of "
                f"{SSD_ALLOWED_ENDPOINT_FAMILIES}, got {self.endpoint_family!r}."
            )
        resolved_families = {resolve_endpoint_family(effect_type) for effect_type in self.allowed_effect_types}
        if resolved_families != {self.endpoint_family}:
            raise ValueError(
                "allowed_effect_types must stay inside a single endpoint family. "
                f"Expected only {self.endpoint_family!r}, got {sorted(resolved_families)!r}."
            )


@dataclass(frozen=True)
class HistoricalTaskPerformance:
    """Per-task historical quality record used for SSD species screening."""

    task_id: str
    species_id: str
    effect_type: str
    endpoint_observation: str
    r2: float
    sample_count: int

    def __post_init__(self) -> None:
        if self.sample_count <= 0:
            raise ValueError("sample_count must be positive.")
        if self.endpoint_family not in SSD_ALLOWED_ENDPOINT_FAMILIES:
            raise ValueError(
                f"Unsupported endpoint family {self.endpoint_family!r} for SSD performance screening."
            )

    @property
    def endpoint_family(self) -> str:
        """Resolve the family used for SSD eligibility checks."""

        return resolve_endpoint_family(self.effect_type)


@dataclass(frozen=True)
class HistoricalTaskFilterDecision:
    """Decision row describing whether a species/task passes the R2 threshold."""

    performance: HistoricalTaskPerformance
    r2_threshold: float
    passes_threshold: bool
    rejection_reason: str | None = None


@dataclass(frozen=True)
class SSDSpeciesScreeningSummary:
    """Structured record of task-level R2 screening before SSD fitting."""

    endpoint_family: str
    endpoint_observation: str
    r2_threshold: float
    decisions: tuple[HistoricalTaskFilterDecision, ...]

    def __post_init__(self) -> None:
        if self.endpoint_family not in SSD_ALLOWED_ENDPOINT_FAMILIES:
            raise ValueError(
                f"endpoint_family must be one of {SSD_ALLOWED_ENDPOINT_FAMILIES}, got {self.endpoint_family!r}."
            )
        if not 0 <= self.r2_threshold <= 1:
            raise ValueError("r2_threshold must be between 0 and 1.")

    @property
    def eligible_species_ids(self) -> tuple[str, ...]:
        """Return unique species IDs whose historical task performance passes the threshold."""

        seen: dict[str, None] = {}
        for decision in self.decisions:
            if decision.passes_threshold:
                seen.setdefault(decision.performance.species_id, None)
        return tuple(seen)

    @property
    def eligible_task_ids(self) -> tuple[str, ...]:
        """Return task IDs that survive the threshold gate."""

        return tuple(
            decision.performance.task_id
            for decision in self.decisions
            if decision.passes_threshold
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the screening summary for manifests or reports."""

        return asdict(self)


@dataclass(frozen=True)
class SSDRequest:
    """SSD request contract covering chemistry, endpoint family, and screening policy."""

    batch_request: BatchPredictionRequest
    endpoint_family: str
    endpoint_filter: EndpointFamilyFilter
    historical_r2_threshold: float
    schema_name: str = SSD_SCHEMA_NAME
    schema_version: str = SSD_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.endpoint_family not in SSD_ALLOWED_ENDPOINT_FAMILIES:
            raise ValueError(
                f"endpoint_family must be one of {SSD_ALLOWED_ENDPOINT_FAMILIES}, got {self.endpoint_family!r}."
            )
        if self.batch_request.endpoint_family != self.endpoint_family:
            raise ValueError(
                "batch_request endpoint family does not match SSD endpoint_family: "
                f"{self.batch_request.endpoint_family!r} != {self.endpoint_family!r}."
            )
        if self.endpoint_filter.endpoint_family != self.endpoint_family:
            raise ValueError("endpoint_filter.endpoint_family must match SSDRequest.endpoint_family.")
        if self.endpoint_filter.endpoint_observation != self.batch_request.endpoint_observation:
            raise ValueError(
                "endpoint_filter.endpoint_observation must match batch_request.endpoint_observation."
            )
        if not 0 <= self.historical_r2_threshold <= 1:
            raise ValueError("historical_r2_threshold must be between 0 and 1.")
        if effect_level_is_required(self.batch_request.effect_type) and self.batch_request.effect_level is None:
            raise ValueError("EC/LC/ICx SSD requests must provide effect_level.")
        if effect_level_is_forbidden(self.batch_request.effect_type) and self.batch_request.effect_level is not None:
            raise ValueError("BCF/BAF SSD requests must not provide effect_level.")
        if self.batch_request.effect_type in BIOACCUMULATION_EFFECT_TYPES and self.batch_request.duration_h <= 0:
            raise ValueError("BCF/BAF SSD requests must retain a positive duration_h.")

    @property
    def effect_type(self) -> str:
        """Expose the underlying effect type directly on the SSD request."""

        return self.batch_request.effect_type

    @property
    def endpoint_observation(self) -> str:
        """Expose the endpoint observation directly on the SSD request."""

        return self.batch_request.endpoint_observation

    def to_dict(self) -> dict[str, Any]:
        """Serialize the SSD request for manifests or API payloads."""

        return asdict(self)


@dataclass(frozen=True)
class SSDStatistic:
    """Placeholder SSD statistic such as HC5 or HC10."""

    name: str
    value: float | None = None
    lower: float | None = None
    upper: float | None = None
    unit: str | None = None
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class SSDUncertaintySummary:
    """Placeholder uncertainty bundle attached to SSD outputs."""

    method: str
    value: float | None = None
    lower: float | None = None
    upper: float | None = None
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class SSDResult:
    """Structured SSD output contract that downstream reporting can consume."""

    request: SSDRequest
    screening_summary: SSDSpeciesScreeningSummary
    prediction_results: tuple[PredictionResult, ...]
    hc5: SSDStatistic
    hc10: SSDStatistic
    uncertainty: SSDUncertaintySummary
    executed_fit: bool = False
    required_outputs: tuple[str, ...] = SSD_REQUIRED_OUTPUTS
    notes: tuple[str, ...] = field(
        default_factory=lambda: (
            "placeholder_ssd_only",
            "no_real_ssd_fit_executed",
        )
    )

    def __post_init__(self) -> None:
        expected_species = set(self.screening_summary.eligible_species_ids)
        result_species = {result.species_id for result in self.prediction_results}
        if result_species - expected_species:
            raise ValueError(
                "prediction_results contain species not admitted by the screening summary: "
                f"{sorted(result_species - expected_species)!r}."
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the SSD result for artifacts or service responses."""

        return asdict(self)


def filter_historical_tasks_for_ssd(
    *,
    endpoint_family: str,
    endpoint_observation: str,
    r2_threshold: float,
    performances: tuple[HistoricalTaskPerformance, ...],
) -> SSDSpeciesScreeningSummary:
    """Apply endpoint-family and task-level R2 screening before SSD fitting."""

    decisions: list[HistoricalTaskFilterDecision] = []
    for performance in performances:
        passes = (
            performance.endpoint_family == endpoint_family
            and performance.endpoint_observation == endpoint_observation
            and performance.r2 >= r2_threshold
        )
        rejection_reason = None
        if not passes:
            if performance.endpoint_family != endpoint_family:
                rejection_reason = "endpoint_family_mismatch"
            elif performance.endpoint_observation != endpoint_observation:
                rejection_reason = "endpoint_observation_mismatch"
            else:
                rejection_reason = "r2_below_threshold"
        decisions.append(
            HistoricalTaskFilterDecision(
                performance=performance,
                r2_threshold=r2_threshold,
                passes_threshold=passes,
                rejection_reason=rejection_reason,
            )
        )
    return SSDSpeciesScreeningSummary(
        endpoint_family=endpoint_family,
        endpoint_observation=endpoint_observation,
        r2_threshold=r2_threshold,
        decisions=tuple(decisions),
    )


def build_placeholder_ssd_result(
    request: SSDRequest,
    *,
    screening_summary: SSDSpeciesScreeningSummary,
    prediction_results: tuple[PredictionResult, ...] = (),
    hc5_notes: tuple[str, ...] = (),
    hc10_notes: tuple[str, ...] = (),
    uncertainty_notes: tuple[str, ...] = (),
) -> SSDResult:
    """Build a stable SSD output contract without executing a fit."""

    return SSDResult(
        request=request,
        screening_summary=screening_summary,
        prediction_results=prediction_results,
        hc5=SSDStatistic(name="hc5", notes=("placeholder_only", *hc5_notes)),
        hc10=SSDStatistic(name="hc10", notes=("placeholder_only", *hc10_notes)),
        uncertainty=SSDUncertaintySummary(
            method="placeholder",
            notes=("placeholder_only", *uncertainty_notes),
        ),
    )


__all__ = [
    "EndpointFamilyFilter",
    "HistoricalTaskFilterDecision",
    "HistoricalTaskPerformance",
    "SSD_ALLOWED_ENDPOINT_FAMILIES",
    "SSD_REQUIRED_OUTPUTS",
    "SSD_SCHEMA_NAME",
    "SSD_SCHEMA_VERSION",
    "SSDRequest",
    "SSDResult",
    "SSDSpeciesScreeningSummary",
    "SSDStatistic",
    "SSDUncertaintySummary",
    "build_placeholder_ssd_result",
    "filter_historical_tasks_for_ssd",
]
