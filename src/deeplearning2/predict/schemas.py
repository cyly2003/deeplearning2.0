"""Structured prediction contracts shared by single, batch, and SSD workflows."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from deeplearning2.models.components.tasks import build_task_id, validate_task_selection


SINGLE_PREDICTION_FIELDS = (
    "smiles",
    "species_id",
    "primary_medium",
    "duration_h",
    "effect_type",
    "endpoint_observation",
    "effect_level",
)
PREDICTION_SCHEMA_NAME = "prediction_contract"
PREDICTION_SCHEMA_VERSION = "0.1.0"
CONCENTRATION_EFFECT_TYPES = ("EC", "LC", "ICx")
THRESHOLD_EFFECT_TYPES = ("NOEC", "LOEC")
BIOACCUMULATION_EFFECT_TYPES = ("BCF", "BAF")
ENDPOINT_FAMILY_MAP = {
    "EC": "EC_LC_ICx",
    "LC": "EC_LC_ICx",
    "ICx": "EC_LC_ICx",
    "NOEC": "NOEC_LOEC",
    "LOEC": "NOEC_LOEC",
    "BCF": "BCF_BAF",
    "BAF": "BCF_BAF",
}


def resolve_endpoint_family(effect_type: str) -> str:
    """Resolve an effect type into the formal endpoint family used downstream."""

    try:
        return ENDPOINT_FAMILY_MAP[effect_type]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported effect_type={effect_type!r}. Expected one of {tuple(ENDPOINT_FAMILY_MAP)}."
        ) from exc


def effect_level_is_required(effect_type: str) -> bool:
    """Return whether the formal task semantics require effect_level."""

    return effect_type in CONCENTRATION_EFFECT_TYPES


def effect_level_is_forbidden(effect_type: str) -> bool:
    """Return whether the formal task semantics forbid effect_level."""

    return effect_type in BIOACCUMULATION_EFFECT_TYPES


def validate_effect_level_value(effect_level: str | int | float | None) -> None:
    """Validate optional effect_level payload without forcing a single numeric type."""

    if effect_level is None:
        return
    try:
        numeric = float(effect_level)
    except (TypeError, ValueError) as exc:
        raise ValueError("effect_level must be numeric when provided.") from exc
    if numeric <= 0 or numeric >= 100:
        raise ValueError("effect_level must be between 0 and 100 for EC/LC/ICx-style requests.")


@dataclass(frozen=True)
class PredictionRequest:
    """Single-species prediction request contract."""

    smiles: str
    species_id: str
    primary_medium: str
    duration_h: float
    effect_type: str
    endpoint_observation: str
    effect_level: str | int | float | None = None
    schema_name: str = PREDICTION_SCHEMA_NAME
    schema_version: str = PREDICTION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.smiles:
            raise ValueError("smiles must be a non-empty string.")
        if not self.primary_medium:
            raise ValueError("primary_medium must be a non-empty string.")
        if self.duration_h <= 0:
            raise ValueError("duration_h must be positive.")
        validate_effect_level_value(self.effect_level)
        validate_task_selection(
            self.species_id,
            self.effect_type,
            self.endpoint_observation,
            effect_level=self.effect_level,
        )

    @property
    def task_id(self) -> str:
        """Return the formal task ID consumed by downstream model selection."""

        return build_task_id(self.species_id, self.effect_type, self.endpoint_observation)

    @property
    def endpoint_family(self) -> str:
        """Return the endpoint family used by target-space and SSD workflows."""

        return resolve_endpoint_family(self.effect_type)

    @property
    def uses_effect_level(self) -> bool:
        """Return whether this request should propagate effect-level features."""

        return effect_level_is_required(self.effect_type)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the request for manifests or API payloads."""

        return asdict(self)


@dataclass(frozen=True)
class BatchPredictionRequest:
    """Batch prediction request sharing one chemistry/context template across species."""

    smiles: str
    species_ids: tuple[str, ...]
    primary_medium: str
    duration_h: float
    effect_type: str
    endpoint_observation: str
    effect_level: str | int | float | None = None
    schema_name: str = PREDICTION_SCHEMA_NAME
    schema_version: str = PREDICTION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.species_ids:
            raise ValueError("species_ids must contain at least one species.")
        if len(set(self.species_ids)) != len(self.species_ids):
            raise ValueError("species_ids must not contain duplicates.")
        for species_id in self.species_ids:
            PredictionRequest(
                smiles=self.smiles,
                species_id=species_id,
                primary_medium=self.primary_medium,
                duration_h=self.duration_h,
                effect_type=self.effect_type,
                endpoint_observation=self.endpoint_observation,
                effect_level=self.effect_level,
            )

    @property
    def endpoint_family(self) -> str:
        """Return the family shared by all batch-expanded prediction requests."""

        return resolve_endpoint_family(self.effect_type)

    def to_single_requests(self) -> tuple[PredictionRequest, ...]:
        """Expand the batch template into per-species prediction requests."""

        return tuple(
            PredictionRequest(
                smiles=self.smiles,
                species_id=species_id,
                primary_medium=self.primary_medium,
                duration_h=self.duration_h,
                effect_type=self.effect_type,
                endpoint_observation=self.endpoint_observation,
                effect_level=self.effect_level,
            )
            for species_id in self.species_ids
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the batch request for manifests or API payloads."""

        return asdict(self)


@dataclass(frozen=True)
class ApplicabilityDomainStatus:
    """Prediction-time AD placeholder aligned with the first-version policy."""

    in_domain: bool | None
    leverage: float | None = None
    standardized_residual: float | None = None
    method: str = "williams_plot"
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class PredictionUncertainty:
    """Prediction-time uncertainty placeholder for ensemble or MC-dropout outputs."""

    value: float | None
    lower: float | None = None
    upper: float | None = None
    method: str = "deep_ensemble"
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class PredictionResult:
    """Structured prediction output without executing a real model."""

    request: PredictionRequest
    predicted_value: float | None = None
    output_value: float | None = None
    output_unit: str | None = None
    target_training_space: str | None = None
    target_output_space: str | None = None
    ad_status: ApplicabilityDomainStatus | None = None
    uncertainty: PredictionUncertainty | None = None
    executed_model: bool = False
    notes: tuple[str, ...] = field(
        default_factory=lambda: (
            "placeholder_prediction_only",
            "no_real_model_inference_executed",
        )
    )

    @property
    def task_id(self) -> str:
        """Expose task identity directly on the prediction result."""

        return self.request.task_id

    @property
    def species_id(self) -> str:
        """Expose species identity directly on the prediction result."""

        return self.request.species_id

    def to_dict(self) -> dict[str, Any]:
        """Serialize the prediction result for artifacts or service responses."""

        return asdict(self)


def expand_batch_prediction_request(batch_request: BatchPredictionRequest) -> tuple[PredictionRequest, ...]:
    """Expand a batch prediction request into validated single prediction requests."""

    return batch_request.to_single_requests()


def build_placeholder_prediction_result(
    request: PredictionRequest,
    *,
    target_training_space: str | None = None,
    target_output_space: str | None = None,
    ad_status: ApplicabilityDomainStatus | None = None,
    uncertainty: PredictionUncertainty | None = None,
    notes: tuple[str, ...] = (),
) -> PredictionResult:
    """Build a non-executed placeholder prediction result with a stable contract."""

    merged_notes = (
        "placeholder_prediction_only",
        "no_real_model_inference_executed",
        *notes,
    )
    return PredictionResult(
        request=request,
        target_training_space=target_training_space,
        target_output_space=target_output_space,
        ad_status=ad_status,
        uncertainty=uncertainty,
        notes=merged_notes,
    )


__all__ = [
    "ApplicabilityDomainStatus",
    "BIOACCUMULATION_EFFECT_TYPES",
    "BatchPredictionRequest",
    "CONCENTRATION_EFFECT_TYPES",
    "ENDPOINT_FAMILY_MAP",
    "PREDICTION_SCHEMA_NAME",
    "PREDICTION_SCHEMA_VERSION",
    "PredictionRequest",
    "PredictionResult",
    "PredictionUncertainty",
    "SINGLE_PREDICTION_FIELDS",
    "THRESHOLD_EFFECT_TYPES",
    "build_placeholder_prediction_result",
    "effect_level_is_forbidden",
    "effect_level_is_required",
    "expand_batch_prediction_request",
    "resolve_endpoint_family",
    "validate_effect_level_value",
]
