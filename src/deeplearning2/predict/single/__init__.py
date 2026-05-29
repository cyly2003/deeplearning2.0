"""Single prediction interfaces."""

from deeplearning2.predict.schemas import (
    ApplicabilityDomainStatus,
    PREDICTION_SCHEMA_NAME,
    PREDICTION_SCHEMA_VERSION,
    PredictionRequest,
    PredictionResult,
    PredictionUncertainty,
    build_placeholder_prediction_result,
)

__all__ = [
    "ApplicabilityDomainStatus",
    "PREDICTION_SCHEMA_NAME",
    "PREDICTION_SCHEMA_VERSION",
    "PredictionRequest",
    "PredictionResult",
    "PredictionUncertainty",
    "build_placeholder_prediction_result",
]
