"""Applicability domain method contracts required in v1."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


AD_METHODS = (
    "williams_plot",
    "leverage",
    "standardized_residual",
)


@dataclass(frozen=True)
class ADMethodSpec:
    """Authoritative applicability-domain method descriptor."""

    name: str
    requires_leverage: bool
    requires_standardized_residual: bool
    produces_williams_plot_coordinates: bool

    def __post_init__(self) -> None:
        if self.name not in AD_METHODS:
            raise ValueError(f"Unsupported AD method '{self.name}'. Expected one of {AD_METHODS}.")


AD_METHOD_SPECS = (
    ADMethodSpec(
        name="williams_plot",
        requires_leverage=True,
        requires_standardized_residual=True,
        produces_williams_plot_coordinates=True,
    ),
    ADMethodSpec(
        name="leverage",
        requires_leverage=True,
        requires_standardized_residual=False,
        produces_williams_plot_coordinates=False,
    ),
    ADMethodSpec(
        name="standardized_residual",
        requires_leverage=False,
        requires_standardized_residual=True,
        produces_williams_plot_coordinates=False,
    ),
)

AD_METHOD_SPEC_BY_NAME = {spec.name: spec for spec in AD_METHOD_SPECS}


@dataclass(frozen=True)
class ADThresholds:
    """Threshold set used to classify an item as in-domain or out-of-domain."""

    leverage_threshold: float | None = None
    standardized_residual_threshold: float | None = None

    def __post_init__(self) -> None:
        if self.leverage_threshold is not None and self.leverage_threshold <= 0:
            raise ValueError("leverage_threshold must be positive when provided.")
        if (
            self.standardized_residual_threshold is not None
            and self.standardized_residual_threshold <= 0
        ):
            raise ValueError("standardized_residual_threshold must be positive when provided.")
        if self.leverage_threshold is None and self.standardized_residual_threshold is None:
            raise ValueError("At least one AD threshold must be provided.")


@dataclass(frozen=True)
class ADAssessment:
    """Per-item AD assessment contract shared by evaluation and prediction flows."""

    sample_id: str
    task_id: str
    methods: tuple[str, ...]
    in_domain: bool
    flagged_by: tuple[str, ...] = ()
    leverage: float | None = None
    standardized_residual: float | None = None
    thresholds: ADThresholds | None = None

    def __post_init__(self) -> None:
        if not self.sample_id:
            raise ValueError("sample_id must not be empty.")
        if not self.task_id:
            raise ValueError("task_id must not be empty.")
        if not self.methods:
            raise ValueError("At least one AD method must be declared.")

        invalid_methods = tuple(method for method in self.methods if method not in AD_METHODS)
        if invalid_methods:
            raise ValueError(f"Unsupported AD methods {invalid_methods}. Expected subset of {AD_METHODS}.")

        invalid_flags = tuple(method for method in self.flagged_by if method not in self.methods)
        if invalid_flags:
            raise ValueError(
                f"flagged_by {invalid_flags} must be a subset of declared methods {self.methods}."
            )

        if "williams_plot" in self.methods:
            if self.leverage is None or self.standardized_residual is None:
                raise ValueError(
                    "Williams plot assessments require both leverage and standardized_residual."
                )

        if "leverage" in self.methods and self.leverage is None:
            raise ValueError("Leverage AD assessments require leverage.")

        if "standardized_residual" in self.methods and self.standardized_residual is None:
            raise ValueError(
                "Standardized residual AD assessments require standardized_residual."
            )

        if self.thresholds is not None:
            if "leverage" in self.methods or "williams_plot" in self.methods:
                if self.thresholds.leverage_threshold is None:
                    raise ValueError(
                        "Leverage-based AD methods require leverage_threshold in thresholds."
                    )
            if "standardized_residual" in self.methods or "williams_plot" in self.methods:
                if self.thresholds.standardized_residual_threshold is None:
                    raise ValueError(
                        "Residual-based AD methods require standardized_residual_threshold in thresholds."
                    )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the assessment for JSON output."""

        return asdict(self)


def get_ad_method_spec(name: str) -> ADMethodSpec:
    """Return the authoritative AD method spec."""

    try:
        return AD_METHOD_SPEC_BY_NAME[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported AD method '{name}'. Expected one of {AD_METHODS}.") from exc


def build_ad_assessment(
    *,
    sample_id: str,
    task_id: str,
    methods: tuple[str, ...],
    in_domain: bool,
    flagged_by: tuple[str, ...] = (),
    leverage: float | None = None,
    standardized_residual: float | None = None,
    thresholds: ADThresholds | None = None,
) -> ADAssessment:
    """Build a validated AD assessment row."""

    return ADAssessment(
        sample_id=sample_id,
        task_id=task_id,
        methods=methods,
        in_domain=in_domain,
        flagged_by=flagged_by,
        leverage=leverage,
        standardized_residual=standardized_residual,
        thresholds=thresholds,
    )


__all__ = [
    "AD_METHOD_SPEC_BY_NAME",
    "AD_METHOD_SPECS",
    "AD_METHODS",
    "ADAssessment",
    "ADMethodSpec",
    "ADThresholds",
    "build_ad_assessment",
    "get_ad_method_spec",
]
