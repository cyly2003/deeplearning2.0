from deeplearning2.evaluation.ad.methods import (
    AD_METHODS,
    ADAssessment,
    ADThresholds,
    get_ad_method_spec,
)
from deeplearning2.evaluation.uncertainty.methods import (
    RELIABILITY_OUTPUTS,
    ReliabilityEstimate,
    UncertaintyEstimate,
    get_uncertainty_method_spec,
)


def test_ad_method_specs_cover_authoritative_v1_methods() -> None:
    assert AD_METHODS == ("williams_plot", "leverage", "standardized_residual")

    williams = get_ad_method_spec("williams_plot")
    assert williams.requires_leverage is True
    assert williams.requires_standardized_residual is True
    assert williams.produces_williams_plot_coordinates is True


def test_ad_assessment_requires_required_payloads_for_williams_plot() -> None:
    thresholds = ADThresholds(leverage_threshold=0.4, standardized_residual_threshold=3.0)
    assessment = ADAssessment(
        sample_id="cmpd-001",
        task_id="daphnia_magna__EC_mortality",
        methods=("williams_plot", "leverage", "standardized_residual"),
        in_domain=True,
        leverage=0.21,
        standardized_residual=1.2,
        thresholds=thresholds,
    )

    assert assessment.thresholds == thresholds
    assert assessment.to_dict()["methods"][0] == "williams_plot"


def test_ad_assessment_rejects_incomplete_thresholds() -> None:
    try:
        ADAssessment(
            sample_id="cmpd-001",
            task_id="daphnia_magna__EC_mortality",
            methods=("leverage",),
            in_domain=False,
            leverage=0.81,
            thresholds=ADThresholds(standardized_residual_threshold=3.0),
        )
    except ValueError as exc:
        assert "leverage_threshold" in str(exc)
    else:
        raise AssertionError("Expected leverage-only AD assessment to require leverage_threshold.")


def test_uncertainty_method_specs_share_comparable_reliability_outputs() -> None:
    assert RELIABILITY_OUTPUTS == (
        "predictive_mean",
        "predictive_std",
        "interval_lower",
        "interval_upper",
    )

    ensemble = get_uncertainty_method_spec("deep_ensemble")
    dropout = get_uncertainty_method_spec("mc_dropout")
    assert ensemble.reliability_outputs == dropout.reliability_outputs


def test_uncertainty_estimate_requires_repeated_predictions_and_serializes() -> None:
    reliability = ReliabilityEstimate(
        predictive_mean=1.25,
        predictive_std=0.17,
        interval_lower=0.91,
        interval_upper=1.59,
        sample_count=10,
    )
    estimate = UncertaintyEstimate(
        sample_id="cmpd-002",
        task_id="oncorhynchus_mykiss__BCF_bioaccumulation",
        method="deep_ensemble",
        reliability=reliability,
        member_predictions=(1.1, 1.3, 1.35),
        epistemic_std=0.14,
    )

    payload = estimate.to_dict()
    assert payload["reliability"]["predictive_std"] == 0.17
    assert payload["member_predictions"] == (1.1, 1.3, 1.35)


def test_uncertainty_estimate_rejects_single_member_prediction() -> None:
    try:
        UncertaintyEstimate(
            sample_id="cmpd-002",
            task_id="oncorhynchus_mykiss__BCF_bioaccumulation",
            method="mc_dropout",
            reliability=ReliabilityEstimate(
                predictive_mean=1.25,
                predictive_std=0.17,
                interval_lower=0.91,
                interval_upper=1.59,
            ),
            member_predictions=(1.1,),
        )
    except ValueError as exc:
        assert "at least two" in str(exc)
    else:
        raise AssertionError("Expected repeated-inference uncertainty estimate to require 2+ predictions.")
