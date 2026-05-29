from deeplearning2.predict.batch import BatchPredictionRequest, expand_batch_prediction_request
from deeplearning2.predict.schemas import (
    PredictionRequest,
    build_placeholder_prediction_result,
    resolve_endpoint_family,
)
from deeplearning2.ssd.contracts import (
    EndpointFamilyFilter,
    HistoricalTaskPerformance,
    SSDRequest,
    SSD_REQUIRED_OUTPUTS,
    build_placeholder_ssd_result,
    filter_historical_tasks_for_ssd,
)


def test_single_prediction_requires_effect_level_for_ec_like_requests() -> None:
    try:
        PredictionRequest(
            smiles="CCO",
            species_id="daphnia_magna",
            primary_medium="water",
            duration_h=48,
            effect_type="EC",
            endpoint_observation="mortality",
        )
    except ValueError as exc:
        assert "effect_level is required" in str(exc)
    else:
        raise AssertionError("Expected EC-like request to require effect_level.")


def test_single_prediction_forbids_effect_level_for_bcf_like_requests() -> None:
    try:
        PredictionRequest(
            smiles="CCO",
            species_id="oncorhynchus_mykiss",
            primary_medium="water",
            duration_h=96,
            effect_type="BCF",
            endpoint_observation="bioaccumulation",
            effect_level=50,
        )
    except ValueError as exc:
        assert "must not be provided" in str(exc)
    else:
        raise AssertionError("Expected BCF-like request to reject effect_level.")


def test_batch_prediction_expands_into_validated_single_requests() -> None:
    batch = BatchPredictionRequest(
        smiles="CCO",
        species_ids=("daphnia_magna", "pimephales_promelas"),
        primary_medium="water",
        duration_h=48,
        effect_type="EC",
        endpoint_observation="mortality",
        effect_level=50,
    )

    requests = expand_batch_prediction_request(batch)
    assert len(requests) == 2
    assert tuple(request.species_id for request in requests) == (
        "daphnia_magna",
        "pimephales_promelas",
    )
    assert all(request.endpoint_family == "EC_LC_ICx" for request in requests)


def test_prediction_placeholder_keeps_task_identity_and_nonexecuted_state() -> None:
    request = PredictionRequest(
        smiles="CCO",
        species_id="daphnia_magna",
        primary_medium="water",
        duration_h=48,
        effect_type="LC",
        endpoint_observation="mortality",
        effect_level=50,
    )

    result = build_placeholder_prediction_result(
        request,
        target_training_space="task_internal_log_space",
        target_output_space="original_effect_concentration",
    )

    assert result.executed_model is False
    assert result.task_id == "daphnia_magna__EC_mortality"
    assert result.target_output_space == "original_effect_concentration"


def test_resolve_endpoint_family_keeps_formal_family_boundaries() -> None:
    assert resolve_endpoint_family("EC") == "EC_LC_ICx"
    assert resolve_endpoint_family("NOEC") == "NOEC_LOEC"
    assert resolve_endpoint_family("BCF") == "BCF_BAF"


def test_ssd_request_requires_matching_endpoint_family_and_effect_level_policy() -> None:
    batch = BatchPredictionRequest(
        smiles="CCO",
        species_ids=("daphnia_magna", "pimephales_promelas"),
        primary_medium="water",
        duration_h=48,
        effect_type="EC",
        endpoint_observation="mortality",
        effect_level=50,
    )
    request = SSDRequest(
        batch_request=batch,
        endpoint_family="EC_LC_ICx",
        endpoint_filter=EndpointFamilyFilter(
            endpoint_family="EC_LC_ICx",
            allowed_effect_types=("EC", "LC", "ICx"),
            endpoint_observation="mortality",
        ),
        historical_r2_threshold=0.6,
    )

    assert request.effect_type == "EC"
    assert request.endpoint_observation == "mortality"


def test_ssd_request_rejects_mixed_endpoint_family() -> None:
    batch = BatchPredictionRequest(
        smiles="CCO",
        species_ids=("daphnia_magna",),
        primary_medium="water",
        duration_h=96,
        effect_type="BCF",
        endpoint_observation="bioaccumulation",
    )

    try:
        SSDRequest(
            batch_request=batch,
            endpoint_family="EC_LC_ICx",
            endpoint_filter=EndpointFamilyFilter(
                endpoint_family="EC_LC_ICx",
                allowed_effect_types=("EC",),
                endpoint_observation="bioaccumulation",
            ),
            historical_r2_threshold=0.5,
        )
    except ValueError as exc:
        assert "does not match" in str(exc)
    else:
        raise AssertionError("Expected SSD request to reject mixed endpoint family.")


def test_ssd_screening_filters_by_family_observation_and_r2_threshold() -> None:
    performances = (
        HistoricalTaskPerformance(
            task_id="daphnia_magna__EC_mortality",
            species_id="daphnia_magna",
            effect_type="EC",
            endpoint_observation="mortality",
            r2=0.72,
            sample_count=128,
        ),
        HistoricalTaskPerformance(
            task_id="pimephales_promelas__EC_mortality",
            species_id="pimephales_promelas",
            effect_type="EC",
            endpoint_observation="mortality",
            r2=0.41,
            sample_count=96,
        ),
        HistoricalTaskPerformance(
            task_id="oncorhynchus_mykiss__NOEC_mortality",
            species_id="oncorhynchus_mykiss",
            effect_type="NOEC",
            endpoint_observation="mortality",
            r2=0.93,
            sample_count=64,
        ),
    )

    summary = filter_historical_tasks_for_ssd(
        endpoint_family="EC_LC_ICx",
        endpoint_observation="mortality",
        r2_threshold=0.6,
        performances=performances,
    )

    assert summary.eligible_species_ids == ("daphnia_magna",)
    assert summary.eligible_task_ids == ("daphnia_magna__EC_mortality",)
    assert summary.decisions[1].rejection_reason == "r2_below_threshold"
    assert summary.decisions[2].rejection_reason == "endpoint_family_mismatch"


def test_ssd_placeholder_exposes_hc_outputs_and_uncertainty_structure() -> None:
    batch = BatchPredictionRequest(
        smiles="CCO",
        species_ids=("daphnia_magna",),
        primary_medium="water",
        duration_h=48,
        effect_type="EC",
        endpoint_observation="mortality",
        effect_level=50,
    )
    request = SSDRequest(
        batch_request=batch,
        endpoint_family="EC_LC_ICx",
        endpoint_filter=EndpointFamilyFilter(
            endpoint_family="EC_LC_ICx",
            allowed_effect_types=("EC", "LC", "ICx"),
            endpoint_observation="mortality",
        ),
        historical_r2_threshold=0.6,
    )
    summary = filter_historical_tasks_for_ssd(
        endpoint_family="EC_LC_ICx",
        endpoint_observation="mortality",
        r2_threshold=0.6,
        performances=(
            HistoricalTaskPerformance(
                task_id="daphnia_magna__EC_mortality",
                species_id="daphnia_magna",
                effect_type="EC",
                endpoint_observation="mortality",
                r2=0.72,
                sample_count=128,
            ),
        ),
    )
    prediction = build_placeholder_prediction_result(
        PredictionRequest(
            smiles="CCO",
            species_id="daphnia_magna",
            primary_medium="water",
            duration_h=48,
            effect_type="EC",
            endpoint_observation="mortality",
            effect_level=50,
        )
    )

    result = build_placeholder_ssd_result(
        request,
        screening_summary=summary,
        prediction_results=(prediction,),
    )

    assert result.executed_fit is False
    assert result.required_outputs == SSD_REQUIRED_OUTPUTS
    assert result.hc5.name == "hc5"
    assert result.hc10.name == "hc10"
    assert result.uncertainty.method == "placeholder"
