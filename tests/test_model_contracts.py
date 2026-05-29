from deeplearning2.evaluation.reports.schemas import (
    REPORT_REQUIRED_SECTIONS,
    REPORT_SCHEMA_NAME,
    REPORT_SCHEMA_VERSION,
)
from deeplearning2.models.baseline.runner import run_baseline_experiment
from deeplearning2.models.components.contracts import (
    ExecutionReportContract,
    RunnerExecutionConfig,
    SplitDependencyContract,
    TargetContract,
    TaskContract,
)
from deeplearning2.models.deep.runner import run_deep_experiment
from deeplearning2.models.transfer.runner import run_transfer_experiment


def _task() -> TaskContract:
    return TaskContract(
        species_id="daphnia_magna",
        effect_type="EC",
        endpoint_observation="mortality",
    )


def _split(name: str = "scaffold_holdout") -> SplitDependencyContract:
    return SplitDependencyContract(
        split_name=name,
        split_group="primary_research_split",
    )


def _target(uses_effect_level_features: bool = True) -> TargetContract:
    return TargetContract(
        family="ecx_like",
        training_space="task_internal_log_space",
        output_space="original_effect_concentration",
        uses_effect_level_features=uses_effect_level_features,
        notes="placeholder_target_contract",
    )


def test_report_schema_metadata_is_stable() -> None:
    assert REPORT_SCHEMA_NAME == "model_execution_report"
    assert REPORT_SCHEMA_VERSION == "0.1.0"
    assert REPORT_REQUIRED_SECTIONS == (
        "task",
        "split",
        "target",
        "artifacts",
        "summary",
    )


def test_baseline_deep_transfer_share_execution_report_shape() -> None:
    baseline = run_baseline_experiment(
        RunnerExecutionConfig(
            runner_family="baseline",
            run_name="baseline_ridge_demo",
            task=_task(),
            split=_split(),
            target=_target(),
            medium_scope="water",
            model_name="ridge",
        )
    )
    deep = run_deep_experiment(
        RunnerExecutionConfig(
            runner_family="deep",
            run_name="deep_joint_demo",
            task=_task(),
            split=_split(),
            target=_target(),
            medium_scope="water_sediment",
        )
    )
    transfer = run_transfer_experiment(
        RunnerExecutionConfig(
            runner_family="transfer",
            run_name="transfer_soil_demo",
            task=_task(),
            split=_split("medium_transfer_split"),
            target=_target(),
            medium_scope="water_sediment_soil",
            transfer_stage="finetune_soil",
            freeze_mode="chemical_encoder_partial",
        )
    )

    reports = (baseline, deep, transfer)
    assert all(isinstance(report, ExecutionReportContract) for report in reports)
    assert {report.schema_version for report in reports} == {"0.1.0"}
    assert {report.report_sections for report in reports} == {REPORT_REQUIRED_SECTIONS}
    assert {report.summary.comparability_group for report in reports} == {"unified_multitask_qsar"}
    assert all(report.summary.executed_training is False for report in reports)


def test_task_contract_matches_authoritative_semantics() -> None:
    task = _task()
    assert task.task_id_definition == "task = species + endpoint semantics"
    assert task.effect_level_as_input is True
    assert task.effect_level_in_task_id is False
    assert task.excludes_nr is True


def test_baseline_runner_requires_registered_model_name() -> None:
    try:
        RunnerExecutionConfig(
            runner_family="baseline",
            run_name="bad_baseline",
            task=_task(),
            split=_split(),
            target=_target(),
            medium_scope="water",
            model_name="svm",
        )
    except ValueError as exc:
        assert "BASELINE_MODELS" in str(exc)
    else:
        raise AssertionError("Expected baseline model_name validation to fail.")


def test_transfer_runner_requires_stage_and_freeze_mode() -> None:
    try:
        RunnerExecutionConfig(
            runner_family="transfer",
            run_name="bad_transfer",
            task=_task(),
            split=_split("medium_transfer_split"),
            target=_target(uses_effect_level_features=False),
            medium_scope="water_sediment_soil",
        )
    except ValueError as exc:
        assert "TRANSFER_STAGES" in str(exc)
    else:
        raise AssertionError("Expected transfer config validation to fail.")


def test_bcf_like_target_can_disable_effect_level_features() -> None:
    target = TargetContract(
        family="bcf_baf",
        training_space="log_bioaccumulation_space",
        output_space="original_bioaccumulation_factor",
        uses_effect_level_features=False,
        notes="BCF/BAF do not use effect_level but retain duration_h.",
    )
    config = RunnerExecutionConfig(
        runner_family="deep",
        run_name="bcf_demo",
        task=TaskContract(
            species_id="oncorhynchus_mykiss",
            effect_type="BCF",
            endpoint_observation="bioaccumulation",
        ),
        split=_split(),
        target=target,
        medium_scope="water",
    )
    report = run_deep_experiment(config)
    assert report.config.target.uses_effect_level_features is False
    assert report.config.task.effect_level_in_task_id is False
