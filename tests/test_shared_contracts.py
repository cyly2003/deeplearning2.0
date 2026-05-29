from deeplearning2.data import DatasetEntrypointContract, load_dataset_entrypoint_contract
from deeplearning2.data.splits.protocols import (
    PRIMARY_SPLIT_PROTOCOL,
    SPLIT_PROTOCOLS,
    build_split_protocol_manifest,
    get_split_protocol,
)
from deeplearning2.evaluation.reports.schemas import (
    RESULT_BUNDLE_SCHEMA_NAME,
    RESULT_BUNDLE_SCHEMA_VERSION,
    SUPPORTED_METRIC_NAMES,
    RunMetadata,
    SplitMetricRow,
    SpeciesMetricRow,
    TaskMetricRow,
    build_result_bundle,
)
from deeplearning2.models.components.tasks import (
    TASK_ID_COMPONENTS,
    TASK_ID_DEFINITION,
    build_task_id,
    canonicalize_task_effect_type,
    load_task_semantics_contract,
    validate_task_selection,
)
from deeplearning2.models.components.targets import (
    TARGET_SPACE_POLICY,
    build_target_contract,
    load_target_space_specs,
    resolve_target_space_spec,
)


def test_dataset_entrypoint_contract_matches_authoritative_sqlite_policy() -> None:
    contract = load_dataset_entrypoint_contract()

    assert isinstance(contract, DatasetEntrypointContract)
    assert contract.source_type == "sqlite"
    assert contract.curated_view == "ecotox_toxicity_joined_curated"
    assert contract.task_expression == "species + endpoint_semantics"
    assert contract.effect_level_is_task_id_component is False
    assert contract.exclude_nr is True


def test_task_semantics_contract_loads_and_keeps_effect_level_out_of_task_id() -> None:
    contract = load_task_semantics_contract()

    assert contract.formal_task_definition == TASK_ID_DEFINITION
    assert contract.task_id_components == TASK_ID_COMPONENTS
    assert contract.effect_level_part_of_task_id is False
    assert "BCF" in contract.effect_level_forbidden_for


def test_task_id_uses_canonical_ec_mortality_semantics() -> None:
    assert canonicalize_task_effect_type("LC", "mortality") == "EC"
    assert build_task_id("daphnia_magna", "LC", "mortality") == "daphnia_magna__EC_mortality"


def test_task_selection_enforces_effect_level_policy() -> None:
    validate_task_selection(
        "daphnia_magna",
        "EC",
        "mortality",
        effect_level=50,
    )

    try:
        validate_task_selection("oncorhynchus_mykiss", "BCF", "bioaccumulation", effect_level=50)
    except ValueError as exc:
        assert "must not be provided" in str(exc)
    else:
        raise AssertionError("Expected BCF task selection to reject effect_level.")


def test_target_space_specs_cover_all_formal_effect_type_families() -> None:
    specs = {spec.family_name: spec for spec in load_target_space_specs()}

    assert set(specs) == {"EC_LC_ICx", "NOEC_LOEC", "BCF_BAF"}
    assert resolve_target_space_spec("LC").family_name == "EC_LC_ICx"
    assert resolve_target_space_spec("BCF").uses_effect_level_features is False
    assert build_target_contract("EC").family == "ec_lc_icx"
    assert TARGET_SPACE_POLICY == "task_internal_unification_with_separate_training_and_output_spaces"


def test_split_protocol_manifest_stays_aligned_with_primary_split_policy() -> None:
    manifest = build_split_protocol_manifest()
    manifest_map = {spec.split_name: spec for spec in manifest}

    assert tuple(manifest_map) == SPLIT_PROTOCOLS
    assert manifest_map[PRIMARY_SPLIT_PROTOCOL].is_primary is True
    assert get_split_protocol("medium_transfer_split").purpose == "cross_medium_transfer"


def test_unified_result_bundle_serializes_task_split_and_species_rows() -> None:
    metadata = RunMetadata(
        run_name="deep_joint_water_demo",
        runner_family="deep",
        medium_scope="water",
        split_name="scaffold_holdout",
        dataset_view="ecotox_toxicity_joined_curated",
        seed=42,
    )
    bundle = build_result_bundle(
        metadata,
        task_rows=(
            TaskMetricRow(
                task_id="daphnia_magna__EC_mortality",
                species_id="daphnia_magna",
                effect_type="EC",
                endpoint_observation="mortality",
                split_name="scaffold_holdout",
                sample_count=128,
                metrics={"r2": 0.72, "rmse": 0.31, "mae": 0.21},
            ),
        ),
        split_rows=(
            SplitMetricRow(
                split_name="scaffold_holdout",
                runner_family="deep",
                medium_scope="water",
                task_count=24,
                metrics={"r2": 0.68, "rmse": 0.35, "mae": 0.24},
            ),
        ),
        species_rows=(
            SpeciesMetricRow(
                species_id="daphnia_magna",
                split_name="scaffold_holdout",
                task_count=4,
                metrics={"r2": 0.7, "rmse": 0.33, "mae": 0.22},
            ),
        ),
    )

    payload = bundle.to_dict()
    assert bundle.schema_name == RESULT_BUNDLE_SCHEMA_NAME
    assert bundle.schema_version == RESULT_BUNDLE_SCHEMA_VERSION
    assert tuple(SUPPORTED_METRIC_NAMES) == ("r2", "rmse", "mae")
    assert payload["task_rows"][0]["metrics"]["r2"] == 0.72
    assert payload["split_rows"][0]["metrics"]["rmse"] == 0.35
