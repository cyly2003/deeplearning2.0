import os
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

from deeplearning2.config.loader import (
    EXPERIMENT_FAMILIES,
    discover_experiment_configs,
    load_experiment_launch_spec,
    load_experiment_records,
    summarize_experiment_families,
)
from deeplearning2.data.dataset_schema import CORE_COLUMNS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"


def _subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    pythonpath_parts = [str(SRC_ROOT), str(PROJECT_ROOT)]
    existing = env.get("PYTHONPATH")
    if existing:
        pythonpath_parts.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    return env


def _create_demo_curated_table(db_path: Path) -> None:
    columns = list(CORE_COLUMNS)
    definition = ", ".join(f'"{column_name}" TEXT' for column_name in columns)
    rows = [
        {
            "species_id": "daphnia_magna",
            "species_name": "Daphnia magna",
            "genus": "Daphnia",
            "family": "Daphniidae",
            "taxon_group_l1": "arthropoda",
            "taxon_group_l2": "crustacea",
            "taxon_group_l3": "cladocera",
            "organism_lifestage": "adult",
            "primary_medium": "water",
            "duration_h": "48",
            "effect_type": "LC",
            "effect_level": "50",
            "endpoint": "LC50 mortality",
            "endpoint_observation": "mortality",
            "is_lethal": "1",
            "is_chronic": "0",
            "is_threshold_endpoint": "0",
            "is_bioaccumulation": "0",
            "smiles": "CCO",
            "target_value": "1.23",
            "target_unit": "mg/L",
        },
        {
            "species_id": "daphnia_magna",
            "species_name": "Daphnia magna",
            "genus": "Daphnia",
            "family": "Daphniidae",
            "taxon_group_l1": "arthropoda",
            "taxon_group_l2": "crustacea",
            "taxon_group_l3": "cladocera",
            "organism_lifestage": "adult",
            "primary_medium": "sediment",
            "duration_h": "96",
            "effect_type": "EC",
            "effect_level": "50",
            "endpoint": "EC50 mortality",
            "endpoint_observation": "mortality",
            "is_lethal": "1",
            "is_chronic": "0",
            "is_threshold_endpoint": "0",
            "is_bioaccumulation": "0",
            "smiles": "CCN",
            "target_value": "2.50",
            "target_unit": "mg/L",
        },
        {
            "species_id": "oncorhynchus_mykiss",
            "species_name": "Oncorhynchus mykiss",
            "genus": "Oncorhynchus",
            "family": "Salmonidae",
            "taxon_group_l1": "chordata",
            "taxon_group_l2": "actinopterygii",
            "taxon_group_l3": "salmoniformes",
            "organism_lifestage": "juvenile",
            "primary_medium": "water",
            "duration_h": "168",
            "effect_type": "BCF",
            "effect_level": None,
            "endpoint": "BCF whole body",
            "endpoint_observation": "bioaccumulation",
            "is_lethal": "0",
            "is_chronic": "1",
            "is_threshold_endpoint": "0",
            "is_bioaccumulation": "1",
            "smiles": "CCC",
            "target_value": "85",
            "target_unit": "L/kg",
        },
    ]

    with sqlite3.connect(db_path) as connection:
        connection.execute(f'CREATE TABLE "ecotox_toxicity_joined_curated" ({definition})')
        quoted_columns = ", ".join(f'"{column_name}"' for column_name in columns)
        placeholders = ", ".join("?" for _ in columns)
        for row in rows:
            connection.execute(
                f'INSERT INTO "ecotox_toxicity_joined_curated" ({quoted_columns}) VALUES ({placeholders})',
                [row[column_name] for column_name in columns],
            )
        connection.commit()


def test_loader_discovers_expected_experiment_families() -> None:
    configs = discover_experiment_configs()
    assert configs, "Expected experiment configs under configs/experiments."

    records = load_experiment_records()
    assert len(records) == len(configs)
    assert {record.family for record in records} == set(EXPERIMENT_FAMILIES)


def test_loader_extracts_known_metadata() -> None:
    records = {record.family: record for record in load_experiment_records()}

    assert records["baseline"].experiment_id == "baseline_water_screen"
    assert records["baseline"].split == "scaffold_holdout"
    assert records["deep"].experiment_id == "deep_direct_joint_water"
    assert records["transfer"].experiment_id == "transfer_pretrain_water_finetune_soil"
    assert records["ablation"].summary == "axes=3"


def test_loader_builds_launch_spec_for_baseline_config() -> None:
    spec = load_experiment_launch_spec(PROJECT_ROOT / "configs" / "experiments" / "baseline" / "baseline_water.yaml")

    assert spec.family == "baseline"
    assert spec.experiment_id == "baseline_water_screen"
    assert spec.medium_scope == "water"
    assert spec.split == "scaffold_holdout"
    assert spec.body["models"][0] == "ridge"


def test_summary_counts_each_family_once() -> None:
    summary = summarize_experiment_families(load_experiment_records())

    for family in EXPERIMENT_FAMILIES:
        assert summary[family]["count"] == 1
    assert summary["baseline"]["with_split"] == 1
    assert summary["ablation"]["with_split"] == 0


def test_runs_list_json_output() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "deeplearning2.cli.main", "runs", "list", "--format", "json"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
        env=_subprocess_env(),
    )

    payload = json.loads(result.stdout)
    assert len(payload) == 4
    assert payload[0]["experiment_id"]
    assert {item["family"] for item in payload} == set(EXPERIMENT_FAMILIES)


def test_runs_summary_text_output() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "deeplearning2.cli.main", "runs", "summary"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
        env=_subprocess_env(),
    )

    assert "family=baseline | count=1" in result.stdout
    assert "family=transfer | count=1" in result.stdout


def test_list_experiments_script_outputs_concise_inventory() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/list_experiments.py"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
        env=_subprocess_env(),
    )

    lines = [line for line in result.stdout.splitlines() if line.strip()]
    assert len(lines) == 4
    assert any(line.startswith("baseline\tbaseline_water_screen\t") for line in lines)


def test_runs_launch_outputs_baseline_placeholder_reports() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "deeplearning2.cli.main",
            "runs",
            "launch",
            "--family",
            "baseline",
            "--config",
            str(PROJECT_ROOT / "configs" / "experiments" / "baseline" / "baseline_water.yaml"),
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
        env=_subprocess_env(),
    )

    payload = json.loads(result.stdout)
    assert len(payload) == 6
    assert payload[0]["runner_family"] == "baseline"
    assert payload[0]["config"]["medium_scope"] == "water"
    assert payload[0]["config"]["extra"]["launch_mode"] == "baseline_task_inventory_launch"
    assert payload[0]["config"]["extra"]["split_is_primary"] is True
    assert payload[0]["config"]["extra"]["task_id"]


def test_runs_launch_binds_to_real_task_inventory_when_sqlite_is_provided(tmp_path: Path) -> None:
    db_path = tmp_path / "launch_inventory.sqlite"
    _create_demo_curated_table(db_path)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "deeplearning2.cli.main",
            "runs",
            "launch",
            "--family",
            "baseline",
            "--config",
            str(PROJECT_ROOT / "configs" / "experiments" / "baseline" / "baseline_water.yaml"),
            "--db-path",
            str(db_path),
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
        env=_subprocess_env(),
    )

    payload = json.loads(result.stdout)
    assert len(payload) == 12
    first = payload[0]["config"]["extra"]
    assert first["task_sample_count"] >= 1
    assert first["target_family"] == "EC_LC_ICx"
    assert "task_id" in first
