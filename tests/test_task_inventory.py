import json
import sqlite3
import subprocess
import sys
from pathlib import Path

from deeplearning2.data.dataset_schema import CORE_COLUMNS
from deeplearning2.data.tasks import (
    build_task_inventory,
    fetch_normalized_task_records,
    summarize_task_inventory,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "list_task_inventory.py"


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


def test_fetch_normalized_task_records_builds_canonical_task_ids(tmp_path: Path) -> None:
    db_path = tmp_path / "inventory.sqlite"
    _create_demo_curated_table(db_path)

    records = fetch_normalized_task_records(db_path)

    assert len(records) == 3
    assert records[0].task_id == "daphnia_magna__EC_mortality"
    assert records[0].target_family == "EC_LC_ICx"
    assert records[2].task_id == "oncorhynchus_mykiss__BCF_bioaccumulation"
    assert records[2].target_family == "BCF_BAF"


def test_build_task_inventory_aggregates_task_counts_and_mediums(tmp_path: Path) -> None:
    db_path = tmp_path / "inventory.sqlite"
    _create_demo_curated_table(db_path)

    inventory = build_task_inventory(fetch_normalized_task_records(db_path))

    assert len(inventory) == 2
    ec_task = next(row for row in inventory if row.task_id == "daphnia_magna__EC_mortality")
    assert ec_task.sample_count == 2
    assert ec_task.distinct_smiles_count == 2
    assert ec_task.mediums == ("sediment", "water")


def test_summarize_task_inventory_returns_compact_counts(tmp_path: Path) -> None:
    db_path = tmp_path / "inventory.sqlite"
    _create_demo_curated_table(db_path)

    summary = summarize_task_inventory(fetch_normalized_task_records(db_path))

    assert summary == {
        "sample_count": 3,
        "task_count": 2,
        "species_count": 2,
        "target_family_count": 2,
    }


def test_task_inventory_cli_outputs_json_rows(tmp_path: Path) -> None:
    db_path = tmp_path / "inventory.sqlite"
    _create_demo_curated_table(db_path)

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--db-path",
            str(db_path),
            "--format",
            "json",
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )

    payload = json.loads(completed.stdout)
    assert len(payload) == 2
    assert payload[0]["task_id"]
    assert {row["target_family"] for row in payload} == {"EC_LC_ICx", "BCF_BAF"}
