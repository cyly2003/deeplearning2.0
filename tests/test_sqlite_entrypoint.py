import sqlite3
import subprocess
import sys
from pathlib import Path

from deeplearning2.data.audit import SQLiteAuditContract, audit_sqlite_entrypoint
from deeplearning2.data.audit.contracts import DEFAULT_KEY_FIELDS
from deeplearning2.data.dataset_schema import CORE_COLUMNS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "audit_sqlite_entrypoint.py"


def _create_curated_relation(
    db_path: Path,
    *,
    relation_name: str = "ecotox_toxicity_joined_curated",
    include_all_columns: bool = True,
    populate_key_fields: bool = True,
) -> None:
    columns = list(CORE_COLUMNS)
    if not include_all_columns:
        columns.remove("target_unit")

    definition = ", ".join(f'"{column_name}" TEXT' for column_name in columns)
    values = {
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
        "effect_type": "EC",
        "effect_level": "50",
        "endpoint": "EC50 mortality",
        "endpoint_observation": "mortality",
        "is_lethal": "1",
        "is_chronic": "0",
        "is_threshold_endpoint": "0",
        "is_bioaccumulation": "0",
        "smiles": "CCO",
        "target_value": "1.23",
        "target_unit": "mg/L",
    }
    if not populate_key_fields:
        for field_name in DEFAULT_KEY_FIELDS:
            values[field_name] = None

    with sqlite3.connect(db_path) as connection:
        connection.execute(f'CREATE TABLE "{relation_name}" ({definition})')
        insertable_columns = columns
        placeholders = ", ".join("?" for _ in insertable_columns)
        quoted_columns = ", ".join(f'"{column_name}"' for column_name in insertable_columns)
        connection.execute(
            f'INSERT INTO "{relation_name}" ({quoted_columns}) VALUES ({placeholders})',
            [values[column_name] for column_name in insertable_columns],
        )
        connection.commit()


def test_audit_sqlite_entrypoint_passes_for_temp_table(tmp_path: Path) -> None:
    db_path = tmp_path / "demo.sqlite"
    _create_curated_relation(db_path)

    contract = SQLiteAuditContract(
        db_path=db_path,
        view_name="ecotox_toxicity_joined_curated",
        required_columns=tuple(CORE_COLUMNS),
        key_fields=DEFAULT_KEY_FIELDS,
        min_row_count=1,
    )
    report = audit_sqlite_entrypoint(contract)

    assert report.passed is True
    assert report.database_exists is True
    assert report.relation_exists is True
    assert report.row_count == 1
    assert report.missing_columns == ()
    assert report.key_field_non_null_counts["species_id"] == 1


def test_audit_sqlite_entrypoint_fails_when_required_columns_are_missing(tmp_path: Path) -> None:
    db_path = tmp_path / "missing_column.sqlite"
    _create_curated_relation(db_path, include_all_columns=False)

    contract = SQLiteAuditContract(
        db_path=db_path,
        view_name="ecotox_toxicity_joined_curated",
        required_columns=tuple(CORE_COLUMNS),
        key_fields=DEFAULT_KEY_FIELDS,
        min_row_count=1,
    )
    report = audit_sqlite_entrypoint(contract)

    assert report.passed is False
    assert "target_unit" in report.missing_columns
    check_map = {check.name: check for check in report.checks}
    assert check_map["required_columns"].passed is False
    assert check_map["key_fields"].passed is False


def test_audit_sqlite_entrypoint_fails_when_relation_is_missing(tmp_path: Path) -> None:
    db_path = tmp_path / "missing_view.sqlite"
    with sqlite3.connect(db_path):
        pass

    contract = SQLiteAuditContract(
        db_path=db_path,
        view_name="ecotox_toxicity_joined_curated",
        required_columns=tuple(CORE_COLUMNS),
        key_fields=DEFAULT_KEY_FIELDS,
        min_row_count=1,
    )
    report = audit_sqlite_entrypoint(contract)

    assert report.passed is False
    assert report.database_exists is True
    assert report.relation_exists is False
    assert report.row_count is None


def test_audit_sqlite_entrypoint_fails_when_key_fields_are_all_null(tmp_path: Path) -> None:
    db_path = tmp_path / "null_keys.sqlite"
    _create_curated_relation(db_path, populate_key_fields=False)

    contract = SQLiteAuditContract(
        db_path=db_path,
        view_name="ecotox_toxicity_joined_curated",
        required_columns=tuple(CORE_COLUMNS),
        key_fields=DEFAULT_KEY_FIELDS,
        min_row_count=1,
    )
    report = audit_sqlite_entrypoint(contract)

    assert report.passed is False
    check_map = {check.name: check for check in report.checks}
    assert check_map["required_columns"].passed is True
    assert check_map["key_fields"].passed is False
    assert report.key_field_non_null_counts["target_value"] == 0


def test_cli_script_reports_pass_for_temp_sqlite(tmp_path: Path) -> None:
    db_path = tmp_path / "cli_demo.sqlite"
    _create_curated_relation(db_path)

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--db-path",
            str(db_path),
            "--view-name",
            "ecotox_toxicity_joined_curated",
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "status=PASS" in completed.stdout
    assert "relation=ecotox_toxicity_joined_curated" in completed.stdout
