"""SQLite audit helpers for the curated project entrypoint."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from deeplearning2.data.audit.contracts import AuditCheck, SQLiteAuditContract, SQLiteAuditReport


def _quote_identifier(identifier: str) -> str:
    escaped = identifier.replace('"', '""')
    return f'"{escaped}"'


def _fetch_relation_metadata(connection: sqlite3.Connection, relation_name: str) -> tuple[bool, str | None]:
    cursor = connection.execute(
        """
        SELECT type
        FROM sqlite_master
        WHERE name = ?
          AND type IN ('table', 'view')
        LIMIT 1
        """,
        (relation_name,),
    )
    row = cursor.fetchone()
    if row is None:
        return False, None
    return True, str(row[0])


def _fetch_columns(connection: sqlite3.Connection, relation_name: str) -> tuple[str, ...]:
    cursor = connection.execute(f"PRAGMA table_info({_quote_identifier(relation_name)})")
    rows = cursor.fetchall()
    return tuple(str(row[1]) for row in rows)


def _fetch_row_count(connection: sqlite3.Connection, relation_name: str) -> int:
    cursor = connection.execute(f"SELECT COUNT(*) FROM {_quote_identifier(relation_name)}")
    row = cursor.fetchone()
    return int(row[0]) if row is not None else 0


def _fetch_key_field_non_null_counts(
    connection: sqlite3.Connection,
    relation_name: str,
    key_fields: tuple[str, ...],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for field_name in key_fields:
        cursor = connection.execute(
            f"SELECT COUNT(*) FROM {_quote_identifier(relation_name)} "
            f"WHERE {_quote_identifier(field_name)} IS NOT NULL"
        )
        row = cursor.fetchone()
        counts[field_name] = int(row[0]) if row is not None else 0
    return counts


def audit_sqlite_entrypoint(contract: SQLiteAuditContract) -> SQLiteAuditReport:
    """Audit a SQLite database against the curated entrypoint contract."""

    db_path = Path(contract.db_path)
    if not db_path.exists():
        checks = (
            AuditCheck("database_exists", False, f"SQLite file not found: {db_path}"),
            AuditCheck("relation_exists", False, f"Cannot inspect relation {contract.view_name!r} without a database file."),
            AuditCheck("required_columns", False, "Required columns were not inspected because the database file is missing."),
            AuditCheck("row_count", False, "Row-count check was not executed because the database file is missing."),
            AuditCheck("key_fields", False, "Key-field checks were not executed because the database file is missing."),
        )
        return SQLiteAuditReport(
            contract=contract,
            database_exists=False,
            relation_exists=False,
            relation_type=None,
            row_count=None,
            checks=checks,
        )

    with sqlite3.connect(db_path) as connection:
        relation_exists, relation_type = _fetch_relation_metadata(connection, contract.view_name)
        if not relation_exists:
            checks = (
                AuditCheck("database_exists", True, f"SQLite file found: {db_path}"),
                AuditCheck("relation_exists", False, f"Relation {contract.view_name!r} was not found in sqlite_master."),
                AuditCheck("required_columns", False, "Required columns were not inspected because the relation is missing."),
                AuditCheck("row_count", False, "Row-count check was not executed because the relation is missing."),
                AuditCheck("key_fields", False, "Key-field checks were not executed because the relation is missing."),
            )
            return SQLiteAuditReport(
                contract=contract,
                database_exists=True,
                relation_exists=False,
                relation_type=None,
                row_count=None,
                checks=checks,
            )

        available_columns = _fetch_columns(connection, contract.view_name)
        available_set = set(available_columns)
        missing_columns = tuple(
            column_name for column_name in contract.required_columns if column_name not in available_set
        )
        row_count = _fetch_row_count(connection, contract.view_name)

        key_field_non_null_counts: dict[str, int] = {}
        key_fields_passed = False
        key_fields_detail = "Key-field counts were skipped because required columns are missing."
        if not missing_columns:
            key_field_non_null_counts = _fetch_key_field_non_null_counts(
                connection,
                contract.view_name,
                contract.key_fields,
            )
            empty_key_fields = [
                field_name
                for field_name, non_null_count in key_field_non_null_counts.items()
                if row_count > 0 and non_null_count == 0
            ]
            key_fields_passed = not empty_key_fields
            if empty_key_fields:
                key_fields_detail = (
                    "Key fields exist but contain no non-null values: "
                    + ", ".join(empty_key_fields)
                )
            else:
                key_fields_detail = "All declared key fields contain at least one non-null value."

        checks = (
            AuditCheck("database_exists", True, f"SQLite file found: {db_path}"),
            AuditCheck(
                "relation_exists",
                True,
                f"Relation {contract.view_name!r} found as SQLite {relation_type}.",
            ),
            AuditCheck(
                "required_columns",
                not missing_columns,
                "All required columns are present."
                if not missing_columns
                else "Missing required columns: " + ", ".join(missing_columns),
            ),
            AuditCheck(
                "row_count",
                row_count >= contract.min_row_count,
                f"Observed {row_count} rows; expected at least {contract.min_row_count}.",
            ),
            AuditCheck("key_fields", key_fields_passed, key_fields_detail),
        )

        return SQLiteAuditReport(
            contract=contract,
            database_exists=True,
            relation_exists=True,
            relation_type=relation_type,
            row_count=row_count,
            available_columns=available_columns,
            missing_columns=missing_columns,
            key_field_non_null_counts=key_field_non_null_counts,
            checks=checks,
        )
