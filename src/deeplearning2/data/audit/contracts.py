"""Contracts for auditing the curated SQLite entrypoint."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from deeplearning2.data.dataset_schema import CORE_COLUMNS
from deeplearning2.data.sqlite import CURATED_VIEW_NAME, SQLiteDataSource


DEFAULT_KEY_FIELDS = (
    "species_id",
    "primary_medium",
    "effect_type",
    "endpoint_observation",
    "smiles",
    "target_value",
)


@dataclass(frozen=True)
class SQLiteAuditContract:
    """Declarative checks for the curated SQLite entrypoint."""

    db_path: Path
    view_name: str
    required_columns: tuple[str, ...]
    key_fields: tuple[str, ...]
    min_row_count: int = 1

    def __post_init__(self) -> None:
        if not self.view_name:
            raise ValueError("view_name must be a non-empty SQLite relation name.")
        if self.min_row_count < 0:
            raise ValueError("min_row_count must be non-negative.")
        if not self.required_columns:
            raise ValueError("required_columns must not be empty.")
        if not self.key_fields:
            raise ValueError("key_fields must not be empty.")

        required_set = set(self.required_columns)
        missing_key_fields = [field_name for field_name in self.key_fields if field_name not in required_set]
        if missing_key_fields:
            raise ValueError(
                "key_fields must be a subset of required_columns. "
                f"Missing from required_columns: {missing_key_fields}."
            )

    @classmethod
    def from_data_source(
        cls,
        data_source: SQLiteDataSource | None = None,
        *,
        min_row_count: int = 1,
        required_columns: tuple[str, ...] = CORE_COLUMNS,
        key_fields: tuple[str, ...] = DEFAULT_KEY_FIELDS,
    ) -> "SQLiteAuditContract":
        """Build the default audit contract from the authoritative data source."""

        source = data_source or SQLiteDataSource()
        return cls(
            db_path=Path(source.db_path),
            view_name=source.curated_view,
            required_columns=tuple(required_columns),
            key_fields=tuple(key_fields),
            min_row_count=min_row_count,
        )


@dataclass(frozen=True)
class AuditCheck:
    """Single check result within an audit report."""

    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class SQLiteAuditReport:
    """Structured result for SQLite entrypoint auditing."""

    contract: SQLiteAuditContract
    database_exists: bool
    relation_exists: bool
    relation_type: str | None
    row_count: int | None
    available_columns: tuple[str, ...] = ()
    missing_columns: tuple[str, ...] = ()
    key_field_non_null_counts: dict[str, int] = field(default_factory=dict)
    checks: tuple[AuditCheck, ...] = ()

    @property
    def passed(self) -> bool:
        """Return whether every declared audit check passed."""

        return all(check.passed for check in self.checks)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the audit report into plain Python objects."""

        payload = asdict(self)
        payload["contract"]["db_path"] = str(self.contract.db_path)
        payload["passed"] = self.passed
        return payload

    def summary_lines(self) -> tuple[str, ...]:
        """Create concise human-readable summary lines for CLI output."""

        relation_label = self.relation_type or "missing"
        row_count_label = "n/a" if self.row_count is None else str(self.row_count)
        lines = [
            f"status={'PASS' if self.passed else 'FAIL'}",
            f"database_exists={self.database_exists}",
            f"relation={self.contract.view_name}",
            f"relation_type={relation_label}",
            f"row_count={row_count_label}",
            f"missing_columns={len(self.missing_columns)}",
        ]
        if self.key_field_non_null_counts:
            details = ",".join(
                f"{field_name}:{count}"
                for field_name, count in sorted(self.key_field_non_null_counts.items())
            )
            lines.append(f"key_field_non_null={details}")
        lines.extend(f"{check.name}={'PASS' if check.passed else 'FAIL'}:{check.detail}" for check in self.checks)
        return tuple(lines)


def build_default_sqlite_audit_contract(
    *,
    min_row_count: int = 1,
) -> SQLiteAuditContract:
    """Expose the authoritative curated SQLite audit contract."""

    return SQLiteAuditContract.from_data_source(min_row_count=min_row_count)
