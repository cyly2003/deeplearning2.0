"""Audit the authoritative curated SQLite entrypoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from deeplearning2.data.audit import SQLiteAuditContract, audit_sqlite_entrypoint
from deeplearning2.data.audit.contracts import DEFAULT_KEY_FIELDS
from deeplearning2.data.dataset_schema import CORE_COLUMNS
from deeplearning2.data.sqlite import CURATED_VIEW_NAME
from deeplearning2.paths import DATA_DB_PATH


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for entrypoint auditing."""

    parser = argparse.ArgumentParser(
        description=(
            "Audit a SQLite database against the project's curated entrypoint contract "
            "without rebuilding the original raw join chain."
        )
    )
    parser.add_argument(
        "--db-path",
        default=str(DATA_DB_PATH),
        help="Path to the SQLite database file to audit.",
    )
    parser.add_argument(
        "--view-name",
        default=CURATED_VIEW_NAME,
        help="SQLite table or view name to audit.",
    )
    parser.add_argument(
        "--min-row-count",
        type=int,
        default=1,
        help="Minimum row count expected for the audited relation.",
    )
    return parser


def build_contract_from_args(args: argparse.Namespace) -> SQLiteAuditContract:
    """Convert parsed arguments into a reusable audit contract."""

    return SQLiteAuditContract(
        db_path=Path(args.db_path),
        view_name=args.view_name,
        required_columns=tuple(CORE_COLUMNS),
        key_fields=DEFAULT_KEY_FIELDS,
        min_row_count=args.min_row_count,
    )


def main(argv: list[str] | None = None) -> int:
    """Run the SQLite entrypoint audit and print a compact summary."""

    parser = build_parser()
    args = parser.parse_args(argv)
    report = audit_sqlite_entrypoint(build_contract_from_args(args))

    for line in report.summary_lines():
        print(line)

    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
