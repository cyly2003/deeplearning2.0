"""Print a compact task inventory from the curated SQLite entrypoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from deeplearning2.data.tasks import build_task_inventory, fetch_normalized_task_records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="List formal task inventory rows from the curated SQLite entrypoint.",
    )
    parser.add_argument("--db-path", help="Optional SQLite path override.", default=None)
    parser.add_argument("--view-name", help="Optional curated view override.", default=None)
    parser.add_argument("--format", choices=("text", "json"), default="text")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    records = fetch_normalized_task_records(args.db_path, view_name=args.view_name)
    inventory = build_task_inventory(records)

    if args.format == "json":
        print(json.dumps([row.__dict__ for row in inventory], ensure_ascii=False, indent=2))
        return

    for row in inventory:
        mediums = ",".join(row.mediums)
        print(
            "\t".join(
                [
                    row.task_id,
                    row.target_family,
                    str(row.sample_count),
                    str(row.distinct_smiles_count),
                    mediums,
                ]
            )
        )


if __name__ == "__main__":
    main()
