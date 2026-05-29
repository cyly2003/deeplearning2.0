"""SQLite entrypoint definitions."""

from __future__ import annotations

from dataclasses import dataclass

from deeplearning2.paths import DATA_DB_PATH


CURATED_VIEW_NAME = "ecotox_toxicity_joined_curated"


@dataclass(frozen=True)
class SQLiteDataSource:
    """Reference to the curated SQLite entrypoint."""

    db_path: str = str(DATA_DB_PATH)
    curated_view: str = CURATED_VIEW_NAME
