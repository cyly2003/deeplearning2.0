"""Project path helpers."""

from __future__ import annotations

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PACKAGE_ROOT.parent
PROJECT_ROOT = SRC_ROOT.parent
CONFIG_ROOT = PROJECT_ROOT / "configs"
DOCS_ROOT = PROJECT_ROOT / "docs"
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"
DATA_DB_PATH = PROJECT_ROOT / "ecotox_clean.sqlite"
