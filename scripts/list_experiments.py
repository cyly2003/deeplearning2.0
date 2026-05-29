"""Print a concise experiment inventory from configs/experiments."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from deeplearning2.config.loader import load_experiment_records


def main() -> None:
    """Emit one concise line per configured experiment."""

    for record in load_experiment_records():
        print(f"{record.family}\t{record.experiment_id}\t{record.config_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
