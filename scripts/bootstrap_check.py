"""Framework bootstrap smoke check."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from deeplearning2.cli.main import build_status_payload


if __name__ == "__main__":
    payload = build_status_payload()
    print(f"configs={len(payload['config_files'])}")
    print(f"baseline_models={len(payload['baseline_models'])}")
    print(f"transfer_stages={len(payload['transfer_stages'])}")
