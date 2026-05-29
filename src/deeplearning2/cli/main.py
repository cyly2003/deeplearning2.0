"""Minimal CLI entrypoint for framework status inspection."""

from __future__ import annotations

import json

from deeplearning2.config.registry import CONFIG_FILES
from deeplearning2.models.baseline.registry import BASELINE_MODELS
from deeplearning2.models.transfer.protocols import TRANSFER_STAGES


def build_status_payload() -> dict[str, object]:
    """Expose the current framework scope for quick smoke tests."""

    return {
        "config_files": {key: str(value) for key, value in CONFIG_FILES.items()},
        "baseline_models": list(BASELINE_MODELS),
        "transfer_stages": list(TRANSFER_STAGES),
    }


def main() -> None:
    """Print a lightweight project status payload."""

    print(json.dumps(build_status_payload(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
