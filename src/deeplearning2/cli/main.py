"""CLI entrypoint for framework status inspection and experiment discovery."""

from __future__ import annotations

import argparse
import json

from deeplearning2.config.registry import CONFIG_FILES
from deeplearning2.cli.runs import register_runs_subcommand
from deeplearning2.models.baseline.registry import BASELINE_MODELS
from deeplearning2.models.transfer.protocols import TRANSFER_STAGES


def build_status_payload() -> dict[str, object]:
    """Expose the current framework scope for quick smoke tests."""

    return {
        "config_files": {key: str(value) for key, value in CONFIG_FILES.items()},
        "baseline_models": list(BASELINE_MODELS),
        "transfer_stages": list(TRANSFER_STAGES),
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level CLI parser."""

    parser = argparse.ArgumentParser(prog="deeplearning2-status")
    subparsers = parser.add_subparsers(dest="command")
    register_runs_subcommand(subparsers)
    return parser


def main(argv: list[str] | None = None) -> None:
    """Print project status or dispatch to subcommands."""

    parser = build_parser()
    args = parser.parse_args(argv)

    if hasattr(args, "handler"):
        args.handler(args)
        return

    print(json.dumps(build_status_payload(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
