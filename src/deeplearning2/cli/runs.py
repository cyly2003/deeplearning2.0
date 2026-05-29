"""Experiment listing and summary CLI helpers."""

from __future__ import annotations

import argparse
import json

from deeplearning2.config.loader import load_experiment_records, summarize_experiment_families


def _build_list_payload() -> list[dict[str, str | None]]:
    return [record.to_dict() for record in load_experiment_records()]


def _build_summary_payload() -> dict[str, dict[str, int]]:
    return summarize_experiment_families(load_experiment_records())


def handle_runs_list(args: argparse.Namespace) -> None:
    """Print experiment configs as JSON or text rows."""

    records = load_experiment_records()
    if args.format == "json":
        print(json.dumps([record.to_dict() for record in records], ensure_ascii=False, indent=2))
        return

    for record in records:
        parts = [
            f"id={record.experiment_id}",
            f"family={record.family}",
            f"config={record.config_path.as_posix()}",
        ]
        if record.medium_scope:
            parts.append(f"medium_scope={record.medium_scope}")
        if record.split:
            parts.append(f"split={record.split}")
        if record.summary:
            parts.append(f"summary={record.summary}")
        print(" | ".join(parts))


def handle_runs_summary(args: argparse.Namespace) -> None:
    """Print family-level experiment counts."""

    payload = _build_summary_payload()
    if args.format == "json":
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    for family, stats in payload.items():
        print(
            " | ".join(
                [
                    f"family={family}",
                    f"count={stats['count']}",
                    f"with_split={stats['with_split']}",
                    f"with_medium_scope={stats['with_medium_scope']}",
                ]
            )
        )


def register_runs_subcommand(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register the runs CLI namespace."""

    parser = subparsers.add_parser(
        "runs",
        help="Inspect experiment configs for baseline/deep/transfer/ablation skeletons.",
    )
    runs_subparsers = parser.add_subparsers(dest="runs_command", required=True)

    list_parser = runs_subparsers.add_parser("list", help="List discovered experiment configs.")
    list_parser.add_argument("--format", choices=("text", "json"), default="text")
    list_parser.set_defaults(handler=handle_runs_list)

    summary_parser = runs_subparsers.add_parser(
        "summary",
        help="Summarize discovered experiment configs by family.",
    )
    summary_parser.add_argument("--format", choices=("text", "json"), default="text")
    summary_parser.set_defaults(handler=handle_runs_summary)
