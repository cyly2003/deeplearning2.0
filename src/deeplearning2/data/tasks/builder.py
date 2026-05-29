"""Build normalized task records and task inventories from the curated SQLite view."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from deeplearning2.data import load_dataset_entrypoint_contract
from deeplearning2.models.components.targets import EFFECT_TYPE_TO_TARGET_FAMILY
from deeplearning2.models.components.tasks import build_task_id, validate_task_selection


@dataclass(frozen=True)
class NormalizedTaskRecord:
    """Normalized sample row aligned with the formal task system."""

    sample_id: str
    task_id: str
    species_id: str
    effect_type: str
    endpoint_observation: str
    primary_medium: str
    duration_h: float | None
    effect_level: str | None
    target_value: float | None
    target_unit: str | None
    smiles: str
    target_family: str


@dataclass(frozen=True)
class TaskInventoryRow:
    """Aggregated task inventory row for downstream experiment setup."""

    task_id: str
    species_id: str
    effect_type: str
    endpoint_observation: str
    target_family: str
    sample_count: int
    distinct_smiles_count: int
    mediums: tuple[str, ...]


def _coerce_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    return float(text)


def _coerce_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def fetch_normalized_task_records(
    db_path: str | Path | None = None,
    *,
    view_name: str | None = None,
) -> list[NormalizedTaskRecord]:
    """Read and normalize records from the authoritative curated SQLite entrypoint."""

    contract = load_dataset_entrypoint_contract()
    resolved_db_path = Path(db_path) if db_path is not None else contract.sqlite_path
    resolved_view_name = view_name or contract.curated_view

    query = (
        f'SELECT rowid, species_id, effect_type, endpoint_observation, primary_medium, '
        f'duration_h, effect_level, "{contract.target_value_column}", "{contract.target_unit_column}", '
        f'smiles FROM "{resolved_view_name}"'
    )

    records: list[NormalizedTaskRecord] = []
    with sqlite3.connect(resolved_db_path) as connection:
        cursor = connection.execute(query)
        for row in cursor.fetchall():
            (
                rowid,
                species_id,
                effect_type,
                endpoint_observation,
                primary_medium,
                duration_h,
                effect_level,
                target_value,
                target_unit,
                smiles,
            ) = row

            species_id_text = str(species_id).strip()
            effect_type_text = str(effect_type).strip()
            endpoint_text = str(endpoint_observation).strip()
            effect_level_text = _coerce_text(effect_level)
            validate_task_selection(
                species_id_text,
                effect_type_text,
                endpoint_text,
                effect_level=effect_level_text,
            )
            task_id = build_task_id(species_id_text, effect_type_text, endpoint_text)
            target_family = EFFECT_TYPE_TO_TARGET_FAMILY[effect_type_text]

            records.append(
                NormalizedTaskRecord(
                    sample_id=f"{resolved_view_name}:{rowid}",
                    task_id=task_id,
                    species_id=species_id_text,
                    effect_type=effect_type_text,
                    endpoint_observation=endpoint_text,
                    primary_medium=str(primary_medium).strip(),
                    duration_h=_coerce_float(duration_h),
                    effect_level=effect_level_text,
                    target_value=_coerce_float(target_value),
                    target_unit=_coerce_text(target_unit),
                    smiles=str(smiles).strip(),
                    target_family=target_family,
                )
            )
    return records


def build_task_inventory(records: list[NormalizedTaskRecord]) -> list[TaskInventoryRow]:
    """Aggregate normalized records into the formal task inventory."""

    grouped: dict[str, dict[str, object]] = {}
    for record in records:
        bucket = grouped.setdefault(
            record.task_id,
            {
                "species_id": record.species_id,
                "effect_type": record.effect_type,
                "endpoint_observation": record.endpoint_observation,
                "target_family": record.target_family,
                "sample_count": 0,
                "smiles": set(),
                "mediums": set(),
            },
        )
        bucket["sample_count"] = int(bucket["sample_count"]) + 1
        bucket["smiles"].add(record.smiles)
        bucket["mediums"].add(record.primary_medium)

    inventory: list[TaskInventoryRow] = []
    for task_id, bucket in sorted(grouped.items()):
        inventory.append(
            TaskInventoryRow(
                task_id=task_id,
                species_id=str(bucket["species_id"]),
                effect_type=str(bucket["effect_type"]),
                endpoint_observation=str(bucket["endpoint_observation"]),
                target_family=str(bucket["target_family"]),
                sample_count=int(bucket["sample_count"]),
                distinct_smiles_count=len(bucket["smiles"]),
                mediums=tuple(sorted(bucket["mediums"])),
            )
        )
    return inventory


def summarize_task_inventory(records: list[NormalizedTaskRecord]) -> dict[str, int]:
    """Return compact counts for quick contract checks and CLI summaries."""

    inventory = build_task_inventory(records)
    return {
        "sample_count": len(records),
        "task_count": len(inventory),
        "species_count": len({record.species_id for record in records}),
        "target_family_count": len({record.target_family for record in records}),
    }
