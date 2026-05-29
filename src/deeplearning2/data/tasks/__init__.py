"""Task-building interfaces for the curated SQLite entrypoint."""

from deeplearning2.data.tasks.builder import (
    NormalizedTaskRecord,
    TaskInventoryRow,
    build_task_inventory,
    fetch_normalized_task_records,
    summarize_task_inventory,
)

__all__ = [
    "NormalizedTaskRecord",
    "TaskInventoryRow",
    "build_task_inventory",
    "fetch_normalized_task_records",
    "summarize_task_inventory",
]
