"""Supported transfer-learning strategies."""

from __future__ import annotations


TRANSFER_STAGES = (
    "pretrain_water",
    "pretrain_water_sediment",
    "finetune_soil",
)

FREEZE_MODES = (
    "none",
    "chemical_encoder_partial",
    "chemical_encoder_full",
)
