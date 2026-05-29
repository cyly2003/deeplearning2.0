# Model Execution Contracts

This document defines the placeholder execution contract for the three formal research lines in this repository:

- `baseline`
- `deep`
- `transfer`

The contract is intentionally limited to execution inputs, task/split dependencies, and report outputs. It does not implement real training.

## Alignment With The Authoritative Architecture

The contract follows the project-level requirements in `AGENTS.md` and `docs/统一多任务残差QSAR架构说明.md`:

- the deep line is reserved for a unified multitask main model
- the baseline line must stay comparable under the same task system, split protocol, and target definition logic
- the transfer line must preserve the `pretrain -> finetune` research route
- `task = species + endpoint semantics`
- `effect_level` remains an input condition rather than part of the task identifier

## Shared Contract Objects

All three runner families use the same core dataclasses from `src/deeplearning2/models/components/contracts.py`:

- `TaskContract`
- `SplitDependencyContract`
- `TargetContract`
- `RunnerExecutionConfig`
- `ExecutionArtifacts`
- `ExecutionSummary`
- `ExecutionReportContract`

## Required Input Semantics

`RunnerExecutionConfig` is the common execution entry.

Common requirements:

- `runner_family` must be one of `baseline`, `deep`, `transfer`
- `dataset_view` defaults to `ecotox_toxicity_joined_curated`
- `medium_scope` must be one of:
  - `water`
  - `water_sediment`
  - `water_sediment_soil`
- `split.split_name` must be one of:
  - `scaffold_holdout`
  - `chemical_id_holdout`
  - `medium_transfer_split`

Task requirements:

- `species_id`
- `effect_type`
- `endpoint_observation`
- `effect_level_as_input = true`
- `effect_level_in_task_id = false`
- `excludes_nr = true`

Runner-specific requirements:

- `baseline` must provide `model_name`, constrained by the baseline registry
- `deep` must not provide `model_name`, `transfer_stage`, or `freeze_mode`
- `transfer` must provide both:
  - `transfer_stage`
  - `freeze_mode`

## Required Output Schema

Every runner returns an `ExecutionReportContract`.

Required report sections:

- `task`
- `split`
- `target`
- `artifacts`
- `summary`

Reserved report guarantees:

- same schema version across runner families
- same comparability group across runner families
- explicit placeholder status indicating that real training was not executed

## Why This Contract Exists

This contract gives Worker D a narrow but useful surface:

- it locks the task identity to the unified multitask definition
- it prevents baseline/deep/transfer from drifting onto incompatible split logic
- it gives later workers a stable report skeleton for metrics, AD, uncertainty, SSD, and manuscript outputs

