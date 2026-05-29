# 共享契约说明

本文档描述当前阶段已经代码化冻结的共享契约。  
它服务于 `M1 / WS0-WS2`，目标是避免 baseline、deep、transfer、AD、SSD 各自维护隐式规则。

权威上游仍然是：

- [AGENTS.md](/D:/深度学习2.0/AGENTS.md)
- [统一多任务残差QSAR架构说明.md](/D:/深度学习2.0/docs/统一多任务残差QSAR架构说明.md)

## 1. 数据入口契约

当前正式数据入口由 [`src/deeplearning2/data/contracts.py`](/D:/深度学习2.0/src/deeplearning2/data/contracts.py) 统一加载：

- source type：`sqlite`
- sqlite path：`ecotox_clean.sqlite`
- curated view：`ecotox_toxicity_joined_curated`
- 默认不重建原始 join 链
- `effect_level` 不进入任务 ID
- `NR` 不进入正式建模

该契约与 [`configs/data/dataset.yaml`](/D:/深度学习2.0/configs/data/dataset.yaml) 保持一一对应。

## 2. 任务语义契约

当前任务语义由 [`src/deeplearning2/models/components/tasks.py`](/D:/深度学习2.0/src/deeplearning2/models/components/tasks.py) 统一提供：

- 正式定义：`task = species + endpoint semantics`
- task ID components：
  - `species`
  - `effect_type`
  - `endpoint_observation`
- `effect_level` 是输入条件，不是 task ID 组成部分
- `BCF / BAF` 禁止携带 `effect_level`
- `EC / LC / ICx` 要求显式 `effect_level`
- `LC + mortality` 在正式任务语义中并入 `EC + mortality`

## 3. 目标空间契约

当前目标空间由 [`src/deeplearning2/models/components/targets.py`](/D:/深度学习2.0/src/deeplearning2/models/components/targets.py) 统一提供。

正式 family 固定为：

- `EC_LC_ICx`
- `NOEC_LOEC`
- `BCF_BAF`

要求：

- `EC / LC / ICx` 使用 task-internal 稳定训练空间，并保留 `effect_level` 特征
- `NOEC / LOEC` 使用阈值型专属空间
- `BCF / BAF` 可使用对数训练空间，但不使用 `effect_level`

## 4. Split 契约

当前 split manifest 由 [`src/deeplearning2/data/splits/protocols.py`](/D:/深度学习2.0/src/deeplearning2/data/splits/protocols.py) 提供。

正式协议固定为：

- `scaffold_holdout`：primary split
- `chemical_id_holdout`：supplementary holdout
- `medium_transfer_split`：cross-medium transfer

其中主 split 仍然是 `scaffold_holdout`。

## 5. 结果汇总契约

当前统一结果 bundle 由 [`src/deeplearning2/evaluation/reports/schemas.py`](/D:/深度学习2.0/src/deeplearning2/evaluation/reports/schemas.py) 提供。

它定义了后续各条工作流都能共享的最小结果结构：

- `RunMetadata`
- `TaskMetricRow`
- `SplitMetricRow`
- `SpeciesMetricRow`
- `UnifiedResultBundle`

当前统一 metrics 固定为：

- `r2`
- `rmse`
- `mae`

这份结果契约是后续：

- baseline/deep/transfer 训练输出
- AD / uncertainty
- prediction / SSD
- 论文图表系统

之间的数据交换基线。

## 6. 当前阶段的直接用途

从现在开始，新实现应优先复用这些代码化契约，而不是再次从 YAML 或旧脚本手写解析：

1. 数据读取工作流复用 `load_dataset_entrypoint_contract()`
2. 任务构造工作流复用 `load_task_semantics_contract()` 与 `build_task_id()`
3. 训练入口复用 `build_target_contract()` 与 split manifest
4. 结果汇总工作流复用 `UnifiedResultBundle`

这样做的目的不是增加抽象层，而是减少不同研究线之间的语义漂移。

## 7. 任务清单骨架

当前已经补上任务清单构造骨架：

- [`src/deeplearning2/data/tasks/builder.py`](/D:/深度学习2.0/src/deeplearning2/data/tasks/builder.py)
- [`scripts/list_task_inventory.py`](/D:/深度学习2.0/scripts/list_task_inventory.py)

它们当前负责：

- 从 `ecotox_toxicity_joined_curated` 读取样本
- 规范化 `species_id / effect_type / endpoint_observation / effect_level`
- 生成正式 `task_id`
- 聚合成 task inventory

当前输出可直接支持：

1. baseline / deep / transfer 共用任务清单
2. 后续 split manifest 绑定
3. species/task 粒度样本量预审计
