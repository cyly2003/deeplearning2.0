# 开发者工作流（最小骨架版）

本文档面向本项目的科研建模开发协作，目标是为“刚起步但需要保持专业性”的代码库提供最小可执行工作流。  
它只约束开发协作、自检与骨架一致性，不替代项目级权威文档。

## 1. 开始前先读什么

每次开始新任务前，先阅读：

1. `D:\深度学习2.0\AGENTS.md`
2. `D:\深度学习2.0\docs\统一多任务残差QSAR架构说明.md`

如果历史脚本、旧实验记录、口头约定与上述两份文档冲突，以这两份文档为准。

## 2. 当前阶段推荐的最小仓库骨架

本阶段不要求训练逻辑已经可跑，但建议尽快补齐下面这套最小结构：

```text
configs/
  data/
  tasks/
  models/
  experiments/
    baseline/
    deep/
    transfer/
    ablation/
docs/
tests/
```

其中：

- `configs/data/`：数据入口、SQLite 路径、主视图、字段选择策略
- `configs/tasks/`：任务语义、终点结构化字段、目标空间约束
- `configs/models/`：chemical encoder、context encoder、fusion 与 heads
- `configs/experiments/baseline/`：Ridge、RandomForest 等基线实验
- `configs/experiments/deep/`：联合训练主线
- `configs/experiments/transfer/`：pretrain / finetune 主线
- `configs/experiments/ablation/`：RDKit/Morgan、grouped/flat、context on/off 等消融

## 3. 配置设计最小约束

配置层建议优先表达“研究契约”，而不是只表达程序参数。

至少要明确这些信息：

1. 数据入口使用 `ecotox_clean.sqlite`
2. 主视图使用 `ecotox_toxicity_joined_curated`
3. 正式任务粒度是 `species + endpoint semantics`
4. `effect_level` 是输入条件，不是任务 ID
5. `BCF / BAF` 纳入统一模型，但不走 `effect_level` 路径
6. 深度模型配置中要能表达 `RDKit + Morgan + grouped descriptors + context`
7. 实验配置中要能区分 `baseline / deep / transfer / ablation`

## 4. 推荐开发顺序

建议按下面顺序推进，能减少返工：

1. 先补配置骨架，再补代码骨架
2. 先固定任务定义与目标空间，再写训练入口
3. 先让 baseline / deep / transfer / ablation 的目录和配置层次清晰，再开始堆实现细节
4. 任何会影响科研结论的改动，都先写进配置或文档，再改代码

## 5. 最小自检清单

在提交或交接前，至少检查：

1. 是否仍以 `AGENTS.md` 与架构说明文档为准
2. 是否误把 `effect_level` 写进任务 ID
3. 是否把 `NR` 混入正式建模任务
4. 是否把 `BCF / BAF` 错误地走成了 `ECx` 风格配置
5. baseline 与 deep 是否共享同一任务定义逻辑
6. 是否为后续 `scaffold holdout` 与 `medium transfer split` 预留了配置位
7. 是否记录了随机种子、数据划分、单位与输出位置

## 6. 测试建议

当前阶段的测试只应覆盖“骨架与配置契约”，不要假设训练逻辑已经稳定可运行。

推荐最小命令：

```bash
pytest D:\深度学习2.0\tests\test_project_structure.py -q
pytest D:\深度学习2.0\tests\test_configs.py -q
```

如果后续补齐了更多骨架，也可统一执行：

```bash
pytest D:\深度学习2.0\tests -q
```

## 7. 这两份测试文件主要检查什么

- `tests/test_project_structure.py`
  - 权威文档是否存在
  - 整理版 SQLite 入口是否存在
  - 文档资源目录是否存在
  - 骨架测试本身是否纳入仓库

- `tests/test_configs.py`
  - 是否存在 `configs/`
  - 是否存在 `data / tasks / models / experiments` 分层
  - 数据配置是否指向整理版 SQLite 与主视图
  - 任务配置是否体现结构化终点语义
  - 模型配置是否体现 grouped RDKit + Morgan + context
  - 实验配置是否覆盖 baseline / deep / transfer / ablation

## 8. 交接时建议附带的信息

多人并行协作时，交接说明最好至少包含：

1. 改了哪些文件
2. 改动是否影响已有接口
3. 哪些测试已经跑过
4. 哪些失败是“骨架尚未补齐”导致，而不是实现 bug
5. 下一位开发者最先该补哪个配置或目录

这会比只留一串命令或一句“已经改好了”更适合科研项目协作。
