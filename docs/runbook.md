# Runbook

本 runbook 只定义实验发现、命名和后续运行入口约定；当前阶段不触发真实训练。

## 1. 目标

当前 CLI 骨架负责三件事：

1. 从 `configs/experiments` 发现实验 YAML
2. 抽取 `experiment` 元信息并分类
3. 为 baseline / deep / transfer / ablation 提供统一清单与汇总入口

这与项目级架构保持一致：

- baseline 是正式研究线，不是附属脚本
- deep 对应统一多任务主模型
- transfer 对应 `pretrain -> finetune` 研究线
- ablation 对应结构化消融矩阵

## 2. 配置约定

实验配置放在：

- `configs/experiments/baseline`
- `configs/experiments/deep`
- `configs/experiments/transfer`
- `configs/experiments/ablation`

每个实验 YAML 当前统一采用：

```yaml
experiment:
  id: some_experiment_id
  family: baseline|deep|transfer|ablation
```

建议补充字段：

- `medium_scope`
- `split`
- baseline: `models`
- deep: `architecture`
- transfer: `stages`
- ablation: `axes`

## 3. CLI 入口

默认状态入口：

```bash
python -m deeplearning2.cli.main
```

列出实验：

```bash
python -m deeplearning2.cli.main runs list
python -m deeplearning2.cli.main runs list --format json
```

按 family 汇总：

```bash
python -m deeplearning2.cli.main runs summary
python -m deeplearning2.cli.main runs summary --format json
```

脚本式简表：

```bash
python scripts/list_experiments.py
```

## 4. 后续运行入口约定

当前只做 skeleton，不做真实训练。后续建议保持以下入口语义：

1. baseline
   - `python -m deeplearning2.cli.main runs launch --family baseline --config <path>`
   - 内部对接 `src/deeplearning2/models/baseline/runner.py`
2. deep
   - `python -m deeplearning2.cli.main runs launch --family deep --config <path>`
   - 内部对接统一多任务主模型 runner
3. transfer
   - `python -m deeplearning2.cli.main runs launch --family transfer --config <path>`
   - 内部对接 `pretrain / finetune` 分阶段 runner
4. ablation
   - `python -m deeplearning2.cli.main runs materialize-ablation --config <path>`
   - 负责从消融轴生成实验清单，而不是直接训练

## 5. 研究一致性要求

后续真实运行实现时，必须保持：

- 同一任务体系：`task = species + endpoint semantics`
- `effect_level` 是输入条件，不进入任务 ID
- baseline / deep / transfer 共用 split 协议
- deep 主线保持统一多任务模型，不拆成多个独立主模型
- ablation 继续覆盖 `RDKit/Morgan`、descriptor fusion、context、endpoint encoding、effect level features、training mode
