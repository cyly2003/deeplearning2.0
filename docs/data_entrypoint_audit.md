# 数据入口审计骨架

本模块用于对项目的正式数据入口做契约化检查，当前只围绕以下入口工作：

- `ecotox_clean.sqlite`
- 主视图：`ecotox_toxicity_joined_curated`

它的目的不是重做数据清洗，而是在训练、评估、迁移学习和批量预测之前，先确认“当前 SQLite 入口是否满足后续流程的最小前提”。

## 检查什么

当前骨架覆盖以下几类检查：

1. SQLite 文件是否存在
2. 指定关系是否存在
   - 允许是 `view`
   - 也兼容测试时使用 `table`
3. Schema / 列存在性
   - 以 [`src/deeplearning2/data/dataset_schema.py`](/D:/深度学习2.0/src/deeplearning2/data/dataset_schema.py:1) 中的 `CORE_COLUMNS` 为权威列集合
4. 基础行数检查
   - 默认要求 `row_count >= 1`
   - 可通过脚本参数调整
5. 关键字段契约
   - 当前关键字段为：
     - `species_id`
     - `primary_medium`
     - `effect_type`
     - `endpoint_observation`
     - `smiles`
     - `target_value`
   - 在列存在的前提下，审计器会检查这些字段在当前关系中是否至少存在一条非空记录

## 不检查什么

当前骨架刻意不做以下事情：

1. 不重建原始 join 链
2. 不回源重算原始清洗逻辑
3. 不自动修复字段
4. 不判断单位换算是否已经完全正确
5. 不判断终点语义映射是否已经科研上最终定稿
6. 不替代后续更细的数据质量审计

这意味着它是“入口完整性检查”，不是“全量数据治理系统”。

## 为什么不重建原始 join 链

项目级权威约束已经明确：

- 正式开发入口以整理版 SQLite 为主
- 原始 SQLite 仅用于回源核对、异常追溯、字段补充
- 新实现中不要默认重建完整原始 join 链

因此本审计骨架只把 `ecotox_clean.sqlite -> ecotox_toxicity_joined_curated` 当作正式接口来校验。这样做有三个直接好处：

1. 保持开发入口唯一，减少不同 worker 对数据来源的理解漂移
2. 让 baseline / deep / transfer / ablation 共用同一个数据契约
3. 让测试可以基于临时 SQLite 完成，不依赖真实项目数据库文件

## 脚本用法

脚本位置：

- [`scripts/audit_sqlite_entrypoint.py`](/D:/深度学习2.0/scripts/audit_sqlite_entrypoint.py:1)

示例：

```powershell
python scripts\audit_sqlite_entrypoint.py
python scripts\audit_sqlite_entrypoint.py --db-path D:\tmp\demo.sqlite --view-name ecotox_toxicity_joined_curated
python scripts\audit_sqlite_entrypoint.py --min-row-count 100
```

输出为简洁摘要，包含：

- 总体 PASS / FAIL
- 数据库是否存在
- 关系类型
- 行数
- 缺失列数量
- 关键字段非空计数
- 每项检查的单独结果

## 后续扩展方向

后续真实接入项目库时，可以在当前契约对象基础上继续增加：

1. 关键字段缺失率阈值
2. 任务定义相关字段的枚举检查
3. `effect_type / effect_level / endpoint_observation` 组合合法性检查
4. 面向 split 前的数据泄漏预检查
5. 面向 SSD / AD / uncertainty 的专用数据入口约束
