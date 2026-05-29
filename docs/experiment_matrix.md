# 正式实验矩阵

本文件把项目第一版正式研究线拆成可执行实验矩阵，用于指导 baseline、deep、transfer、AD、uncertainty、SSD 的并行推进。

重要说明：

1. 本文件只定义“应该做什么实验”和“实验之间如何依赖”。
2. 本文件不报告任何虚构结果。
3. 并非所有实验都应一次性全量展开；建议按优先级分层执行。

权威约束来源：

- [AGENTS.md](</D:/深度学习2.0/AGENTS.md>)
- [统一多任务残差QSAR架构说明.md](</D:/深度学习2.0/docs/统一多任务残差QSAR架构说明.md>)

## 1. 全部实验共用的固定协议

以下内容在所有实验中默认固定，除非实验条目明确说明自己在做该轴的消融：

1. 数据入口：`ecotox_clean.sqlite`
2. 主视图：`ecotox_toxicity_joined_curated`
3. 正式任务定义：`task = species + endpoint semantics`
4. `effect_level` 只作为条件输入
5. `LCx` 语义并入 `ECx + mortality`
6. `NR` 不参与正式建模
7. `BCF / BAF` 纳入统一主模型，但不走 `effect_level` 路径
8. 主 split：`scaffold holdout`
9. 补充 split：`chemical_id holdout`、`medium transfer split`
10. 训练路线必须覆盖：
    - direct joint
    - transfer learning
11. baseline 至少覆盖：
    - `Ridge / ElasticNet`
    - `RandomForest`
    - `XGBoost`
    - `LightGBM`
    - `CatBoost`
12. uncertainty 至少覆盖：
    - `deep ensemble`
    - `MC dropout`
13. AD 至少覆盖：
    - `Williams plot`
    - leverage
    - standardized residual

## 2. 执行层级

建议把实验分成 4 层，而不是上来做完整笛卡尔积。

### L1：协议与可运行性验证

目标：

- 确认任务构造、目标空间、split、结果汇总都可运行。

### L2：正式主线对比

目标：

- 回答 baseline vs deep、direct joint vs transfer 两个主问题。

### L3：关键消融

目标：

- 回答化学表征、上下文、终点语义、effect level 编码方式对性能的贡献。

### L4：可靠性与应用

目标：

- 补齐 AD、uncertainty、SSD。

## 3. L1 协议与可运行性验证

| 实验 ID | 目标 | 固定设置 | 变量 | 交付物 | 前置依赖 | 优先级 |
| --- | --- | --- | --- | --- | --- | --- |
| P-01 | 验证任务构造可运行 | 主数据入口、正式任务定义、主 split | 任务生成流程 | 任务样本清单、任务计数、family 覆盖清单 | 无 | P0 |
| P-02 | 验证目标空间映射可运行 | 正式任务体系 | `EC/LC/ICx`、`NOEC/LOEC`、`BCF/BAF` 三类目标空间 | 目标变换说明、反变换说明、边界条件清单 | P-01 | P0 |
| P-03 | 验证 split 协议可运行 | 正式任务体系 | `scaffold holdout`、`chemical_id holdout`、`medium transfer split` | split 清单、可复现实验索引 | P-01 | P0 |
| P-04 | 验证统一结果格式可运行 | 统一任务和 split | baseline/deep/transfer 结果表结构 | 统一结果表头、任务级摘要表头 | P-02, P-03 | P0 |

说明：

- `P-01` 到 `P-04` 不追求模型性能，只追求协议冻结和下游可消费性。

## 4. L2 正式主线对比

### 4.1 baseline 主线

| 实验 ID | 模型 | 数据范围 | split | 目标 | 交付物 | 前置依赖 | 优先级 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| B-01 | Ridge / ElasticNet | `water` | scaffold | baseline 起点 | 结果表、训练配置 | P-04 | P1 |
| B-02 | RandomForest | `water` | scaffold | baseline 起点 | 结果表、训练配置 | P-04 | P1 |
| B-03 | XGBoost | `water` | scaffold | baseline 起点 | 结果表、训练配置 | P-04 | P1 |
| B-04 | LightGBM | `water` | scaffold | baseline 起点 | 结果表、训练配置 | P-04 | P1 |
| B-05 | CatBoost | `water` | scaffold | baseline 起点 | 结果表、训练配置 | P-04 | P1 |
| B-06 | 最优 baseline 子集复跑 | `water + sediment` | scaffold | 扩展介质覆盖 | 结果表、对照摘要 | B-01~B-05 | P1 |
| B-07 | 最优 baseline 子集复跑 | `water + sediment + soil` | scaffold | 扩展介质覆盖 | 结果表、对照摘要 | B-06 | P1 |

说明：

- `B-06` 与 `B-07` 不要求把全部 baseline 模型无限扩展；建议先在 `water` 上筛出优先模型，再向更大介质范围复跑。

### 4.2 unified multitask deep 直接联合训练主线

| 实验 ID | 主线 | 数据范围 | split | 固定设置 | 交付物 | 前置依赖 | 优先级 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| D-01 | direct joint | `water` | scaffold | `RDKit grouped + Morgan + context + nonlinear level features` | 主模型结果、训练日志 | P-04 | P1 |
| D-02 | direct joint | `water + sediment` | scaffold | 同 D-01 | 主模型结果、训练日志 | D-01 | P1 |
| D-03 | direct joint | `water + sediment + soil` | scaffold | 同 D-01 | 主模型结果、训练日志 | D-02 | P1 |

### 4.3 迁移学习主线

| 实验 ID | 主线 | 预训练范围 | 微调范围 | 策略变量 | 交付物 | 前置依赖 | 优先级 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| T-01 | transfer | `water` | `soil` | 全量微调 | 结果表、训练日志 | P-04 | P1 |
| T-02 | transfer | `water` | `soil` | 冻结 `ChemicalEncoder` | 对照结果 | T-01 | P1 |
| T-03 | transfer | `water` | `soil` | 部分冻结 `ChemicalEncoder` | 对照结果 | T-01 | P1 |
| T-04 | transfer | `water + sediment` | `soil` | 全量微调 | 结果表、训练日志 | T-01 | P1 |
| T-05 | transfer | `water + sediment` | `soil` | 模块级差异学习率 | 对照结果 | T-04 | P1 |

说明：

- `T-01` 到 `T-05` 的比较目标是回答 transfer 是否优于 direct joint，以及哪种冻结策略更稳妥。

## 5. L3 关键消融矩阵

消融实验一次只改一个主轴，其余设置保持与 `D-01` 对齐，避免结论不可解释。

| 实验 ID | 消融轴 | 对照设置 | 变更设置 | 目标问题 | 前置依赖 | 优先级 |
| --- | --- | --- | --- | --- | --- | --- |
| A-01 | 化学输入 | `RDKit + Morgan` | `RDKit only` | Morgan 分支是否贡献主效应 | D-01 | P2 |
| A-02 | 化学输入 | `RDKit + Morgan` | `Morgan only` | 描述符分支是否贡献主效应 | D-01 | P2 |
| A-03 | 描述符融合 | grouped heads | flat fusion | 分组头是否优于平铺描述符 | D-01 | P2 |
| A-04 | 上下文 | context on | context off | 物种/介质/终点语义是否贡献泛化 | D-01 | P2 |
| A-05 | 终点编码 | structured endpoint | simple endpoint | 结构化终点语义是否优于简化编码 | D-01 | P2 |
| A-06 | level 编码 | nonlinear level features | raw level only | 非线性 level 特征是否有效 | D-01 | P2 |
| A-07 | 训练路线 | direct joint | transfer best setting | transfer 是否优于 direct joint | D-03, T-05 | P2 |

说明：

- `A-07` 是路线级消融，总结 direct joint 与 transfer 的最终差异。

## 6. L4 可靠性与应用矩阵

### 6.1 适用域 AD

| 实验 ID | 模块 | 方法 | 输入来源 | 交付物 | 前置依赖 | 优先级 |
| --- | --- | --- | --- | --- | --- | --- |
| R-01 | AD | Williams plot | 最优 baseline / deep 主结果 | 图、阈值说明、异常样本摘要 | B-06 或 D-02 | P2 |
| R-02 | AD | leverage | 同 R-01 | leverage 结果表 | R-01 | P2 |
| R-03 | AD | standardized residual | 同 R-01 | residual 结果表 | R-01 | P2 |

### 6.2 不确定度

| 实验 ID | 模块 | 方法 | 输入来源 | 交付物 | 前置依赖 | 优先级 |
| --- | --- | --- | --- | --- | --- | --- |
| U-01 | uncertainty | deep ensemble | 最优 deep 主线 | ensemble 结果表 | D-02 或 D-03 | P2 |
| U-02 | uncertainty | MC dropout | 最优 deep 主线 | dropout 结果表 | D-02 或 D-03 | P2 |
| U-03 | reliability 对照 | baseline vs deep | B 系列 + U 系列 | 对照汇总表 | U-01, U-02, B-06 | P2 |

### 6.3 SSD 工作流验证

| 实验 ID | 模块 | 输入家族 | 固定规则 | 交付物 | 前置依赖 | 优先级 |
| --- | --- | --- | --- | --- | --- | --- |
| S-01 | SSD | `EC / LC / ICx` | 可指定 `effect_level`；按历史任务级 `R2` 过滤物种 | SSD 输入样本表、过滤日志 | U-03 | P3 |
| S-02 | SSD | `BCF / BAF` | 不使用 `effect_level`；保留 `duration_h` | SSD 输入样本表、过滤日志 | U-03 | P3 |
| S-03 | SSD | `EC / LC / ICx` family 内部 | 拟合 SSD，输出 `HC5 / HC10 / uncertainty` | SSD 结果表与图 | S-01 | P3 |
| S-04 | SSD | `BCF / BAF` family 内部 | 拟合 SSD，输出 `HC5 / HC10 / uncertainty` | SSD 结果表与图 | S-02 | P3 |

说明：

- `S-01` 到 `S-04` 明确要求 family 内处理，不允许混入 `NOEC / LOEC`。

## 7. 建议执行顺序

### 第一轮：先保证能跑通

推荐执行：

`P-01 -> P-02 -> P-03 -> P-04 -> B-01~B-05 + D-01 + T-01`

目的：

- 验证协议、baseline、deep、transfer 三线都可运行。

### 第二轮：扩主线

推荐执行：

`B-06 -> B-07 -> D-02 -> D-03 -> T-02~T-05`

目的：

- 建立正式对照主线。

### 第三轮：做关键消融

推荐执行：

`A-01 ~ A-07`

目的：

- 回答结构设计是否合理。

### 第四轮：做可靠性与 SSD

推荐执行：

`R-01 ~ R-03 -> U-01 ~ U-03 -> S-01 ~ S-04`

目的：

- 闭合应用层与论文结果链路。

## 8. 每类实验的最低交付物

### baseline / deep / transfer

每个实验至少产出：

1. 实验 ID
2. 数据范围
3. 任务 family 覆盖说明
4. split 标识
5. 配置摘要
6. 统一结果表
7. 失败样本或异常日志

### ablation

每个实验至少产出：

1. 被改变的唯一主轴
2. 与对照组完全一致的其余设置说明
3. 对照结果与差值表

### AD / uncertainty

每个实验至少产出：

1. reliability 方法说明
2. 结果表
3. 可视化图件
4. 可供预测或 SSD 消费的结构化输出

### SSD

每个实验至少产出：

1. endpoint family 说明
2. species 过滤规则与阈值
3. 拟合输入样本数
4. `HC5 / HC10 / uncertainty`
5. family 隔离检查记录

## 9. 不建议的做法

1. 在任务定义未冻结时直接铺开大规模训练。
2. 在不同工作流中各自实现不同版本的 target 变换。
3. 同时改动“化学输入 + 上下文 + 终点编码 + 训练路线”后再把结果称为消融。
4. 在没有任务级 `R2` 摘要的前提下直接构建 SSD。
5. 把 `NOEC / LOEC` 为了凑样本量混入 `ECx` SSD。

## 10. 使用建议

1. 主代理可以把本文件作为派单索引，直接按实验 ID 分发。
2. 并行 worker 在提交结果时应引用实验 ID，避免口头描述不一致。
3. 当资源有限时，优先保证 `P`、`B`、`D`、`T` 系列，再做 `A`、`R`、`U`、`S` 系列。
