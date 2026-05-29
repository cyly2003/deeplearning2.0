# 项目总体路线图

本文件用于把项目总体目标拆解为可执行阶段、关键依赖与阶段性交付物，供主代理和并行 worker 统一参照。

权威约束来源：

- [AGENTS.md](</D:/深度学习2.0/AGENTS.md>)
- [统一多任务残差QSAR架构说明.md](</D:/深度学习2.0/docs/统一多任务残差QSAR架构说明.md>)

若后续实现、旧脚本或旧实验记录与本文件冲突，以以上两份权威文档为准；本文件只负责执行拆解，不新增架构假设，不报告任何未完成实验结果。

## 1. 任务边界与冻结项

在进入并行开发前，以下事项视为全项目冻结约束，所有子任务必须遵守：

1. 正式任务定义固定为 `task = species + endpoint semantics`。
2. `effect_level` 是输入条件，不是任务 ID 的一部分。
3. `LCx` 语义并入 `ECx + mortality`。
4. `NR` 不进入正式建模。
5. `BCF / BAF` 纳入统一主模型，但不走 `effect_level` 路径，必须保留 `duration_h`。
6. 第一版采用一个统一多任务主模型，不默认拆分为多个彼此独立主模型。
7. 化学主分支必须保留 `RDKit descriptors + Morgan fingerprint` 双分支。
8. RDKit 描述符默认走“按组编码 + 组内/组间可训练融合”，不能把全部描述符直接平铺为唯一正式方案。
9. 必须并行保留两条研究线：直接联合训练线、迁移学习线。
10. baseline、ablation、AD、uncertainty、SSD 都是正式主线，不是附属内容。
11. 主评估场景优先级固定为：
    - 已知物种上的新化合物泛化
    - 跨介质迁移
12. 主 split 优先采用 `scaffold holdout`，可补充 `chemical_id holdout` 与 `medium transfer split`。
13. SSD 中禁止混用 `ECx`、`NOEC`、`LOEC`。

## 2. 总体阶段拆解

项目执行建议按 7 个阶段推进，但不是简单串行；部分阶段在前置契约冻结后可并行展开。

### 阶段 P0：架构与协议对齐

目标：

- 把任务定义、终点语义、目标空间、训练路线、评估口径写成统一执行契约。

核心输出：

- 冻结后的任务语义说明
- 冻结后的训练路线说明
- 冻结后的实验矩阵草案
- 多 worker 的共享边界说明

完成标志：

- 所有参与者明确哪些内容是“可以实现”，哪些内容是“不可擅自重定义”。

依赖：

- 只依赖当前两份权威文档。

并行性：

- 可与环境盘点、项目骨架设计同步开始，但优先级最高。

### 阶段 P1：数据语义与建模入口冻结

目标：

- 围绕 `ecotox_clean.sqlite` 与 `ecotox_toxicity_joined_curated` 固化正式数据入口。
- 冻结任务生成、终点语义拆解、目标空间映射、数据审计口径。

核心输出：

- 数据字段到模型字段的映射表
- 任务构造规则
- `EC / LC / ICx`、`NOEC / LOEC`、`BCF / BAF` 的目标空间定义
- 训练/验证/测试划分协议

完成标志：

- baseline、deep、transfer 共用同一任务体系与 split 协议。

依赖：

- P0。

并行边界：

- 一旦任务表结构和 split 契约冻结，P2 与 P3 可并行开展。

### 阶段 P2：实验框架底座

目标：

- 建立能复用的实验配置、运行、记录、汇总框架。

核心输出：

- baseline 实验入口
- deep 主模型实验入口
- transfer 实验入口
- ablation 配置矩阵
- 统一评估与日志记录骨架

完成标志：

- 不同模型路线可以用同一任务定义、同一 split、同一输出协议运行。

依赖：

- P1 冻结的数据与 split 契约。

并行边界：

- baseline、deep、transfer 三条线可以拆给不同 worker。

### 阶段 P3：正式建模主线

目标：

- 跑通 baseline、联合训练、迁移学习、关键消融。

核心输出：

- baseline 结果表
- unified multitask deep 结果表
- direct joint vs transfer 对照结果表
- 关键 ablation 结果表

完成标志：

- 至少能回答“统一主模型是否优于基线”“分组描述符头是否有价值”“上下文与 level 非线性是否贡献泛化”“transfer 是否优于 direct joint”这四类问题。

依赖：

- P2。

并行边界：

- baseline 与 deep 可并行。
- direct joint 与 transfer 可并行。
- 消融实验应在主模型可稳定收敛后分批并行，不建议在协议未冻结时盲目铺开。

### 阶段 P4：可靠性主线

目标：

- 在正式模型之上补齐 AD 与 uncertainty。

核心输出：

- `Williams plot`
- leverage / standardized residual 结果
- `deep ensemble` 结果
- `MC dropout` 结果
- baseline 与 deep 的 reliability 对照输出

完成标志：

- 单点预测与批量预测都能返回预测值之外的可靠性信息。

依赖：

- P3 中至少有一版稳定基线模型与一版稳定深度模型。

并行边界：

- AD 与 uncertainty 可拆成两个 worker，但要共用统一评估样本与输出协议。

### 阶段 P5：应用层工作流

目标：

- 把预测、批量预测、SSD 工作流接到正式模型产物上。

核心输出：

- 单点预测协议
- 批量预测协议
- SSD species filtering 协议
- SSD 拟合与 `HC5 / HC10 / uncertainty` 输出协议

完成标志：

- 对 `EC / LC / ICx` 与 `BCF / BAF` 两类输入都能按规则执行；不会把 `NOEC / LOEC` 错混进 SSD。

依赖：

- P3 的稳定模型结果
- P4 的 reliability 输出

并行边界：

- SSD 逻辑实现必须晚于任务家族边界冻结；否则极易产生错误混用。

### 阶段 P6：论文图表与总结交付

目标：

- 以统一样式库导出论文级图表与最终实验包。

核心输出：

- baseline vs deep 图
- ablation 图
- transfer 图
- AD / uncertainty 图
- SSD 图
- 汇总表格与方法说明

完成标志：

- 图表符合中文 `黑体`、英文 `Arial`、`PNG + SVG`、`300 dpi+` 等硬约束。

依赖：

- P3、P4、P5 的最终结果与协议。

并行边界：

- 图表系统可以提前搭骨架，但正式出图必须等待结果冻结。

## 3. 关键依赖关系

建议按以下顺序理解依赖，而不是把所有工作视为完全串行：

```text
P0 架构与协议对齐
  -> P1 数据语义与建模入口冻结
    -> P2 实验框架底座
      -> P3 正式建模主线
        -> P4 可靠性主线
        -> P5 应用层工作流
          -> P6 图表与总结交付
```

其中允许的前置并行关系：

1. `P0` 完成后，数据契约整理与实验框架设计可以交叉推进。
2. `P1` 完成后，baseline、deep、transfer 可并行。
3. `P3` 中 baseline 与 deep 可并行；direct joint 与 transfer 可并行。
4. `P4` 与 `P5` 可以部分并行，但前提是二者都读取同一版稳定模型输出。

## 4. 里程碑与交付物

### M1：协议冻结

必须交付：

- 正式任务定义
- 正式终点语义编码
- 正式目标空间说明
- 正式 split 协议

### M2：实验可运行

必须交付：

- baseline 可跑通
- deep 主模型可跑通
- transfer 主线可跑通
- 日志和结果汇总格式统一

### M3：核心研究问题可回答

必须交付：

- baseline vs deep
- direct joint vs transfer
- 关键 ablation
- 至少一版 scaffold holdout 主结果

### M4：可靠性与应用工作流闭环

必须交付：

- AD
- uncertainty
- 单点/批量预测协议
- SSD 协议与输出

### M5：论文交付包

必须交付：

- 统一风格图表
- 关键结果表
- 方法与实验设置说明

## 5. 风险点与对应策略

### 5.1 任务定义漂移

高风险表现：

- 把 `effect_level` 写进任务 ID
- 把 `LCx` 单独做成与 `ECx + mortality` 平行的新体系
- 把 `NOEC / LOEC` 和 `ECx` 混成同一目标空间

对应策略：

- 任务构造逻辑由单一权威实现或单一权威文档冻结。

### 5.2 数据入口漂移

高风险表现：

- 不经审计地重建原始 join 链
- baseline 与 deep 读入不同版本数据

对应策略：

- 所有正式实验默认以 `ecotox_clean.sqlite` 主视图为入口。

### 5.3 实验不可比

高风险表现：

- baseline、deep、transfer 使用不同 split 或不同目标定义
- 消融时同时改动多个轴，导致结论不可解释

对应策略：

- 统一实验协议先冻结，再开跑。
- 消融一次只改一个主轴，其他配置锁定。

### 5.4 SSD 误用

高风险表现：

- 混合不同终点家族做 SSD
- 使用无历史任务级 `R2` 支撑的物种直接入 SSD

对应策略：

- SSD 前置 species 过滤与 endpoint family 过滤必须写死在流程中。

## 6. 本路线图如何使用

1. 主代理用本文件判断当前阶段、前置依赖和是否允许并行。
2. 各 worker 在接任务时先标注自己属于哪个阶段、依赖哪个里程碑。
3. 若某 worker 需要修改会影响任务定义、目标空间、split、评估口径的共享文件，应先回到 P0/P1 契约层确认，而不是直接各自实现。
4. 若多个 worker 并行推进同一阶段，优先以“共享契约先冻结、实现后并行”为原则拆分。
