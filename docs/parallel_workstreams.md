# 并行工作流拆分

本文件用于指导多人或多代理并行推进本项目，重点定义：

- 每条工作流负责什么
- 它依赖什么
- 它向谁交付什么
- 哪些边界不能越过

权威来源仍是：

- [AGENTS.md](</D:/深度学习2.0/AGENTS.md>)
- [统一多任务残差QSAR架构说明.md](</D:/深度学习2.0/docs/统一多任务残差QSAR架构说明.md>)

## 1. 并行执行总原则

1. 共享契约优先于各自实现。
2. 所有正式实验必须共用同一数据入口、同一任务定义、同一 split 协议、同一目标定义逻辑、同一评估口径。
3. 不允许任何 worker 私自把 `effect_level` 改成任务 ID。
4. 不允许任何 worker 私自把 `BCF / BAF` 拉出统一主模型体系之外。
5. 不允许任何 worker 私自把 `NOEC / LOEC` 与 `EC / LC / ICx` 混成同一 SSD 或同一目标空间。
6. 不允许以旧 `target_ptox` 假设覆盖当前架构说明。
7. 若遇到文件冲突，基于当前最新文件内容兼容追加，不回退他人修改。

## 2. 推荐并行工作流

建议至少拆成 8 条工作流。它们不是 8 个孤岛，而是围绕共享契约协同。

### WS0：协议与集成治理

职责：

- 维护任务定义、终点语义、目标空间、split、评估输出格式的统一约束。
- 负责跨工作流集成检查。

输入：

- 权威文档
- 其他工作流的接口需求

输出：

- 统一字段定义
- 统一产物清单
- 集成检查清单

前置依赖：

- 无

并行边界：

- 可与所有工作流并行，但它的约束对其他工作流具有前置约束作用。

### WS1：数据语义与任务构造

职责：

- 基于 `ecotox_clean.sqlite` 与 `ecotox_toxicity_joined_curated` 冻结正式建模入口。
- 输出任务构造逻辑、终点语义拆解逻辑、目标空间映射逻辑。

输入：

- 权威文档
- 数据源主视图

输出：

- 任务表定义
- endpoint 结构化字段定义
- family 级过滤规则
- 训练/验证/测试 split 清单或生成协议

前置依赖：

- WS0

并行边界：

- 完成后才能让 WS2、WS3、WS4、WS5 进入正式实现。

不可越界事项：

- 不重建完整原始 join 链作为默认入口。
- 不重新定义任务粒度。

### WS2：评估协议与结果汇总

职责：

- 定义并固化 baseline、deep、transfer、AD、uncertainty、SSD 的统一结果结构。
- 负责主 split 与补充 split 的结果汇总口径。

输入：

- WS1 的任务和 split 契约

输出：

- 统一结果表头
- 历史任务级性能汇总格式
- SSD 物种筛选所需的任务级 `R2` 汇总格式

前置依赖：

- WS1

并行边界：

- 可与 WS3、WS4、WS5 并行推进。

不可越界事项：

- 不得让不同模型路线采用不同统计口径。

### WS3：baseline 工作流

职责：

- 基于统一数据与任务协议实现并运行 baseline：
  - `Ridge / ElasticNet`
  - `RandomForest`
  - `XGBoost`
  - `LightGBM`
  - `CatBoost`

输入：

- WS1 的任务数据
- WS2 的评估协议

输出：

- baseline 训练产物
- 主 split 与补充 split 结果
- 可用于 SSD species filtering 的任务级性能摘要

前置依赖：

- WS1
- WS2

并行边界：

- 可与 WS4、WS5 并行。

不可越界事项：

- 不得使用与 deep 不一致的任务或目标定义。

### WS4：统一多任务 deep 主模型工作流

职责：

- 实现统一主模型的 direct joint 训练路线。

强约束：

- 保留 `RDKit descriptors + Morgan fingerprint` 双分支。
- RDKit 描述符正式主方案采用分组头与可训练融合。
- context 分支必须编码 species、taxon、medium、endpoint semantics、duration。
- `EC / LC / ICx` 路线必须显式考虑 `level_fraction / logit / probit`。

输入：

- WS1 的任务数据
- WS2 的评估协议

输出：

- `water`
- `water + sediment`
- `water + sediment + soil`

三条 direct joint 主线结果与模型产物。

前置依赖：

- WS1
- WS2

并行边界：

- 可与 WS3、WS5 并行。

不可越界事项：

- 不得拆成多个相互独立的主模型替代统一主模型。

### WS5：迁移学习工作流

职责：

- 实现并比较迁移学习路线：
  - `pretrain water`
  - `pretrain water + sediment`
  - `finetune soil`

允许方案：

- 冻结 `ChemicalEncoder`
- 部分冻结 `ChemicalEncoder`
- 模块级差异学习率

输入：

- WS1 的任务数据
- WS2 的评估协议
- WS4 的统一主模型结构约束

输出：

- direct joint vs transfer 对照结果
- 不同冻结策略对照结果

前置依赖：

- WS1
- WS2

并行边界：

- 可与 WS3、WS4 并行，但必须共用同一主模型任务体系。

不可越界事项：

- 不得把 transfer 线实现成与 unified multitask 架构脱钩的另一套任务系统。

### WS6：AD 与 uncertainty 工作流

职责：

- 在稳定 baseline 与 deep 结果之上构建 reliability 体系。

内容边界：

- AD：`Williams plot`、leverage、standardized residual
- uncertainty：`deep ensemble`、`MC dropout`

输入：

- WS2 的结果协议
- WS3 或 WS4/WS5 的稳定模型产物

输出：

- reliability 指标与图
- 单点/批量预测可消费的 reliability 输出

前置依赖：

- WS2
- WS3 或 WS4/WS5 至少一条稳定主线

并行边界：

- AD 与 uncertainty 可以再细分给不同 worker。

不可越界事项：

- 不得脱离正式主线单独定义另一套评估样本。

### WS7：预测与 SSD 工作流

职责：

- 把模型结果接入单点预测、批量预测与 SSD。

强约束：

- `EC / LC / ICx` 可指定 `effect_level`
- `BCF / BAF` 不指定 `effect_level`
- SSD 前必须按历史任务级 `R2` 阈值筛选物种
- 严禁混合 `ECx`、`NOEC`、`LOEC`

输入：

- WS2 的任务级结果汇总
- WS6 的 reliability 输出

输出：

- 预测接口协议
- SSD 输入输出协议
- `HC5 / HC10 / uncertainty` 结果格式

前置依赖：

- WS2
- WS6

并行边界：

- 可在主模型稳定后独立推进，但高度依赖上游结果格式稳定。

### WS8：图表与论文产物工作流

职责：

- 统一图表风格与最终导出。

强约束：

- 中文：`黑体`
- 英文：`Arial`
- 不允许字体回退
- `PNG + SVG`
- `300 dpi+`

输入：

- WS2 的结果表
- WS6、WS7 的分析产物

输出：

- baseline 图
- ablation 图
- transfer 图
- AD / uncertainty 图
- SSD 图

前置依赖：

- WS2
- 对应结果工作流完成

并行边界：

- 样式库可先建，但正式出图应等待结果冻结。

## 3. 工作流之间的依赖图

```text
WS0
  -> WS1
    -> WS2
      -> WS3
      -> WS4
      -> WS5
      -> WS6
        -> WS7
      -> WS8
```

更准确地说：

1. `WS3 / WS4 / WS5` 依赖 `WS1 + WS2`。
2. `WS6` 依赖 `WS2 + 稳定模型产物`。
3. `WS7` 依赖 `WS2 + WS6`。
4. `WS8` 依赖统一结果口径，也依赖对应分析结果完成。

## 4. 共享契约清单

以下内容必须由所有工作流共享，不能各自维护一份隐式版本：

1. 正式数据入口
2. 任务 ID 规则
3. endpoint semantics 结构化字段
4. 目标空间映射逻辑
5. split 协议
6. 结果表头
7. 历史任务级性能汇总格式
8. SSD 物种筛选规则
9. 图表样式规范

## 5. 推荐交付节奏

### 第 1 批交付

- WS0：共享契约清单
- WS1：任务与数据协议
- WS2：统一结果格式

用途：

- 让所有后续工作流开始实现而不互相漂移。

### 第 2 批交付

- WS3：baseline 首轮可运行版本
- WS4：direct joint 首轮可运行版本
- WS5：transfer 首轮可运行版本

用途：

- 建立正式研究主线。

### 第 3 批交付

- WS6：AD / uncertainty 首轮结果
- WS7：预测与 SSD 工作流

用途：

- 闭合应用与可靠性链路。

### 第 4 批交付

- WS8：正式图表与论文交付物

## 6. 多 worker 协同时的检查点

每条工作流在提交前至少回答以下问题：

1. 我是否仍然使用 `task = species + endpoint semantics`？
2. 我是否仍然把 `effect_level` 作为条件输入而非任务 ID？
3. 我是否与其他模型路线共用同一 split 和目标定义？
4. 我是否错误地把 `BCF / BAF` 拉出了统一主模型体系？
5. 我是否会导致 `ECx`、`NOEC`、`LOEC` 在 SSD 中混用？
6. 我的输出能否被 WS2、WS6、WS7、WS8 直接消费？

若任一问题回答为“否”或“不确定”，应先回到共享契约层处理，而不是继续并行扩散。

## 7. 建议的任务领取方式

为了降低并发冲突，建议主代理按以下顺序派发任务：

1. 先派 `WS0 + WS1 + WS2`
2. 再并行派 `WS3 + WS4 + WS5`
3. 再派 `WS6 + WS7`
4. 最后派 `WS8`

如果人员足够，也可以这样拆：

1. 一个代理守共享契约与集成
2. 一个代理做 baseline
3. 一个代理做 deep direct joint
4. 一个代理做 transfer
5. 一个代理做 AD / uncertainty
6. 一个代理做 SSD / prediction
7. 一个代理做图表

这样拆的前提是：共享契约已经冻结，否则并行度越高，返工越多。
