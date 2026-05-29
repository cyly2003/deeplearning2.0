# 统一多任务残差 QSAR 架构说明

本说明文档基于当前对话中已经明确对齐的需求、研究目标、建模边界、评估方式与图表规范编写，不以某一张示意图为依据倒推解释。

本文件与项目根目录下的 [AGENTS.md](</D:/深度学习2.0/AGENTS.md>) 共同构成当前项目的架构与开发权威来源：

- `AGENTS.md` 负责面向新对话固化开发约束、流程和优先级
- 本文档负责详细阐明模型架构、任务定义、实验逻辑与图表规范

若后续新对话继续本项目开发，应先读取 `AGENTS.md`，再读取本文件。

对应图示资产当前包括：

- [unified_multitask_residual_qsar_architecture.svg](</D:/深度学习2.0/unified_multitask_residual_qsar_architecture.svg>)

后续若生成新的机理图、论文图或矢量版图示，均应以本说明文档中的架构定义为准。

## 1. 项目定位

本项目不是普通的单任务 QSAR，也不是只对单一量纲的水相毒性做统一回归。

本项目要构建的是一个统一的、多任务的、条件化生态毒理响应预测系统，目标是学习如下映射：

```text
化合物结构
+ 物种上下文
+ 介质
+ 暴露时间
+ 终点语义
+ 效应水平（若适用）
-> 条件化毒性/富集响应
```

系统第一版必须同时支持：

- 多终点家族建模
- baseline 对比
- 消融实验
- 迁移学习
- 适用域
- 不确定度
- 单点预测
- 批量预测
- SSD
- 论文风格图表输出

## 2. 数据入口与任务边界

### 2.1 正式数据入口

当前正式开发的数据入口以整理版 SQLite 为主，即：

- `ecotox_clean.sqlite`
- 主视图：`ecotox_toxicity_joined_curated`

原始 SQLite 的角色是：

- 回源核对
- 异常追溯
- 字段补充

而不是作为第一开发入口重复重建全部 join 逻辑。

### 2.2 任务定义

正式任务粒度不是：

- `species + EC50`
- `species + LC50`
- `species + EC_mortality_50`

而是：

```text
task = species + endpoint semantics
```

例如：

- `species + EC_mortality`
- `species + EC_growth`
- `species + EC_reproduction`
- `species + NOEC_reproduction`
- `species + BCF_bioaccumulation`

其中：

- `effect_level` 是输入条件
- `effect_level` 不进入任务 ID

这样做的原因是：

1. 原始数据中存在 `LC10 / LC50 / LC90` 等多种水平
2. 如果把水平写入任务，会导致任务过碎
3. 这会削弱模型学习浓度-效应连续关系的能力

## 3. 终点语义编码

当前已对齐的终点拆解逻辑如下：

- `effect_type = LC / EC / NOEC / LOEC / BCF / BAF / ICx`
- `effect_level = 10 / 20 / 50 / 90 / none`
- `endpoint_observation = mortality / growth / reproduction / biochemical / population / bioaccumulation / endocrine / physiology`
- `is_lethal = 0 / 1`
- `is_chronic = 0 / 1`
- `is_threshold_endpoint = 0 / 1`
- `is_bioaccumulation = 0 / 1`

特别约定：

1. `LCx` 语义上并入 `ECx + mortality`
2. `NOEC / LOEC` 作为阈值型终点，不与 `ECx` 点估计终点混为一类
3. `BCF / BAF` 纳入统一主模型，但不走 `effect_level` 路线
4. `NR` 不参与正式建模

## 4. 总体建模思想

模型总体采用“结构主效应 + 条件残差 + 可选规则纠偏”的残差型 QSAR 思想。

可表达为：

```text
y_pred = y_chemical + alpha * Δy_context + beta * Δy_rule
```

其中：

### 4.1 化学主效应 `y_chemical`

表示由化合物结构本身决定的基础毒性/富集信号。

这是项目的核心 QSAR 主线。  
无论后续加入多少上下文、迁移学习、SSD 或应用功能，化学结构主效应都应当是模型最核心、最可解释的部分。

### 4.2 条件残差 `Δy_context`

表示由于以下因素引起的系统偏移：

- 物种差异
- 介质差异
- 暴露时间差异
- 终点语义差异
- 效应水平差异

它不是用来替代化学结构，而是用来描述：

“同一个化学物在不同生物学/实验场景下为什么会偏离其基础毒性主效应”。

### 4.3 规则纠偏 `Δy_rule`

规则层在第一版保留接口，用于后续纳入显式机制修正，例如：

- 溶解度
- 生物有效性
- 暴露可达性
- 疏水性
- 富集相关经验规则

第一版不要求其成为正式主结论来源，但要为后续机制增强预留结构位置。

## 5. 化学编码器设计

### 5.1 第一版化学输入

第一版明确采用：

- `RDKit descriptors`
- `Morgan fingerprint`

暂不引入 GNN，但代码结构要保留后续接口。

### 5.2 描述符不是一锅煮

这里需要特别写清楚你刚强调的点：

**分子描述符不应仅被视作一个平坦向量直接送入同一个 MLP。**

而应根据其计算原理或物理化学含义进行分组，再对各组建立分组头。

例如可以按如下原则分组：

- constitutional
- topological
- electronic
- physicochemical
- fragment / substructure related
- 3D / geometric（若后续启用）

### 5.3 描述符分组头

更符合本项目设计意图的做法是：

1. 每个描述符组单独经过一个小型编码头
2. 组内特征先在本组内学习表示
3. 组间再通过可训练权重进行融合

也就是说：

- 组内权重可训练
- 组间权重也可训练

这种设计的意义在于：

1. 不同描述符组的统计性质不同
2. 不同描述符组承载的结构信息层次不同
3. 某些终点家族可能更依赖特定组
4. 这比把所有描述符完全平权拼接后直接送入 MLP 更有解释性

因此，化学编码器的更合理组织不是：

```text
all descriptors -> one MLP
```

而更接近：

```text
descriptor groups -> group-specific heads -> trainable group fusion
```

再与 Morgan 指纹分支融合，形成统一 `chemical embedding`。

### 5.4 Morgan 指纹的角色

Morgan 指纹仍然非常重要，因为它负责表达：

- 局部结构模式
- 官能团环境
- 子结构相似性

在本项目中，Morgan 指纹不是描述符的附庸，而是化学结构主效应的重要并行分支。

## 6. 上下文编码器设计

上下文编码器用于解释条件化偏移，而不是代替结构学习。

### 6.1 物种/介质分支

应编码以下信息：

- `species_id`
- `taxon_group_l1 / l2 / l3`
- `genus`
- `family`
- `organism_lifestage`
- `primary_medium`
- 物种栖息地或 curated 类群

这里更推荐 embedding ID 方案，而不是只做稀疏 one-hot。

### 6.2 终点语义分支

编码：

- `effect_type`
- `endpoint_observation`
- 各类 endpoint flags

这一步的目标是让模型学到：

- 死亡、生长、繁殖、生化终点之间的语义差异
- 阈值型终点与浓度-效应型终点之间的差异
- 富集类终点与毒性类终点之间的差异

### 6.3 暴露与效应水平分支

编码：

- `duration_h`
- `log(1 + duration_h)`

对于 `EC / LC / ICx` 类终点，`effect_level` 应以显式非线性特征参与建模，例如：

- `level_fraction`
- `logit(level_fraction)`
- `probit(level_fraction)`

这里并不是先验断言原始实验一定用 `logit` 或 `probit`，而是利用这类变换作为合理的先验特征，让模型自己学习其贡献。

对于 `BCF / BAF`：

- 不使用 `effect_level`
- 暴露时间必须参与

## 7. 统一主模型，而非多个小模型

第一版架构明确采用：

- 一个统一的多任务主模型

而不是：

- 每个 endpoint family 一个完全独立模型
- 每个小任务一个单独训练进程

这样设计的理由是：

1. 很多任务单独样本量不足
2. 独立小模型不利于学习共享的结构毒理规律
3. 统一主模型更适合迁移学习
4. 统一主模型更便于后续做 baseline 对比、消融实验、AD、uncertainty 和 SSD 的统一评估

## 8. 目标空间策略

### 8.1 不采用全局统一 pTox

当前我们已经明确：

- 不希望把所有任务强行统一为全局 `pTox`
- 因为这会导致大量不能可靠摩尔化的记录失去训练价值

因此采用：

```text
任务内统一
训练空间与输出空间分离
```

### 8.2 各终点家族的目标空间

#### EC / LC / ICx

- 在各任务内部统一量纲
- 训练时可用更稳定的目标变换
- 输出时再映射回用户易理解量纲

#### NOEC / LOEC

- 作为阈值型终点单独定义目标空间
- 不与 `ECx` 点估计终点混训为同一响应机制

#### BCF / BAF

- 训练时允许采用更稳定的对数空间
- 输出恢复到原始富集因子

## 9. 训练主线

### 9.1 直接联合训练线

需要保留三种数据范围：

- `water`
- `water + sediment`
- `water + sediment + soil`

### 9.2 迁移学习线

需要保留：

- `pretrain water`
- `pretrain water + sediment`
- `finetune soil`

迁移学习线允许：

- 冻结 `ChemicalEncoder`
- 部分冻结 `ChemicalEncoder`
- 模块级差异学习率

但主体结构仍应保持统一，便于与直接联合训练线比较。

## 10. baseline 与深度模型的可比性

baseline 不是附属实验，而是正式研究主线的一部分。

当前基线模型至少包括：

- `Ridge / ElasticNet`
- `RandomForest`
- `XGBoost`
- `LightGBM`
- `CatBoost`

并要求：

- 与深度模型使用同一任务体系
- 使用同一 split 协议
- 使用同一目标定义逻辑
- 使用同一评价指标体系

只有这样，baseline 与 deep 的性能比较才有研究意义。

## 11. 消融实验矩阵

消融实验不是补充，而是正式实验方案的一部分。

当前应至少考虑以下轴：

- `RDKit only / Morgan only / RDKit + Morgan`
- `descriptor flat fusion / descriptor grouped heads`
- `context on / off`
- `simple endpoint / structured endpoint`
- `raw level / nonlinear level features`
- `direct joint / transfer learning`

其核心目标是回答：

1. 化学结构主效应中，哪些结构表征最有价值
2. 分组描述符头是否优于平坦拼接
3. 物种/介质/暴露上下文是否真的贡献泛化
4. 效应水平显式非线性编码是否有效
5. 迁移学习是否优于直接联合训练

## 12. 评估重点

当前主评估场景优先是：

1. 已知物种上的新化合物泛化
2. 跨介质迁移

正式主 split 优先尝试：

- `scaffold holdout`

并可配合：

- `chemical_id holdout`
- `medium transfer split`

## 13. AD 与 uncertainty

### 13.1 适用域

第一版 AD 以经典化学空间方案为主：

- `Williams plot`
- `leverage`
- `standardized residual`

即优先完成标准 QSAR 的适用域实现。

### 13.2 不确定度

第一版 uncertainty 采用实验对比方式：

- `deep ensemble`
- `MC dropout`

同时要求 baseline 与 deep 尽量都有对应的可比输出。

## 14. 预测与 SSD

### 14.1 单点预测

用户输入：

- 化合物
- 介质
- 物种
- 暴露时间
- `effect_type`
- `endpoint_observation`
- 若适用则输入 `effect_level`

输出：

- 预测值
- 原始量纲结果
- AD 状态
- uncertainty

### 14.2 SSD

SSD 是正式应用层功能。

工作流为：

1. 用户输入化合物、介质、终点家族
2. 若为 `EC / LC / ICx`，可指定 `effect_level`
3. 若为 `BCF / BAF`，不指定 `effect_level`
4. 批量预测物种集合
5. 按用户指定的历史任务级 `R2` 阈值筛物种
6. 拟合 SSD
7. 输出 `HC5 / HC10 / uncertainty`

必须避免把：

- `ECx`
- `NOEC`
- `LOEC`

混入同一 SSD。

## 15. 图表规范

后续所有图都必须走统一样式库。

### 15.1 字体规则

这是强约束：

- 中文固定：`黑体`
- 英文固定：`Arial`

并且：

- 不使用字体回退机制
- 必须中英分开显式设置
- 不允许依赖系统自动替换

### 15.2 输出规则

- 输出格式：`PNG + SVG`
- 分辨率：`300 dpi+`
- 配色先定义色卡
- 避免大块浓暗色
- 除数据点数字标签外，其余文字均加粗
- 标签不得重叠

### 15.3 风格目标

整体图件风格参考：

- *Environmental Science & Technology*
- *Journal of Hazardous Materials*

## 16. 当前结论

本项目当前已经明确的架构结论是：

1. 采用统一多任务主模型
2. 化学结构主效应是主线
3. 描述符应按计算原理分组建头，组内与组间权重可训练
4. 物种、介质、暴露、终点语义用于解释条件偏移
5. `effect_level` 是条件输入，不是任务 ID
6. `BCF / BAF` 纳入统一主模型，但不走 `logit / probit` 水平编码
7. baseline、消融、迁移学习、AD、uncertainty、SSD 都属于第一版正式范围
8. 所有图表必须使用中英分设字体规则，不允许依赖回退机制

这份说明文档应作为后续：

- 重新绘制机理图
- 生成正式论文图
- 搭建项目目录
- 设计实验矩阵
- 编写配置文件

的统一依据。
