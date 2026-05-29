# AGENTS.md

本文件是本项目在 Codex / GPT 类编程助手中的项目级权威开发指令。  
新对话开始时，应优先阅读本文件，再阅读：

- [docs/统一多任务残差QSAR架构说明.md](</D:/深度学习2.0/docs/统一多任务残差QSAR架构说明.md>)

若本文件与旧脚本、旧实验目录、旧审计文档存在冲突，以本文件和上述架构说明文档为准。

---

## 1. 项目目标

本项目用于构建一个统一多任务的生态毒理 QSAR 与迁移学习系统。

该系统需要学习：

```text
化合物结构
+ 物种上下文
+ 介质
+ 暴露时间
+ 终点语义
+ 效应水平（若适用）
-> 条件化毒性/富集响应
```

并支持：

- baseline 对比
- 消融实验
- 联合训练
- 迁移学习
- 单点预测
- 批量预测
- SSD
- AD
- uncertainty
- 论文图表输出

---

## 2. 权威架构来源

本项目的正式架构定义不以旧图、旧脚本或旧 `target_ptox` 假设为准。

正式架构说明以：

- [docs/统一多任务残差QSAR架构说明.md](</D:/深度学习2.0/docs/统一多任务残差QSAR架构说明.md>)

为唯一权威说明来源。

任何新代码、新配置、新图表或新实验，都必须与该文档保持一致。

---

## 3. 数据入口

正式开发入口以整理版 SQLite 为主：

- `ecotox_clean.sqlite`
- 主视图：`ecotox_toxicity_joined_curated`

原始 SQLite 用于：

- 回源核对
- 异常追溯
- 字段补充

不要在新实现中默认重建完整原始 join 链，除非用户明确要求。

---

## 4. 任务定义

正式任务粒度为：

```text
task = species + endpoint semantics
```

例如：

- `species + EC_mortality`
- `species + EC_growth`
- `species + EC_reproduction`
- `species + NOEC_reproduction`
- `species + BCF_bioaccumulation`

重要约束：

1. `effect_level` 是输入条件，不是任务 ID 的一部分。
2. `LCx` 在语义上并入 `ECx + mortality`。
3. `NR` 不参与正式建模。
4. `BCF / BAF` 纳入统一主模型，但不走 `effect_level` 路径。

---

## 5. 终点语义编码

当前建议采用以下结构化字段：

- `effect_type = LC / EC / NOEC / LOEC / BCF / BAF / ICx`
- `effect_level = 10 / 20 / 50 / 90 / none`
- `endpoint_observation = mortality / growth / reproduction / biochemical / population / bioaccumulation / endocrine / physiology`
- `is_lethal`
- `is_chronic`
- `is_threshold_endpoint`
- `is_bioaccumulation`

不要把原始 `endpoint` 字符串直接当作唯一建模语义。

---

## 6. 模型总体结构

总体建模思想为：

```text
y_pred = y_chemical + alpha * Δy_context + beta * Δy_rule
```

解释：

- `y_chemical`：化学结构主效应，是 QSAR 主线
- `Δy_context`：物种、介质、暴露、终点语义、效应水平引起的条件偏移
- `Δy_rule`：可选规则纠偏接口，第一版保留接口，不要求主导结论

第一版采用：

- 一个统一的多任务主模型

不要默认拆成多个彼此独立的主模型。

---

## 7. 化学编码器硬约束

第一版化学输入采用：

- `RDKit descriptors`
- `Morgan fingerprint`

但必须注意：

### 7.1 描述符不能简单平铺

分子描述符应按计算原理或物理化学含义分组，例如：

- constitutional
- topological
- electronic
- physicochemical
- fragment / substructure related
- 3D / geometric（若后续启用）

### 7.2 分组头

推荐实现方式：

```text
descriptor groups
-> group-specific heads
-> trainable intra-group weighting
-> trainable inter-group fusion
-> chemical embedding
```

也就是说：

- 组内权重可训练
- 组间权重可训练

不要默认把所有描述符直接拼成一个平坦向量送入单一 MLP。

### 7.3 Morgan 指纹

Morgan 指纹是化学结构主效应的重要并行分支，不应被弱化为可有可无的附加项。

---

## 8. 上下文编码器硬约束

上下文分支应编码：

- `species_id`
- `taxon_group_l1 / l2 / l3`
- `genus`
- `family`
- `organism_lifestage`
- `primary_medium`
- `effect_type`
- `endpoint_observation`
- endpoint flags
- `duration_h`

对于 `EC / LC / ICx`：

- `effect_level` 应显式以非线性形式参与建模
- 至少考虑：
  - `level_fraction`
  - `logit(level_fraction)`
  - `probit(level_fraction)`

对于 `BCF / BAF`：

- 不使用 `effect_level`
- 但必须保留 `duration_h`

---

## 9. 目标空间策略

不要强制把所有任务统一到全局 `pTox`。

采用：

```text
任务内统一
训练空间与输出空间分离
```

即：

- 训练时可采用更稳定的模型空间
- 输出时恢复为用户可理解的原始量纲

典型要求：

- `EC / LC / ICx`：任务内统一量纲并允许稳定变换
- `NOEC / LOEC`：单独定义阈值型目标空间
- `BCF / BAF`：训练时可用更稳定的对数空间，输出恢复原始富集因子

---

## 10. 训练路线

必须并行保留两条研究线：

### 10.1 直接联合训练线

- `water`
- `water + sediment`
- `water + sediment + soil`

### 10.2 迁移学习线

- `pretrain water`
- `pretrain water + sediment`
- `finetune soil`

迁移学习允许：

- 冻结 `ChemicalEncoder`
- 部分冻结 `ChemicalEncoder`
- 模块级差异学习率

---

## 11. baseline 与消融实验

baseline 不是附属内容，而是正式研究主线。

至少保留：

- `Ridge / ElasticNet`
- `RandomForest`
- `XGBoost`
- `LightGBM`
- `CatBoost`

并要求与深度模型共用：

- 同一任务体系
- 同一 split 协议
- 同一目标定义逻辑
- 同一评估口径

必须规划消融实验矩阵，至少覆盖：

- `RDKit only / Morgan only / both`
- `descriptor flat fusion / descriptor grouped heads`
- `context on / off`
- `simple endpoint / structured endpoint`
- `raw level / nonlinear level features`
- `direct joint / transfer learning`

---

## 12. 评估重点

主评估场景优先：

1. 已知物种上的新化合物泛化
2. 跨介质迁移

主 split 优先：

- `scaffold holdout`

可补充：

- `chemical_id holdout`
- `medium transfer split`

---

## 13. AD 与 uncertainty

第一版 AD 采用：

- `Williams plot`
- `leverage`
- `standardized residual`

第一版 uncertainty 同时纳入：

- `deep ensemble`
- `MC dropout`

并要求 baseline 与 deep 尽量提供可比的 reliability 输出。

---

## 14. SSD

SSD 是正式功能，不是附属脚本。

工作流：

1. 用户输入化合物、介质、终点家族
2. 若为 `EC / LC / ICx`，可指定 `effect_level`
3. 若为 `BCF / BAF`，不指定 `effect_level`
4. 批量预测物种集合
5. 按用户指定的历史任务级 `R2` 阈值筛选物种
6. 拟合 SSD
7. 输出 `HC5 / HC10 / uncertainty`

禁止把：

- `ECx`
- `NOEC`
- `LOEC`

混入同一 SSD。

---

## 15. 图表硬约束

后续所有图都必须使用统一样式库。

### 15.1 字体

强约束如下：

- 中文固定：`黑体`
- 英文固定：`Arial`

并且：

- 不允许使用字体回退机制
- 必须中英分开显式设置
- 不允许依赖系统自动替换

### 15.2 输出

- 输出格式：`PNG + SVG`
- 分辨率：`300 dpi+`
- 先定义配色卡
- 避免大块浓暗色
- 除数据点数字标签外，其余文字均加粗
- 标签不得重叠

风格参考：

- *Environmental Science & Technology*
- *Journal of Hazardous Materials*

---

## 16. Codex 工作方式要求

新对话继续本项目时，默认应：

1. 先阅读本 `AGENTS.md`
2. 再阅读 [docs/统一多任务残差QSAR架构说明.md](</D:/深度学习2.0/docs/统一多任务残差QSAR架构说明.md>)
3. 若需要画图、搭环境、写配置、写代码、做实验，均以这两份文件为准

每次修改代码前，应说明：

- 准备修改哪些文件
- 每个文件修改目的
- 是否影响已有接口
- 是否需要新增测试

每次修改代码后，应输出：

- 修改文件列表
- 新增函数列表
- 运行测试命令
- 下一步建议

---

## 17. 当前阶段执行优先级

当前优先级：

1. 需求与架构对齐
2. 环境与依赖清单
3. 项目骨架与配置规范
4. baseline / deep / transfer / ablation 实验框架
5. 预测、AD、uncertainty、SSD、图表系统

当前不优先：

- 复杂 Web 前端
- 复杂 GUI
- 过早引入 GNN
- 无审计地大规模删数据

