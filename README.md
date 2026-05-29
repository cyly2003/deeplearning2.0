# deeplearning2.0

统一多任务生态毒理 QSAR 与迁移学习研究框架。

## 项目目标

本项目围绕以下条件化响应建模任务展开：

```text
化合物结构
+ 物种上下文
+ 介质
+ 暴露时间
+ 终点语义
+ 效应水平（若适用）
-> 条件化毒性/富集响应
```

当前代码框架以以下权威文档为准：

- [AGENTS.md](/D:/深度学习2.0/AGENTS.md)
- [统一多任务残差QSAR架构说明](/D:/深度学习2.0/docs/统一多任务残差QSAR架构说明.md)

## 当前阶段范围

第一阶段重点是：

1. 环境与依赖清单
2. 项目骨架与配置规范
3. baseline / deep / transfer / ablation 实验框架
4. 预测、AD、uncertainty、SSD、图表系统的统一接口

## 数据入口

- SQLite: `ecotox_clean.sqlite`
- 主视图: `ecotox_toxicity_joined_curated`

默认不重建原始 join 链，除非任务明确要求。

## 目录概览

```text
configs/                 分层配置
docs/                    权威说明与执行文档
scripts/                 命令行脚本入口
src/deeplearning2/       主包
tests/                   框架与配置测试
artifacts/               结果输出目录
```

## 快速开始

环境准备与开发流程见：

- [环境配置说明](/D:/深度学习2.0/docs/environment_setup.md)
- [开发工作流](/D:/深度学习2.0/docs/developer_workflow.md)
- [项目路线图](/D:/深度学习2.0/docs/project_roadmap.md)

GPU 训练运行时安装清单见：

- [requirements-gpu.txt](/D:/深度学习2.0/requirements-gpu.txt)

## 当前实现说明

当前仓库已完成：

- 统一目录骨架
- 配置分层模板
- baseline / deep / transfer / evaluation / SSD 接口占位
- 最小测试与开发流程文档

后续将在此基础上继续补齐：

- SQLite 数据审计与 schema 绑定
- RDKit descriptors + Morgan 特征流水线
- scaffold holdout 与迁移学习训练逻辑
- AD / uncertainty / SSD 可运行实现
