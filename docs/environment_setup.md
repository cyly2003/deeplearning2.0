# 环境与依赖方案

本文档定义本项目当前推荐的环境与依赖配置方案，不代表当前机器已经全部安装完成。

本项目的环境、代码、实验与图表规范，仍以以下权威文件为准：

- [AGENTS.md](/D:/深度学习2.0/AGENTS.md)
- [统一多任务残差QSAR架构说明.md](/D:/深度学习2.0/docs/统一多任务残差QSAR架构说明.md)

## 1. 任务理解

本项目不是单一 QSAR 脚本，而是后续要承载以下能力的统一研究框架：

- SQLite 数据读取与审计
- RDKit descriptors + Morgan fingerprint 化学表示
- baseline 模型对比
- 多任务深度学习与迁移学习
- AD / uncertainty / SSD
- 论文风格静态图输出
- 测试与工程化开发

因此环境方案不能只满足“能开一个 notebook”，而要同时支持：

- Windows 上的可落地安装
- RDKit 的稳定可用
- baseline 与 deep 共存
- GPU 训练
- 后续实验框架扩展
- 开发依赖分层

## 2. 当前机器的 GPU 前提

当前主代理已在本机验证到以下事实：

- GPU：`NVIDIA GeForce RTX 4060 Ti`
- Driver Version：`576.02`
- `nvidia-smi` 报告 CUDA Version：`12.9`

因此本项目主环境已明确切换为 `GPU-only`。

## 3. 推荐工作流程

推荐采用四层结构：

1. `environment.yml` 作为 Conda 主环境方案  
   负责 Python、RDKit、科学栈、baseline 与开发工具。
2. `requirements-gpu.txt` 负责 PyTorch 官方 `cu124` GPU wheels
3. `pyproject.toml` 作为项目元数据、依赖分层与开发工具配置入口
4. `requirements-dev.txt` 只承载额外开发工具

## 4. 依赖分层设计

### 4.1 核心运行层

适用范围：

- 数据读写
- 特征工程
- 通用统计分析
- 通用评估

代表依赖：

- `numpy`
- `pandas`
- `scipy`
- `scikit-learn`
- `statsmodels`
- `sqlalchemy`
- `pyarrow`
- `PyYAML`

### 4.2 化学建模层

适用范围：

- SMILES 解析
- RDKit descriptors
- Morgan fingerprint

代表依赖：

- `rdkit`

说明：

- Windows 下优先通过 Conda 安装 RDKit
- 不建议把 pip 作为 RDKit 主安装路径

### 4.3 baseline 层

适用范围：

- Ridge / ElasticNet
- RandomForest
- XGBoost
- LightGBM
- CatBoost

### 4.4 deep / transfer 层

适用范围：

- 统一多任务主模型
- 迁移学习
- deep ensemble
- MC dropout

代表依赖：

- `torch`
- `torchvision`
- `torchaudio`
- `torchmetrics`
- `tensorboard`
- `optuna`

当前策略：

- 正式主环境使用 `PyTorch GPU`
- `torch / torchvision / torchaudio` 通过 PyTorch 官方 `cu124` wheel 源安装
- 不再保留 CPU 作为正式训练主方案

### 4.5 可视化与解释层

适用范围：

- 论文图
- 误差分析
- 特征解释

代表依赖：

- `matplotlib`
- `seaborn`
- `shap`
- `umap-learn`

### 4.6 开发工具层

适用范围：

- 测试
- 格式化
- 静态检查
- 预提交钩子

代表依赖：

- `pytest`
- `pytest-cov`
- `ruff`
- `black`
- `mypy`
- `pre-commit`

## 5. 文件角色说明

### 5.1 `environment.yml`

定位：

- 作为 Windows / Conda / NVIDIA GPU 主环境定义文件
- 只负责 Python、RDKit、科学栈、baseline、可视化与开发工具

### 5.2 `requirements-gpu.txt`

定位：

- 只安装 PyTorch 官方 GPU wheels
- 避免让 Conda 同时承担 RDKit 科学栈和 PyTorch CUDA 运行时求解

当前锁定：

- `torch==2.5.1`
- `torchvision==0.20.1`
- `torchaudio==2.5.1`
- `--index-url https://download.pytorch.org/whl/cu124`

### 5.3 `pyproject.toml`

定位：

- 记录项目元数据
- 提供可选依赖分组
- 统一 `black` / `ruff` / `pytest` 基础配置
- 标注本项目 deep 运行时默认是 GPU-only

### 5.4 `requirements-dev.txt`

定位：

- 只安装额外开发工具
- 不重复承担主运行时环境职责

## 6. 推荐安装步骤

### 6.1 安装 Conda 发行版

推荐优先使用：

- Miniforge
- Mambaforge

### 6.2 设置 channel priority

建议在 PowerShell 中执行：

```powershell
conda config --set channel_priority strict
```

### 6.3 创建主环境

在项目根目录执行：

```powershell
conda env create -f environment.yml
conda activate ecotox-qsar
```

### 6.4 安装 PyTorch GPU 运行时

```powershell
pip install -r requirements-gpu.txt
```

### 6.5 补装开发工具

如需单独补装或更新开发工具，可执行：

```powershell
pip install -r requirements-dev.txt
```

## 7. GPU 路线说明

当前文档采用的 GPU 主路线是：

- Python：`3.11`
- PyTorch：`2.5.1`
- Wheel 通道：`cu124`

说明：

- 这里锁的是 PyTorch 官方 GPU wheel 运行时，不要求你单独安装完全同号的系统 CUDA toolkit
- 关键前提是 NVIDIA 驱动足够新
- 本机当前驱动满足这一前提

## 8. 本机验证结果与边界

### 8.1 本次已完成

本次已完成：

- 起草并写入环境文件
- 对依赖分层做文档化说明
- 对 GPU 主方案与开发工具职责做约束说明
- 用 `nvidia-smi` 确认本机 GPU / 驱动前提
- 用 `pytest tests -q` 验证项目骨架测试通过

### 8.2 本次未完成

本次没有宣称完成以下事项：

- 尚未完成最终 GPU 环境创建
- 尚未验证 RDKit / PyTorch / LightGBM 在新环境内成功导入
- 尚未验证系统字体是否满足“黑体 + Arial 且无回退”要求

## 9. 建议的本机验证命令

### 9.1 Python 与核心库

```powershell
python -c "import sys, numpy, pandas, scipy, sklearn; print(sys.version)"
```

### 9.2 RDKit 与 Morgan 指纹

```powershell
python -c "from rdkit import Chem; from rdkit.Chem import AllChem, Descriptors; mol = Chem.MolFromSmiles('CCO'); fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048); print(mol is not None, fp.GetNumBits(), Descriptors.MolWt(mol))"
```

### 9.3 baseline 框架

```powershell
python -c "import xgboost, lightgbm, catboost; print(xgboost.__version__, lightgbm.__version__, catboost.__version__)"
```

### 9.4 PyTorch + GPU

```powershell
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

### 9.5 图形与字体

```powershell
python -c "import matplotlib.font_manager as fm; names = {f.name for f in fm.fontManager.ttflist}; print('Arial' in names, 'SimHei' in names or '黑体' in names)"
```

## 10. 注意事项

### 10.1 RDKit 安装来源

本项目中 RDKit 是化学建模主线依赖，不应把其稳定性寄托在临时 pip 轮子可用性上。

推荐优先级：

1. `conda-forge`
2. 其他方式仅作补充

### 10.2 SQLite 使用方式

当前项目正式开发入口是：

- `ecotox_clean.sqlite`
- 主视图 `ecotox_toxicity_joined_curated`

### 10.3 图表字体不是 Python 包问题

即使 `matplotlib` 安装成功，也不等于满足项目级字体约束。

### 10.4 不确定性与 AD 依赖

第一版 AD / uncertainty 不强依赖额外重型外部框架：

- Williams plot
- leverage
- standardized residual
- deep ensemble
- MC dropout

## 11. 后续优化建议

1. 增加 `environment-lock.yml` 或平台锁文件  
   用于固定跨机器复现实验环境。
2. 增加严格锁定的 `environment.gpu.lock.yml`  
   用于锁定未来实验机群上的 GPU 运行时组合。
3. 增加图表字体检查脚本  
   在生成论文图前自动失败，而不是依赖人工肉眼发现字体回退。
