# GPU 环境验证报告

本报告只记录在当前机器上已经实际验证到的事实，并明确区分“已验证项”和“剩余问题”。

## 1. 验证范围

- 项目目录：`D:\深度学习2.0`
- GPU-only 原则：仅验证 GPU 训练运行时，不回退 CPU 方案
- Conda 环境路径：`D:\.conda\envs\ecotox-qsar`
- 验证脚本：`scripts/verify_gpu_environment.py`

## 2. 当前机器

- GPU: `NVIDIA GeForce RTX 4060 Ti`
- Driver Version: `576.02`
- `nvidia-smi` reported CUDA Version: `12.9`

说明：

- 驱动报告的 `CUDA 12.9` 是主机驱动可支持的最高 CUDA 版本。
- 当前项目运行时使用的是 PyTorch `cu124` wheel，对应用户态 CUDA runtime `12.4`，这一组合已验证可工作。

## 3. 已验证项

### 3.1 Python 与环境入口

已验证：

- `D:\.conda\envs\ecotox-qsar\python.exe` 存在且可启动
- `python --version` 返回 `Python 3.11.15`

### 3.2 核心科学栈可导入

执行命令：

```powershell
& D:\.conda\envs\ecotox-qsar\python.exe -c "import numpy, pandas, scipy, sklearn, rdkit; print('core_ok')"
```

结果：

- 返回 `core_ok`

说明：

- `numpy / pandas / scipy / scikit-learn / rdkit` 已可在当前环境中正常导入
- `environment.yml` 对应的 Conda 主环境主体可用

### 3.3 PyTorch GPU 运行时可用

执行命令：

```powershell
& D:\.conda\envs\ecotox-qsar\python.exe -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO_GPU')"
```

结果：

- `2.5.1+cu124`
- `12.4`
- `True`
- `NVIDIA GeForce RTX 4060 Ti`

说明：

- 当前 GPU-only 路线已经建立
- `torch.cuda.is_available() == True`
- GPU 设备可被当前 PyTorch 运行时直接识别

### 3.4 GPU 依赖版本状态

执行命令：

```powershell
& D:\.conda\envs\ecotox-qsar\python.exe -m pip show pillow torchvision torch
```

结果摘要：

- `torch==2.5.1+cu124`
- `torchvision==0.20.1+cu124`
- `pillow==12.2.0`

说明：

- GPU 运行时仍然保持 `cu124` 路线
- 没有混回 CPU-only wheel

### 3.5 Pillow 本体可用

执行命令：

```powershell
& D:\.conda\envs\ecotox-qsar\python.exe -c "from PIL import Image; print('pil_ok', Image.__version__)"
```

结果：

- `pil_ok 12.2.0`

补充验证：

```powershell
& D:\.conda\envs\ecotox-qsar\python.exe -c "from PIL import features; print(features.pilinfo())"
```

结果摘要：

- `PIL CORE support ok`
- `FREETYPE2 support ok`
- `LITTLECMS2 support ok`
- `JPEG / OPENJPEG / ZLIB / LIBTIFF / WEBP support ok`

说明：

- Pillow 自身及其主要图像编解码依赖并未整体损坏
- 问题不是“Pillow 完全不可用”

## 4. 问题排查与当前状态

### 4.1 先前已复现的问题

失败复现命令：

```powershell
& D:\.conda\envs\ecotox-qsar\python.exe -c "import torchvision; print('torchvision_ok', torchvision.__version__)"
```

失败现象：

- 在 `torchvision -> PIL.Image -> PIL._imaging` 处报错
- 典型错误：

```text
ImportError: DLL load failed while importing _imaging
```

### 4.2 已验证的关键诊断结论

执行命令：

```powershell
& D:\.conda\envs\ecotox-qsar\python.exe -c "from PIL import Image; print('pil_first', Image.__version__); import torch; print('torch_ok', torch.__version__); import torchvision; print('tv_ok', torchvision.__version__)"
```

结果：

- `pil_first 12.2.0`
- `torch_ok 2.5.1+cu124`
- `tv_ok 0.20.1+cu124`

说明：

- `torchvision` 不是绝对不可用
- 先前问题更接近 Windows 下的 DLL 加载顺序或底层依赖初始化时序问题
- 具体表现为：`torchvision` 冷启动直接导入可能失败，但在预先加载 `PIL.Image` 的同一 Python 进程中可以成功导入

### 4.3 本次修复动作

为避免内网镜像污染，已显式绕过用户级 `pip.ini` 中的内网源配置，执行：

```powershell
& D:\.conda\envs\ecotox-qsar\python.exe -m pip install --force-reinstall --no-cache-dir --index-url https://pypi.org/simple pillow==12.2.0
```

结果：

- 同版本 `Pillow 12.2.0` 已从官方 PyPI 重新安装
- 未改动 `torch==2.5.1+cu124`、`torchvision==0.20.1+cu124`、`torchaudio==2.5.1+cu124`
- GPU-only 路线保持不变

### 4.4 修复后复测结果

修复后执行了多次冷启动直接导入复测：

```powershell
1..5 | ForEach-Object { & D:\.conda\envs\ecotox-qsar\python.exe -c "import torchvision; print('ok', __import__('torchvision').__version__)" }
```

结果：

- 5 次均返回 `ok 0.20.1+cu124`

修复后执行了多次验证脚本复测：

```powershell
1..3 | ForEach-Object { & D:\.conda\envs\ecotox-qsar\python.exe scripts\verify_gpu_environment.py }
```

结果：

- 3 次均显示 `torchvision.direct_import.ok == true`
- 未再出现 `PIL._imaging` DLL load failed

## 5. 当前结论

### 5.1 已经收口的部分

- GPU-only 主路线已验证成立
- 核心科学栈可用
- `torch 2.5.1+cu124` 可正常使用 GPU
- 设备 `RTX 4060 Ti` 可被 PyTorch 识别

### 5.2 当前剩余风险

- 当前这轮修复后，`torchvision` 直接导入问题已暂时消失
- 但该问题在修复前已被真实复现过，因此仍建议保留 `scripts/verify_gpu_environment.py` 作为环境回归检查
- 如果后续再次出现相同错误，优先检查用户级 `pip` 源是否又将 Pillow 安装回了非官方镜像构建

## 6. 推荐修复路径

优先级从稳妥到激进如下。

### 路径 A：保留当前环境，继续用脚本做回归检查

执行：

```powershell
& D:\.conda\envs\ecotox-qsar\python.exe scripts\verify_gpu_environment.py
```

用途：

- 输出 `python / core scientific stack / torch / cuda / device / torchvision` 的完整状态
- `torchvision` 检查已改为子进程冷启动探测，避免被当前进程中已加载的 `torch` 掩盖
- 若问题回归，会把“直接导入失败、PIL 预加载后成功”显式记录下来

适用场景：

- 当前阶段先搭主框架、训练骨架、baseline、deep runner
- 暂时还不依赖 `torchvision` 图像增强功能

### 路径 B：若问题回归，显式官方源重装 Pillow

如果要继续清理该问题，优先避免内网代理源，直接使用官方 PyPI：

```powershell
& D:\.conda\envs\ecotox-qsar\python.exe -m pip install --force-reinstall --no-cache-dir --index-url https://pypi.org/simple pillow
```

说明：

- 之前默认 `pip` 源被代理到内网镜像，导致重装失败
- 如果继续用 `pip` 修复，必须显式指定官方源

### 路径 C：在同一环境中重装 `torchvision`

若路径 B 后仍复现，可继续：

```powershell
& D:\.conda\envs\ecotox-qsar\python.exe -m pip install --force-reinstall --no-cache-dir --index-url https://download.pytorch.org/whl/cu124 torchvision==0.20.1
```

说明：

- 保持 `cu124` GPU-only 路线不变
- 不要混入 CPU wheel

### 路径 D：新建干净环境复刻 GPU-only 方案

如果未来必须强依赖 `torchvision` 且上述修复仍失败，建议新建干净环境重新安装：

1. 用 `environment.yml` 建立 Conda 主环境
2. 再用 `requirements-gpu.txt` 安装 PyTorch `cu124` wheels
3. 最后运行 `scripts/verify_gpu_environment.py`

这条路径最稳，但成本高于当前阶段所需。

## 7. 与项目当前阶段的关系

依据项目级 `AGENTS.md`，当前优先级首先是：

1. 需求与架构对齐
2. 环境与依赖清单
3. 项目骨架与配置规范

从这个角度看：

- 当前环境已经满足“GPU-only 主训练底座可用”的阶段目标
- `torchvision` 问题已被定位到较小范围
- 在后续真正接入图像增强或视觉数据管线之前，可以先保留这一剩余问题并持续记录
