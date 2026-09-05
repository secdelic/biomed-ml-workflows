# BioMed ML Workflows 中文正式使用说明书

## 1. 软件简介

BioMed ML Workflows 是科研分析执行工具，不是自动研究设计器。当前提供 DenseNet121 二维医学影像分类、SegResNet 三维医学影像分割、CoxPH 右删失生存预测，以及 52 个科研绘图函数。研究者确定科学问题和方案，软件负责执行明确的数据与模型流程。

## 2. 当前版本和适用范围

Version: **0.1.0**。Tested Python: **Python 3.12**；当前安装要求为 `>=3.12,<3.13`，不是“其他 Python 版本也已支持”。精确依赖见 [pyproject.toml](../pyproject.toml)。

正式分析范围限于上述三条工作流。XGBoost、LightGBM、Random Forest、SVM、KNN 尚未正式集成；没有 DCA、NRI、IDI 或 SHAP 计算工作流。能画特征重要性、聚类或预测诊断图，不代表集成了对应模型训练或解释计算。其他限制见 [KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md)。

本说明书描述 v0.1.0 的软件接口；这是发布后更新到 main 的文档，不声称新增文字已包含在冻结的 v0.1.0 release 中。

## 3. 安装

先安装 Git 和 Python 3.12。以下命令从准备存放软件的父目录执行；后续示例命令均在仓库根目录执行，并使用已激活的虚拟环境。

Windows PowerShell：

```powershell
git clone https://github.com/secdelic/biomed-ml-workflows.git
cd biomed-ml-workflows
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[all,test]"
python --version
pytest
```

如本机策略禁止激活脚本，无须修改系统策略：将后续 `python` 替换为 `.\.venv\Scripts\python.exe`，将 `pytest` 替换为 `.\.venv\Scripts\python.exe -m pytest` 即可。

Linux/macOS：

```bash
git clone https://github.com/secdelic/biomed-ml-workflows.git
cd biomed-ml-workflows
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[all,test]"
python --version
pytest
```

CPU 可运行测试及全部示例，不要求 GPU。GPU 使用取决于本机 PyTorch/CUDA 环境，不保证跨设备数值完全一致。安装失败时先核对解释器、平台依赖与网络，不擅自放宽版本锁定。上述命令安装当前 main；如研究要求冻结版本，在创建环境前明确执行 `git checkout v0.1.0`，但该标签不包含本轮新增文档。

## 4. 推荐项目目录

软件不是数据库管理工具，workflow 消费 arrays、tensors、DataLoader 或 prepared features，没有强制输入目录。推荐将软件目录 `biomed-ml-workflows/` 与研究目录 `MyStudy/` 并列存放：

```text
<MY_PROJECT>/
├── input/
├── scripts/
└── output/
    ├── metrics/
    ├── predictions/
    ├── figures/
    └── logs/
```

`input/` 只是用户研究项目的推荐约定，不是软件自动读取路径。`scripts/` 存该研究的数据适配和执行脚本。真实患者数据留在受控研究目录，不放入公开软件仓库，不提交到 GitHub；去标识化不等于获准公开。不要把凭据或身份映射写进公开日志。

## 5. Quick Start

**不需要用户输入文件**。脚本内部生成固定 synthetic data，默认 seed 为 `20260904`，不下载数据或预训练权重。

```bash
python examples/quick_start/run_quick_start.py
```

默认输出根目录是**软件仓库的** `output/`，不是任意当前工作目录中的同名文件夹。实际生成 10 个文件：

```text
output/
├── metrics/
│   ├── classification.json
│   ├── segmentation.json
│   └── survival.json
├── predictions/
│   ├── classification.npz
│   ├── segmentation.npy
│   └── survival.npz
├── figures/
│   ├── classification_roc.png
│   ├── segmentation_panel.png
│   └── survival_curves.png
└── logs/
    └── quick_start.json
```

| 文件 | 实际保存内容 |
| --- | --- |
| `metrics/classification.json` | accuracy、auroc、auroc_status、best_epoch、test_count |
| `metrics/segmentation.json` | best_epoch、prediction_shape、dice（含通道与状态） |
| `metrics/survival.json` | best_epoch、concordance、comparable_pairs、test_count |
| `predictions/classification.npz` | `labels`、`probabilities`，默认测试集分别为 [4]、[4,2] |
| `predictions/segmentation.npy` | 默认预测类别掩膜 [2,1,8,8,8] |
| `predictions/survival.npz` | `log_risk`、`relative_risk`、`times`、`survival`；生存矩阵 [6,T] |
| `figures/` 的三个 PNG | 分类 ROC、分割图像/标签/预测面板、三条模型生存曲线 |
| `logs/quick_start.json` | status、三条工作流摘要、environment、scientific_claim |

这些是 [Quick Start 源码](../examples/quick_start/run_quick_start.py) 实际选择保存的字段，并非完整 API 对象。没有自动保存全部 logits、sample IDs、分区清单、训练历史或模型文件。状态为 `SYNTHETIC_TECHNICAL_DEMONSTRATION_ONLY`，`scientific_claim` 为 `NONE`；数值不能用作论文性能证据。

## 6. 自定义Quick Start输出位置

`--output` 接受输出根目录，四个子目录直接建在该根目录下，不再附加一层 `output`。

```bash
python examples/quick_start/run_quick_start.py --output "../Demo/output"
```

相对路径按**执行命令时的工作目录**解析。也可给绝对路径；例如 PowerShell 在仓库根目录生成一个指向同级 Demo 项目的绝对路径：

```powershell
$DEMO_OUTPUT = Join-Path (Get-Location).Parent.FullName "Demo/output"
python examples/quick_start/run_quick_start.py --output "$DEMO_OUTPUT"
```

最终写入该路径下的 `metrics/`、`predictions/`、`figures/`、`logs/`。脚本支持 `--seed`，但修改 seed 后已不是默认示例运行。重复指定同一根目录会覆盖同名示例文件；每次验证建议使用新的运行目录。

## 7. 正式科研分析的输入原则

真实研究采用：研究数据 → study-specific adapter/script → workflow API → 显式保存输出。不要通过替换 Quick Start 的 synthetic data 来隐式定义正式分析。

调用前由 Researcher 冻结：research question、population/纳排标准、unit of analysis、outcome、time origin（如适用）、group/patient ID、train/validation/test design、missing-data strategy 和 validation design。适配脚本位于研究项目中，明确文件读取、类型转换、标签映射及 ID 对齐；本软件没有通用 CSV 或医学影像文件入口。

先检查输入来源、样本数、重复 ID、缺失、标签和 shape，再分组划分。任何需要拟合的预处理只使用训练分区；验证集用于选择，测试集保留给最终评估。数据来源或会影响科学结论的定义不明确时停止，不由软件或 Codex 猜测。

## 8. Classification输入与输出

入口：[模型构造](../biomed_ml_workflows/methods/classification/__init__.py)、[workflow API](../biomed_ml_workflows/workflows/classification/__init__.py)。模型为未使用预训练权重的 DenseNet121；`build_densenet121(spatial_dims=2, in_channels=C, out_channels=K, device="cpu")`，K 至少为 2。

### 输入与分区

- 批量图像为 float32 `[B,C,H,W]`；标签建议为整数 `[B]`，类别编码 `0..K-1`。读取层应先验证整数性，不依赖类型转换纠错。
- DataLoader 的 batch 为 `{"image": images, "label": labels, "sample_id": ids}`（ID 可选），或 `(images, labels)`。**`predict_classifier` 和 `evaluate_classifier` 都要求 label**，不能当成无标签预测入口。
- `sample_ids` 是每个样本唯一 ID；`group_ids` 是可重复的患者/分组 ID，用于 `split_samples`，不是模型特征。评估 batch 若提供 sample ID，须每行一个、全程提供且唯一，否则返回 `sample_ids=None`。
- `split_samples(sample_ids, labels, train_fraction=..., validation_fraction=..., test_fraction=..., seed=..., group_ids=..., require_groups=True)` 返回 `SplitResult`：三组 `*_indices`、`*_ids`、`*_group_ids`，以及 seed、分层/分组状态、请求比例和观察比例。
- 三个比例均为正且和为 1；分组后比例作用于组数，不保证精确样本数比例。默认分层要求足够样本/组；同一组用于分层的标签必须一致，不一致应重新确认研究设计，不静默关闭分层。

同一患者多张图片必须 group-aware split。模型构建前调用 `configure_reproducibility(seed=..., device=...)`，DataLoader 可用 `make_dataloader` 设置 seed。训练函数不会替调用方自动设置全局 seed。

图像变换在 methods.classification：`ImageTransformConfig`、`build_train_transforms(config, seed=...)`、`build_eval_transforms(config)`。默认原图是无通道轴的 [H,W]，已有 [C,H,W] 时设 `channel_dim=0`；默认目标尺寸 (32,32)，各维至少 32。默认逐图强度缩放不是跨队列拟合。随机增强仅用于 train，validation/test 用确定性变换。

### 输出与使用边界

`fit_classifier(model, train_loader, validation_loader, device=..., config=TrainingConfig(...))` 返回 `TrainingResult`，不是训练后模型的新副本。默认训练 1 epoch、Adam、交叉熵、学习率 1e-3。按 validation loss 选择并将最佳状态重载到传入的 model；不接收测试集，checkpoint 为内存状态，不自动落盘。

`TrainingResult` 主要字段：`history`（epoch、train/validation 的 loss、accuracy、sample_count）、`best_epoch`、`best_validation_loss`、`selection_partition`、`checkpoint_storage`、`best_checkpoint_reloaded`、`test_data_used`；可用 `to_dict()`。

`evaluate_classifier(model, test_loader, device=...)` 和 `predict_classifier` 返回 `EvaluationResult`：

| 字段 | 内容 |
| --- | --- |
| `logits`、`probabilities` | CPU tensor [N,K]；后者为 softmax |
| `predicted_classes`、`labels` | CPU tensor [N]；预测取 argmax |
| `sample_ids` | 对齐的 ID tuple，未提供时为 None |
| `metrics` | `accuracy`、`auroc`、`auroc_status` |
| `probability_interpretation` | 未校准分数的解释标记 |

二分类 AUROC 使用第 1 类分数，多分类使用 macro one-vs-rest；缺少必需类别时 AUROC 为 None，并记录 `UNDEFINED_REQUIRED_CLASS_ABSENT`，不能填为 0。softmax 不等于已经校准的临床风险。测试集不得用于模型选择。

可调用的 7 类图：类别计数柱状图/饼图、类别样本 montage、混淆矩阵、ROC、PR、训练历史。详见 [绘图接口](FIGURE_CAPABILITIES.md#classification)；API 返回这些结果不代表自动生成所有图。

## 9. Segmentation输入与输出

入口：[SegResNet 模型](../biomed_ml_workflows/methods/segmentation/__init__.py)、[segmentation workflow](../biomed_ml_workflows/workflows/segmentation/__init__.py)。图像 float32 `[B,C,D,H,W]`，batch 键同分类，或 `(images, labels)`；sample ID 对齐规则相同。

使用 `SegmentationLabelContract` 明确标签语义：

| encoding | 目标标签 | 概率与预测 |
| --- | --- | --- |
| `INTEGER_CLASS_MAP` | 整数 [B,D,H,W] 或 [B,1,D,H,W]，0..K-1，K≥2 | logits/probabilities [B,K,D,H,W]；softmax 后 argmax，预测 [B,1,D,H,W] |
| `MULTICHANNEL` | float32 [B,K,D,H,W]，值在 [0,1]，K≥1 | 独立 sigmoid；按 `prediction_threshold`（默认 0.5）得到同形预测，不自动认定通道互斥 |

`out_channels` 必须与模型一致；背景是否计入由 `include_background` 决定，不从数据推断。排除背景时排除通道 0，K=1 不允许排除。阈值由研究方案预先规定，不使用测试集调整。

**先按 volume/patient 分区，再生成 patch/crop**。`split_samples` 可复用分组划分；`validate_patch_partitioning(split, patch_source_volume_ids, patch_partitions)` 检查 patch 来源是否属于已指定分区，并返回数量审计。它不替代患者身份或空间对齐检查。

`SegmentationTransformConfig` 默认图像已有通道轴 0、整数标签没有通道轴；原始 [D,H,W] 图像应设 `image_channel_dim="no_channel"`，多通道标签设 `label_channel_dim=0`。训练变换成对处理图像/标签，只对图像做强度增强；评估不随机增强，标签重采样用最近邻。默认不强制 resize/crop，也没有医学影像文件加载器。

`build_segresnet(spatial_dims=3, in_channels=C, out_channels=K)` 默认下采样结构要求空间维可被 8 整除；改变 blocks 后按结构重新核实，不能把 Quick Start 的 8³ 小例子当成所有正式模型的输入配置。

`fit_segmenter(..., label_contract=contract, device=..., config=SegmentationTrainingConfig(...))` 接收 train/validation loader，默认 Adam/Dice loss、1 epoch、学习率 1e-3；按验证损失选择并重载最佳状态。返回 `SegmentationTrainingResult`：history（epoch、train/validation 的 loss、sample_count、batch_count）、best_epoch、best_validation_loss、label_encoding、loss_name、amp_enabled，以及选择分区、选择指标、checkpoint 和 test 使用状态。训练历史**不自动包含逐 epoch Dice**。

`evaluate_segmenter` / `predict_segmenter` 均要求标签，返回 `SegmentationEvaluationResult`：

- CPU `logits`、`probabilities` 为 [N,K,D,H,W]；`predictions`、`labels` 按上述契约；还有 `sample_ids`、`label_encoding`、`inference_mode`、`score_interpretation`。
- `dice` 为 `SegmentationDiceResult`，含 `mean_dice`、`channels`、`empty_target_policy`；每通道含 channel_index、dice、status、valid_sample_count、total_sample_count。目标和预测均空的样本/通道项为未定义而被排除，不自动记满分；overall 是已定义通道均值的平均。
- 可传 `SlidingWindowConfig(roi_size=(...), overlap=0.25, sw_batch_size=1)`，ROI 要满足模型空间约束。输出恢复到输入空间大小；批次间拼接要求空间尺寸一致，异形体积可分别调用。AMP 需显式启用且仅限 CUDA。

无标签推理可用 `infer_segmentation_logits(model, images, device=..., sliding_window=...)`，仅返回 logits tensor，不自动生成 Dice 或完整评估对象。8 种分割图及所需切片/通道、已计算历史见 [分割绘图接口](FIGURE_CAPABILITIES.md#segmentation)。

## 10. Survival输入与输出

入口：[CoxPH 模型](../biomed_ml_workflows/methods/survival/__init__.py)、[survival workflow](../biomed_ml_workflows/workflows/survival/__init__.py)。模型是 pycox CoxPH 神经网络风险模型。

输入为准备好的数值 features [N,P]、durations [N]、events [N]；推荐 NumPy arrays。features 必须有限、非空，durations 必须有限且非负；events 必须严格为数值 0/1：**1 = observed event，0 = right-censored**。行顺序和 sample IDs 一一对应，不把 ID 放入特征。训练、验证及 concordance 评估需要事件；没有可比较样本对时 concordance 报错，而不是制造一个分数。

时间单位、time origin、结局定义及删失合理性由研究者决定。使用 `split_survival_samples` 可按事件分层并按组划分；同组分层标记需一致，重复观测的科学处理不能靠分组 API 自动解决。

`fit_train_only_preprocessor(preprocessor, train_features, validation_features, test_features)` 仅在训练数据 fit，其他分区只 transform；返回 `PreprocessedPartitions` 的 train、validation、test 及拟合边界标记。**此函数在 fit 前就拒绝 NaN/非有限输入**，不能直接传入原始缺失矩阵期待 SimpleImputer 自动处理。先在 study-specific adapter 中按已冻结方案、训练拟合边界处理缺失，再传入有限特征做标准化等操作；保存使用过的变量编码和预处理配置。

模型构建前调用 `configure_coxph_reproducibility(seed=..., device=...)`。用 `build_coxph_model(in_features=P, hidden_dims=(16,), device="cpu")` 构建模型，再调用：

```text
fit_coxph(model,
          train_features, train_durations, train_events,
          validation_features, validation_durations, validation_events,
          config=CoxPHTrainingConfig(...))
```

该函数没有 device 参数，不接受 test 数据，使用每轮完整训练分区风险集更新。默认 Adam、20 epochs、学习率 1e-3。按验证损失选模、重载最佳状态，随后**仅以训练 features/durations/events 估计 baseline hazard**。

返回 `CoxPHTrainingResult`：history（epoch、train_loss、validation_loss）、best_epoch、best_validation_loss、optimizer、full_training_partition_per_update、selection_partition、checkpoint_storage、best_checkpoint_reloaded、test_data_accepted_by_training_api、test_data_used、baseline_hazard_partition、baseline_hazard_uses_training_outcomes_only；可用 `to_dict()`。

`evaluate_coxph(model, features, durations, events, sample_ids=...)` 返回 `CoxPHEvaluationResult`：

| 字段 | 内容 |
| --- | --- |
| `risk` | `RiskPrediction`：sample_ids、log_risk [N]、relative_risk [N]、interpretation |
| `survival` | `SurvivalPrediction`：sample_ids、times [T]、survival_probabilities [N,T]、interpretation |
| `concordance` | `ConcordanceResult`：value、comparable_pairs、method、interpretation；Harrell C-index，无 IPCW |

也可分别调用 `predict_log_risk` 和 `predict_survival`，只需 features 及可选 sample_ids；生存预测要求已有 baseline hazard。缺省 ID 是按行生成的字符串，不是患者身份验证。relative_risk = exp(log_risk)，**relative risk != probability**；模型生存概率也未自动校准。

绘制生存曲线使用 [T,N]，而 workflow 输出为 [N,T]，必须按语义转置。7 种 [生存图](FIGURE_CAPABILITIES.md#survival) 中，KM、Brier、NBLL、学习率搜索/调度图所需序列须另有已验证结果；当前训练/评估不自动计算这些全部项目，也不完成 PH assessment 或校准。

## 11. 输出位置与保存建议

Quick Start 自动保存第 5 节的固定文件。正式 Python API 主要返回内存对象，不决定研究目录，也不自动建立完整审计档案。推荐在用户脚本中明确根目录及四个子目录：

```python
from pathlib import Path

OUTPUT = Path("../MyStudy/output/run_001").resolve()  # 相对当前工作目录
for name in ("metrics", "predictions", "figures", "logs"):
    (OUTPUT / name).mkdir(parents=True, exist_ok=True)
```

建议 metrics 保存指标及未定义状态；predictions 保存数值与对齐 ID/标签；figures 保存图；logs 保存输入清单、split、seed、环境、参数、预处理拟合范围及执行命令。Tensor 转 NumPy 使用 `tensor.detach().cpu().numpy()`；有 `to_dict()` 的对象可转 JSON，但不能假定所有评估对象都能直接 JSON 序列化。

软件不会自动保存磁盘 checkpoint。模型/预处理器是否保存及如何安全重载，应由研究脚本显式实现并验证；不要加载来源不明的序列化对象。每次运行用新目录，不覆盖原始数据或已确认结果。这只是保存约定，不是新增框架。

## 12. 52种绘图功能

| 类别 | 数量 | 模块 |
| --- | --- | --- |
| Statistical | 27 | `biomed_ml_workflows.figures.statistical` |
| Classification | 7 | `biomed_ml_workflows.figures.classification` |
| Segmentation | 8 | `biomed_ml_workflows.figures.segmentation` |
| Survival | 7 | `biomed_ml_workflows.figures.survival` |
| Model Interpretation | 3 | `biomed_ml_workflows.figures.interpretation` |

完整函数名、shape、例调用和限制见 [FIGURE_CAPABILITIES.md](FIGURE_CAPABILITIES.md)，不必复制整张能力表。基本契约是“显式结果 → `(Figure, Axes)`”；多面板的第二项可能是 Axes 数组。函数默认不保存，不重新训练/拟合模型，不选择临床 threshold，也不决定研究设计。

部分函数会根据输入计算 ROC/PR、混淆计数、密度或相关性等描述性量；不能笼统理解为“完全不计算”。KM/特征重要性/遮挡敏感性等函数绘制已计算输入，不负责对应估计或归因计算。

运行默认 Quick Start 后，在仓库根目录复制以下代码即可保存已有预测的 ROC：

```python
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from biomed_ml_workflows.figures.classification import plot_roc_curve

root = Path("output")  # 自定义Quick Start根目录时同步修改
with np.load(root / "predictions" / "classification.npz", allow_pickle=False) as data:
    fig, ax = plot_roc_curve(
        data["labels"], data["probabilities"][:, 1], class_names=["Positive"]
    )
target = root / "figures"
target.mkdir(parents=True, exist_ok=True)
for extension in ("png", "pdf", "svg"):
    fig.savefig(target / f"classification_roc_export.{extension}")
plt.close(fig)
```

这是读取既有示例结果的保存用法，不是重新训练。二分类上述调用只画阳性一条曲线，因此 class_names 只有一个名称。正式结果必须核实类别编码和分数列顺序，不能机械套用。PNG/PDF/SVG 通过 Matplotlib Figure.savefig 保存，不是给 plot 函数传不存在的保存参数。

## 13. 一次性生成全部图形示例

```bash
python examples/figure_gallery/run_all_figures.py
```

[gallery 脚本](../examples/figure_gallery/run_all_figures.py) 使用 seed `20260904`，为 52 个函数各生成 1 张 PNG，打印 `GENERATED_FIGURES=52`。默认固定在**仓库根目录** `output/figure_gallery/`，结构为：

```text
output/figure_gallery/<family>/<family>_<function_name>.png
```

family 为 statistical、classification、segmentation、survival、interpretation。例如 `classification/classification_plot_roc_curve.png`。该脚本**没有自定义输出路径的 CLI 参数，不支持 --output**；需要研究专用保存路径时按第 11–12 节调用函数并保存，不假设 gallery 支持该参数。

所有图均为 **SYNTHETIC DEMONSTRATION — NOT SCIENTIFIC PERFORMANCE EVIDENCE**。这与 Quick Start 的三个 workflow 示例图不同，也不代表正式研究已生成 52 个有效结果。重复运行会写入同名示例文件。

## 14. 推荐的正式分析工作流

冻结研究设计 → 核对数据及 patient/group unit → 固定 train/validation/test 分区 → train-only 预处理及分区内 patch/crop → 调用 workflow 训练 → validation 选模 → 保留 test 最终评估 → 保存 metrics/predictions/logs → 调用 figures → 人工核对 → 论文报告。

不要先对全体数据拟合预处理、扩增或裁 patch 后再随机分区。若需外部验证、重采样或其他设计，由研究者预先指定并在项目脚本实现，不能把一次随机划分自动当作完整验证。

## 15. 使用Codex的标准方法

让 Codex 首先读取仓库 [AGENTS.md](../AGENTS.md)，然后提供 RESEARCH QUESTION、DATA PATH、OUTPUT PATH、UNIT OF ANALYSIS、OUTCOME、TIME ORIGIN、GROUP ID、MISSING DATA STRATEGY、VALIDATION DESIGN，以及研究人群。替换下列模板中所有占位符；不适用的项写明不适用，而不是留给 Codex 推断。

应先检查输入契约，再在研究项目的 scripts 中复用 API；不让 Codex 改变临床定义、静默修复标签或用 test 选参。额外约定见 [CODEX_USAGE.md](CODEX_USAGE.md)。五个模板也集中在 [CODEX_PROMPTS_ZH_CN.md](CODEX_PROMPTS_ZH_CN.md) 供复制。

## 16. Codex通用工作流提示词

```text
# BioMed ML Workflows Study Analysis

你现在在：<REPOSITORY_PATH>
使用 BioMed ML Workflows。首先读取：
AGENTS.md
docs/USER_GUIDE_ZH_CN.md
docs/FIGURE_CAPABILITIES.md

研究项目：<PROJECT_NAME>
研究问题：<RESEARCH_QUESTION>
研究人群及纳排标准：<POPULATION>
输入数据位置：<INPUT_PATH>
输出根目录：<OUTPUT_PATH>
分析类型：<CLASSIFICATION / SEGMENTATION / SURVIVAL>
样本单位：<UNIT_OF_ANALYSIS>
患者/分组ID：<GROUP_ID>
结局或标签：<OUTCOME_OR_LABEL>
时间起点：<TIME_ORIGIN_OR_NOT_APPLICABLE>
缺失数据策略：<MISSING_DATA_STRATEGY>
验证设计（含train/validation/test划分）：<VALIDATION_DESIGN>

我的要求：
1. 先检查输入来源、结构、shape、ID唯一性、标签编码与缺失情况。
2. 不改变上述Researcher定义；先确认分组与划分，再拟合预处理、生成patch或训练。
3. 在研究项目scripts目录编写必要的数据适配脚本，优先调用仓库现有workflow，
   不重新实现已有方法，不直接改写软件的synthetic examples。
4. train-only preprocessing；augmentation只用于train。
5. validation用于模型选择；test不得参与预处理拟合、调参或模型选择。
6. 如果发现patient/group leakage风险，停止并报告。
7. 在模型构建前设置并记录seed、设备、版本、参数和分区清单。
8. 运行相关测试及最小输入/输出验证，不把TECHNICAL PASS写成SCIENTIFIC VALIDATION。
9. 所有正式输出写入：
   <OUTPUT_PATH>/metrics
   <OUTPUT_PATH>/predictions
   <OUTPUT_PATH>/figures
   <OUTPUT_PATH>/logs
10. 优先使用仓库现有figure functions，不为了图形美观修改分析结果。
11. 不覆盖原始数据，不将患者数据、身份信息或凭据提交到公开仓库。
12. 最终报告输入数据、shape、split、模型、参数、tests、metrics、
    prediction文件、figures、output paths和remaining scientific limitations。

如果输入来源不明，或缺少会改变科学结论的重要定义：STOP，不要自行猜测。
```

## 17. Classification专用Codex提示词

```text
请在 <REPOSITORY_PATH> 读取AGENTS.md、中文说明书和FIGURE_CAPABILITIES.md。
使用现有DenseNet121 classification workflow，不重新实现模型。
先使用通用模板补齐研究问题、人群、样本单位、缺失策略和验证设计；
未提供这些定义时STOP。

INPUT_PATH: <INPUT_PATH>
OUTPUT_PATH: <OUTPUT_PATH>
IMAGE: <图像来源、通道、尺寸及读取约定>
LABEL: <标签字段、定义及从0开始的整数编码>
PATIENT/GROUP ID: <GROUP_ID>
CLASS NAMES: <按类别编码排列的名称>
SPLIT DESIGN: <train/validation/test划分、分组规则及seed>

检查图像[B,C,H,W]、标签[B]及sample ID对齐；同一患者不得跨分区。
仅train使用augmentation，预处理仅在train拟合，validation选模，test最终评估。
predict_classifier也需要标签，不虚构无标签的高层推理接口。
在项目scripts中适配数据，运行相关测试，调用现有绘图函数。
保存metrics、predictions、figures、logs到OUTPUT_PATH下，报告参数、
分区、输出路径及科学限制；缺少定义或发现泄漏即STOP，不覆盖原始数据。
```

## 18. Segmentation专用Codex提示词

```text
请在 <REPOSITORY_PATH> 读取AGENTS.md、中文说明书和FIGURE_CAPABILITIES.md。
使用现有SegResNet segmentation workflow。
先使用通用模板补齐研究问题、人群、样本单位、缺失策略和验证设计；
未提供这些定义时STOP。

INPUT_PATH: <INPUT_PATH>
OUTPUT_PATH: <OUTPUT_PATH>
IMAGE CHANNELS: <通道含义、顺序及空间方向/分辨率>
MASK/LABEL CONTRACT: <INTEGER_CLASS_MAP或MULTICHANNEL、类别/通道定义、
out_channels、背景约定及预先确定的prediction_threshold>
PATIENT/GROUP ID: <GROUP_ID>
VOLUME UNIT: <体积ID、患者对应关系及分析单位>
SPLIT DESIGN: <train/validation/test划分、分组规则及seed>

先volume/patient split，后patch/crop；检查patch来源分区，禁止跨分区泄漏。
核实图像[B,C,D,H,W]与标签shape、空间对齐和编码。
train-only preprocessing及augmentation；validation选模，test最终评估。
显式记录滑窗roi_size、overlap、sw_batch_size、设备及Dice背景/空目标策略。
运行相关测试；优先用现有分割图函数，切片选择按研究者预定规则。
保存metrics、predictions、figures、logs到OUTPUT_PATH下；报告输入、参数、
分区、输出路径及科学限制。契约不明确或发现泄漏即STOP，不覆盖原始数据。
```

## 19. Survival专用Codex提示词

```text
请在 <REPOSITORY_PATH> 读取AGENTS.md、中文说明书和FIGURE_CAPABILITIES.md。
使用现有CoxPH survival workflow。
先使用通用模板补齐研究问题、人群和样本单位；未提供这些定义时STOP。

INPUT_PATH: <INPUT_PATH>
OUTPUT_PATH: <OUTPUT_PATH>
FEATURES: <变量清单、单位、编码和测量时点>
DURATION: <随访时长字段及单位>
EVENT: <结局定义及编码映射：1=observed event，0=right-censored>
TIME ORIGIN: <时间起点>
PATIENT/GROUP ID: <GROUP_ID>
MISSING DATA STRATEGY: <预先指定的缺失处理及train-only拟合方案>
VALIDATION DESIGN: <train/validation/test划分、seed、PH assessment及calibration计划>

核实features[N,P]、durations[N]、events[N]、有限值和行/ID对应关系。
先分区后拟合预处理；原始缺失值须由study-specific adapter按方案处理，
不能将NaN直接传给fit_train_only_preprocessor。
validation选模，test不得参与拟合/选择；baseline hazard仅用training outcome估计。
event编码、PH assessment、calibration、time origin仍属于study-specific
scientific validation，不宣称本workflow自动完成这些验证。
relative risk不是probability；绘制生存曲线时核对[N,T]到[T,N]的方向。
运行相关测试，保存metrics、predictions、figures、logs到OUTPUT_PATH下，
报告参数、分区、指标适用范围及科学限制；发现缺失定义或泄漏即STOP。
```

## 20. 只绘图的Codex提示词

```text
请在 <REPOSITORY_PATH> 读取AGENTS.md及docs/FIGURE_CAPABILITIES.md。
我已有分析结果：<RESULT_PATH>
图形输出目录：<FIGURE_OUTPUT_PATH>
需要的图形、变量含义及单位：<FIGURE_TYPES_AND_DATA_CONTRACT>
已冻结的类别顺序、threshold、risk group及时间单位：<EXISTING_DEFINITIONS>

先检查结果来源、shape、ID/标签/时间轴对齐及与研究定义的一致性。
仅调用现有figure functions，不重新训练模型，不重新拟合preprocessing，
不改变threshold，不改变prediction，不重新定义risk group。
不把绘图API内的描述性计算当作研究设计或模型验证。
函数所需结果缺失时STOP，不自行补做模型分析。
使用函数返回的Figure，通过Figure.savefig分别保存PNG、PDF、SVG
（以当前Matplotlib后端实际支持情况为准），不虚构函数的save参数。
报告源结果、绘图函数、参数、输出文件及视觉问题；不覆盖原始结果，
不把synthetic demonstration作为科学证据。
```

## 21. 论文中如何记录

以下是记录模板，不含研究结果；将占位符替换为本次实际运行信息：

```text
Software: BioMed ML Workflows v0.1.0
Repository: https://github.com/secdelic/biomed-ml-workflows
Version/tag and exact commit: <实际tag及git rev-parse HEAD>
Python: <实际3.12.x>
Main dependencies: <本次环境中的torch、monai、pycox、numpy等版本>
Seed/device: <seed、CPU/GPU、确定性设置>
Split: <样本与分组单位、各分区数量、分层策略及清单版本>
Preprocessing: <变量/图像处理、缺失策略、仅train拟合的证据>
Model: <构造参数、训练参数、validation选择规则>
Validation: <留出/外部验证方案、指标定义、未定义项及限制>
Figure generation: <函数、输入结果版本、参数和文件路径>
```

软件版本与文档所在分支分别记录，不把更新后的 main 等同于冻结标签。环境与复现建议见 [REPRODUCIBILITY.md](REPRODUCIBILITY.md)，引用元数据见 [CITATION.cff](../CITATION.cff)。不虚构 DOI、运行记录或论文指标；合成例子的指标不能转为正式研究结果。

## 22. 科研边界

软件不能替 Researcher 决定 population、outcome、exposure、time origin、missing data、clinical threshold、validation design、causal interpretation 或 clinical recommendations。它也不能自动证明样本量充分、无偏倚、标签可信、概率已校准或模型可推广。

**TECHNICAL PASS != SCIENTIFIC VALIDATION。** 测试通过、图像可读或示例执行成功，只说明对应技术路径运行；不能用于临床决策，也不能代替研究特异性审查、比例风险检验、校准和外部验证。

## 23. 常见问题

**Q1 输入文件必须放在仓库里吗？** 不需要。数据位置由研究脚本指定，真实患者数据不应提交到公开仓库。

**Q2 Quick Start 输出在哪？** 默认在软件仓库的 output 下，包含 metrics、predictions、figures、logs，具体 10 个文件见第 5 节。

**Q3 如何修改输出目录？** Quick Start 用 `--output`；正式 API 由调用脚本决定保存路径。gallery 没有对应 CLI 参数，不能混用。

**Q4 为什么 figure function 不自动保存？** 它返回 Figure/Axes，由调用方决定路径、文件名和格式；调用 `fig.savefig(...)`，保存后可 `plt.close(fig)`。

**Q5 GPU 必须吗？** 不必须。CPU 可运行全部示例和测试；正式三维数据的内存与耗时仍需按数据量评估。

**Q6 能否直接输入 CSV？** 没有通用 CSV loader 或“CSV 路径 → 三条 workflow”的入口。需 study-specific adapter 读取文件、核对变量、缺失与行对应关系，再转换成 API 所需 arrays/tensors/DataLoader/features。

**Q7 可以做 XGBoost 吗？** v0.1.0 尚未正式集成 XGBoost workflow；可以绘制兼容的既有结果，不等于软件实现了该模型。

**Q8 可以直接用于论文吗？** 软件实现可作为分析工具，但研究方案、数据质量、统计假设、校准、验证和解释仍须研究者负责，不能以 synthetic examples 替代真实研究证据。

## 24. GitHub中文文档入口

[README 中文入口](../README.md#chinese-documentation) 链接本说明书和 [Codex中文提示词](CODEX_PROMPTS_ZH_CN.md)。GitHub 当前文档见 [main 分支](https://github.com/secdelic/biomed-ml-workflows/tree/main/docs)。

main 可以包含发布后的新版文档；冻结的 v0.1.0 tag/release 不随之改变。本轮不移动标签、不重建 release，也不创建 v0.1.1。如果以后要求归档版本包含新版说明书，可由 Researcher 另行决定是否发布 v0.1.1 documentation release；这不是本轮自动动作。
