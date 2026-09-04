# BioMed ML Workflows 中文使用指南

## 1. 软件定位

本软件提供可复现的生物医学机器学习工作流和科学绘图函数，包括二维图像分类、三维图像分割、生存分析以及 52 个绘图函数。它提供技术实现，不替代研究方案、统计分析计划或临床判断。

## 2. 安装

仅认证 Python 3.12：

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -e ".[all,test]"
```

Windows 激活命令为 `.venv\Scripts\activate`；Linux 和 macOS 为 `source .venv/bin/activate`。

## 3. 数据准备

在加载数据前明确样本单位、分组单位、结局、时间起点和训练/验证/测试划分。重复测量或同一受试者的多张图像必须通过 `group_ids` 保持在同一分区。缺失值处理和标准化只能在训练分区拟合。本仓库不附带数据集。

## 4. Classification

使用 `biomed_ml_workflows.methods.classification` 构建 DenseNet121，使用 `biomed_ml_workflows.workflows.classification` 完成分区、数据加载、训练和留出集评估。输入为二维图像，标签为从零开始的整数类别。模型输出是未校准分数及其 softmax 值，不能直接解释为临床概率。

## 5. Segmentation

使用 `biomed_ml_workflows.methods.segmentation` 构建三维 SegResNet，使用 `biomed_ml_workflows.workflows.segmentation` 定义标签编码、变换、训练、Dice 评估和滑窗推理。应先按体积分区，再生成 patch，以免同一体积跨分区泄漏。

## 6. Survival

使用 `biomed_ml_workflows.methods.survival` 和 `biomed_ml_workflows.workflows.survival` 构建 CoxPH。持续时间必须非负，事件必须编码为 0/1，预处理器只在训练数据拟合。风险分数不是生存概率；生存曲线需要先根据训练数据估计基线风险。

## 7. 绘图功能

`biomed_ml_workflows.figures` 导出 52 个函数，覆盖 Statistical、Classification、Segmentation、Survival 和 Model Interpretation。函数接收显式数组，返回 Matplotlib Figure/Axes，不保存文件、不训练模型、不改变输入。完整接口见 [FIGURE_CAPABILITIES.md](FIGURE_CAPABILITIES.md)。

## 8. Quick Start

```bash
python examples/quick_start/run_quick_start.py
```

该命令以固定随机种子运行三条合成数据工作流，并在 `output/` 下生成指标、预测、图和日志。这些结果仅用于技术验证。

## 9. Codex 使用方法

让 Codex 先读取根目录 `AGENTS.md`，再说明研究问题、样本单位、结局、时间起点、数据分区和目标输出。要求它优先复用现有工作流与绘图函数，并运行相关测试。具体约定见 [CODEX_USAGE.md](CODEX_USAGE.md)。

## 10. 论文分析使用方法

先冻结研究方案和统计分析计划，再把明确的数据契约映射到软件接口。分别报告训练、模型选择和最终评估使用的分区；记录随机种子、软件版本和预处理拟合范围。图表只展示已经计算且含义明确的量。

## 11. 输出解释

分类输出包括 logits、softmax 值、类别预测和基础判别指标；分割输出包括 logits、预测掩膜和按通道 Dice；生存输出包括对数相对风险、相对风险、生存概率和 concordance。空目标等特殊情况通过状态字段显式标记。

## 12. 科研边界

软件通过不等于研究有效。用户仍需验证队列代表性、偏倚、样本量、标签质量、缺失机制、校准、比例风险假设、外部验证和临床适用性。禁止把合成示例结果作为论文证据。

## 13. 常见问题

- 安装失败：确认解释器是 Python 3.12，并升级 pip。
- GPU 不可用：所有认证命令均可在 CPU 上运行。
- 指标异常：先检查标签编码、分区独立性和输出含义。
- 图没有自动保存：绘图 API 按设计返回 Figure，由调用方决定保存位置。
- 生存曲线不可用：确认模型已在训练数据上估计基线风险。

## 14. 如何引用

请使用仓库根目录 [CITATION.cff](../CITATION.cff) 中的作者、标题和版本信息。只有在正式归档服务签发 DOI 后才能引用 DOI；本版本不预填或推测 DOI。
