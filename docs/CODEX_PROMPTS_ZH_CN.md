# BioMed ML Workflows Codex中文提示词

## 1. 通用分析Prompt

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

## 2. Classification Prompt

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

## 3. Segmentation Prompt

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

## 4. Survival Prompt

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

## 5. Figure-only Prompt

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
