# BioMed ML Workflows

Reproducible Python workflows for biomedical machine learning, medical-image deep learning, survival analysis, and scientific visualization.

Version: **0.1.1** · Tested with **Python 3.12**

## Scope

- DenseNet121 2-D classification with explicit splitting, training, and held-out evaluation.
- SegResNet 3-D segmentation with explicit label contracts, patch-boundary checks, training, Dice evaluation, and sliding-window inference.
- CoxPH right-censored survival modeling with train-only preprocessing, validation-only selection, risk prediction, survival prediction, and concordance evaluation.
- 52 reusable Matplotlib figure functions across statistical, classification, segmentation, survival, and model-interpretation families.

## Out of scope

The package does not define study populations, outcomes, time origins, missing-data strategies, clinical thresholds, calibration plans, or external-validation designs. Software execution is not clinical or scientific validation.

## Supported analysis workflows

Public APIs are under:

- `biomed_ml_workflows.workflows.classification`
- `biomed_ml_workflows.workflows.segmentation`
- `biomed_ml_workflows.workflows.survival`

Model constructors are under `biomed_ml_workflows.methods`.

## Figure capabilities

The 52 figure functions consume explicit arrays or validated workflow outputs, do not train models, do not mutate inputs, and return Matplotlib `Figure` and `Axes` objects. See [the complete figure manual](docs/FIGURE_CAPABILITIES.md).

## Installation

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[all,test]"
```

Python versions other than 3.12 have not been certified for this release.

## Quick Start

```bash
python examples/quick_start/run_quick_start.py
```

This runs all three workflows with fixed synthetic data and writes ignored outputs under `output/metrics`, `output/predictions`, `output/figures`, and `output/logs`.

## Using with Codex

Ask Codex to read [AGENTS.md](AGENTS.md), preserve the study definitions, select an existing workflow, validate its input/output contract, and use existing figure functions. See [docs/CODEX_USAGE.md](docs/CODEX_USAGE.md).

## Chinese documentation

- [中文正式使用说明书](docs/USER_GUIDE_ZH_CN.md)
- [Codex中文提示词](docs/CODEX_PROMPTS_ZH_CN.md)

## Synthetic examples

Generate one demonstration for every public figure function:

```bash
python examples/figure_gallery/run_all_figures.py
```

All example results are synthetic technical demonstrations and are not scientific performance evidence.

## Validation

```bash
pytest
python examples/quick_start/run_quick_start.py
python examples/figure_gallery/run_all_figures.py
```

The public tests cover imports, all three workflows, leakage-sensitive boundaries, all 52 figure functions, headless rendering, examples, documentation, metadata, and release-cleanliness gates.

## Reproducibility

Random seeds, partition roles, preprocessing fit boundaries, checkpoint selection, and environment reporting are explicit. See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## Citation

Preferred formal citation for **BioMed ML Workflows v0.1.1**: [Version DOI: 10.5281/zenodo.22346004](https://doi.org/10.5281/zenodo.22346004). Cite this fixed version rather than moving main.

All versions / Concept DOI: [10.5281/zenodo.22346003](https://doi.org/10.5281/zenodo.22346003).

See the [GitHub v0.1.1 Release](https://github.com/secdelic/biomed-ml-workflows/releases/tag/v0.1.1) and [CITATION.cff](CITATION.cff) for software and citation metadata.

## License

Released under the [MIT License](LICENSE). Dependency licenses are listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Data privacy

The repository contains no patient data, datasets, credentials, or serialized models. Examples generate synthetic data locally and perform no downloads.

