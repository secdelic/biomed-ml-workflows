# Reproducibility

BioMed ML Workflows makes technical reproducibility explicit but cannot guarantee scientific reproducibility without a complete study protocol and governed data.

## Certified environment

Version 0.1.0 is certified only on Python 3.12. The installation command is:

```bash
python -m pip install -e ".[all,test]"
```

Dependencies are pinned in `pyproject.toml`. Record the operating system, Python version, installed package versions, hardware, and command line with each analysis.

## Randomness

Public workflow configuration accepts explicit seeds. The Quick Start and figure gallery use a fixed seed. Repeated runs on the same CPU software stack are expected to be deterministic; numerical variation can occur across hardware, operating systems, libraries, and accelerator kernels.

## Partition boundaries

Create train, validation, and test partitions before preprocessing, augmentation, patch generation, or model fitting. Use group-aware splitting whenever observations share a subject or acquisition unit. Fit preprocessing only on training data, select checkpoints only with validation data, and evaluate the test set only after selection is complete.

## Artifacts

The example commands write to ignored `output/` directories. Preserve the input checksum or governed data version, configuration, environment, metrics, predictions, figures, and logs for a real study. Serialized models are intentionally not distributed with the package.

## Verification

Run:

```bash
pytest
python examples/quick_start/run_quick_start.py
python examples/figure_gallery/run_all_figures.py
```

Passing these checks establishes software execution for the tested environment. It does not validate a dataset, endpoint, causal claim, clinical threshold, calibration model, or external generalizability.
