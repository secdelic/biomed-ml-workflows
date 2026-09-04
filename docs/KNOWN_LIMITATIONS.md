# Known Limitations

- Only Python 3.12 is certified for version 0.1.0.
- CPU execution is the release baseline. GPU behavior can vary by driver, accelerator, and deterministic-kernel availability.
- The package does not ingest clinical file formats or define a data-governance process.
- DenseNet121 softmax outputs are uncalibrated model scores unless a study-specific calibration procedure is performed and validated.
- SegResNet requires the caller to define channel semantics, voxel spacing policy, label encoding, patch strategy, and clinically meaningful overlap metrics.
- CoxPH requires a defensible time origin, censoring definition, proportional-hazards assessment, and study-specific calibration. Relative risk is not an absolute probability.
- Built-in metrics are intentionally limited and do not replace confidence intervals, sensitivity analyses, subgroup analyses, calibration assessment, or external validation.
- Figure functions visualize caller-supplied values. They do not verify the scientific validity of those values or fit statistical models.
- Synthetic examples test software paths only and must not be cited as biomedical evidence.
- No patient data, pretrained weights, fitted models, or DOI are included.
