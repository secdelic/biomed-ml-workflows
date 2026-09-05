# BioMed ML Workflows v0.1.1

## Scope

Documentation-synchronized stable maintenance release.
Release type: DOCUMENTATION_AND_USAGE_SYNCHRONIZATION_RELEASE.

## Supported analysis workflows

- DenseNet121 2-D classification
- SegResNet 3-D segmentation
- CoxPH right-censored survival modeling

## Figure capabilities

52 public plotting functions across statistical, classification, segmentation, survival, and model interpretation families.

## Documentation

- Expanded Chinese formal user guide.
- Explicit input/output locations and conventions.
- Reusable Codex prompts for generic analysis, classification, segmentation, survival, and figure-only execution.
- README Chinese documentation navigation, reproducibility guidance, and known limitations.

## Validation

- Python 3.12.10: 18 tests passed; two existing dependency deprecation warnings.
- Quick Start: classification, segmentation, and survival passed; all ten documented files verified.
- Figure gallery: 52 public functions, 52 generated and readable PNGs, zero failed functions.
- Documentation: 88 local links and anchors passed, including Chinese guide and prompt navigation.
- Public-file cleanliness gates passed: no source mappings, course traces, audit artifacts, private paths, patient data, secrets, serialized models, or unexpected large files detected.
- Diff review confirmed frozen implementations and dependency configuration unchanged; git diff --check passed.

Publication additionally requires successful GitHub CI on the exact release commit before tagging. Release-date metadata is deferred until publication actually succeeds.

## Behavioral compatibility

- ANALYTICAL_BEHAVIOR_CHANGE = NO
- PLOTTING_BEHAVIOR_CHANGE = NO
- PUBLIC_API_BREAKING_CHANGE = NO
- MODEL_BEHAVIOR_CHANGE = NO
- SCIENTIFIC_SCOPE_EXPANSION = NO

Dependencies, Python compatibility, algorithms, model architectures, metrics, splitting, preprocessing, thresholds, and figure appearance are unchanged. Existing dependency and GitHub Actions deprecation notices remain KNOWN_NONBLOCKING_MAINTENANCE_WARNINGS.

Synthetic examples establish technical execution only, not study-specific scientific or clinical validity. v0.1.0 remains an IMMUTABLE_HISTORICAL_RELEASE.

## License

MIT.

## Citation

Use CITATION.cff. No DOI is claimed before archival issues one. GitHub publication does not itself establish Zenodo archival; actual publication dates are recorded on moving main after release creation without moving the frozen tag.
