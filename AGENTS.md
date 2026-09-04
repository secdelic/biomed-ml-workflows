# Codex execution contract

Preserve the Researcher's study question and definitions of population, outcome, time origin, split unit, missing-data handling, and validation design. Stop when a required scientific definition is missing.

Use an existing workflow before creating analysis code and an existing function in `biomed_ml_workflows.figures` before creating plotting code. Plot functions must consume explicit outputs; never train, refit, select checkpoints, tune thresholds, or change cohorts inside them.

Keep fitting and preprocessing restricted to training data, use validation data only for declared selection, and reserve test data for held-out evaluation. Never load unknown serialized models or add patient data, credentials, or downloaded datasets.

Run relevant tests and examples. Report technical outputs separately from scientific conclusions; successful execution does not establish calibration, generalizability, clinical utility, or external validity.

