# Using BioMed ML Workflows with Codex

Ask Codex to read `AGENTS.md` before changing or running the repository. Supply the study question, population, outcome, time origin, split unit, data dictionary, and desired outputs explicitly.

Codex must preserve researcher-defined study semantics. It should use an existing workflow before creating analysis code and use an existing figure function before writing plotting code. A plotting function must never fit or refit a model.

A useful request has this form:

> Use the existing segmentation workflow on my prepared arrays. Treat subject ID as the split group, preserve the integer-class-map label contract, run the relevant tests, and report software outputs separately from scientific interpretation.

Before accepting a result, require Codex to report:

- input origin and shape;
- partition and preprocessing boundaries;
- exact command and package environment;
- tests and checks actually run;
- output paths;
- limitations that remain study-specific.

Do not ask Codex to infer missing clinical definitions, silently repair labels, mix partitions, select a model using test data, or present synthetic demonstrations as evidence.
