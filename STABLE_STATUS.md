# Stable baseline status

Software version: **0.1.0**. Validation date: **2026-09-05**.

Validated main commit at freeze: `2a3465d5ffd31a5f4df17f367b582732b29e0fbf`.
This identifies the frozen software and documentation content before adding this status-only record; the record's commit is not a new software release.

Current baseline: `POST_V0_1_0_DOCUMENTATION_SYNCHRONIZED`.
Researcher decision: `APPROVED_FOR_STABLE_FREEZE`.
Final status: `BIOMED_ML_WORKFLOWS_FREEZE_WITH_NONBLOCKING_WARNINGS`.

## Frozen scope

- `ANALYTICAL_BEHAVIOR = FROZEN`: DenseNet121 classification, SegResNet segmentation, and CoxPH survival workflows validated.
- `PLOTTING_BEHAVIOR = FROZEN`: 52 public figure functions; complete synthetic gallery passes.
- `PUBLIC_API = FROZEN_FOR_V0_1_SERIES`.
- `PUBLIC_REPOSITORY_STRUCTURE = FROZEN`, except for this requested status record.
- `DOCUMENTATION = SYNCHRONIZED_AND_FROZEN`: [Chinese guide](docs/USER_GUIDE_ZH_CN.md), [Codex prompts](docs/CODEX_PROMPTS_ZH_CN.md), [Codex usage contract](docs/CODEX_USAGE.md), and [execution contract](AGENTS.md).

Any future analytical or plotting behavior change requires a new versioned development stage. This record declares the approved baseline; it does not configure remote branch protection.

## Validation

- Python 3.12.10, `pytest`: **18 passed**, two dependency deprecation warnings.
- Quick Start with a separate output root: **PASS**, all ten expected files generated.
- Complete figure gallery: **PASS**, 52 PNGs generated.
- Documentation links, including README Chinese links and local anchors: **PASS** (82 existing links checked).
- `git diff --check`: **PASS**; working tree clean before this status-only addition.
- Validated main [GitHub CI](https://github.com/secdelic/biomed-ml-workflows/actions/runs/33938801842): **SUCCESS**, including tests, Quick Start, and gallery.

Technical validation does not establish study-specific scientific or clinical validity. No analytical, plotting, API, dependency, or existing documentation changes are included in this freeze.

## Historical release integrity

`V0_1_0_STATUS = IMMUTABLE_HISTORICAL_RELEASE`.
The v0.1.0 annotated tag remains `5b82fc7273372e64e16fcf2c4936bd7041bda955`, pointing to commit `fd0543ee8382fa3f3a92dbe51808142bdeb974c4`.
The published v0.1.0 release is unchanged; no tag move, release recreation, or historical force-push is permitted. The post-release Chinese documentation is not retroactively included in that snapshot.

## Nonblocking maintenance warnings

`KNOWN_NONBLOCKING_MAINTENANCE_WARNINGS`:

- Dependency use of deprecated `torch.jit.interface` (two warnings during local tests).
- GitHub Actions reports Node.js 20 deprecation for `actions/checkout@v4` and `actions/setup-python@v5`, with execution forced onto Node.js 24.

Do not upgrade dependencies or actions merely to remove these warnings. Escalate if they cause installation failure, test failure, CI failure, or a security issue.

## Recommended next release

Candidate only: **v0.1.1**, `DOCUMENTATION_AND_USAGE_RELEASE`.

- `ANALYTICAL_BEHAVIOR_CHANGE = NO`
- `PLOTTING_BEHAVIOR_CHANGE = NO`
- `PUBLIC_API_BREAKING_CHANGE = NO`

Create it later only if the Researcher requests an immutable Zenodo-archived software snapshot containing the synchronized Chinese guide and Codex prompts. No v0.1.1 tag, release, or archive is created by this freeze.
