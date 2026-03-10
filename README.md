# RRSACPSO Repository

MATLAB research repository for UAV global path planning with PSO-based optimizers and RL-based parameter adaptation.

The current trustworthy story of this repo is narrower than some legacy filenames and drafts suggest:
- The main proposed method is `RRSACPSO`.
- The strongest validated contribution is the `RankResidualControl` path inside `algorithms/rrsacpso/`.
- Critic-side SAC variants remain research branches, not settled claims.

## Start Here

- Main comparison entry point: `run_comparison.m`
- Proposed algorithm: `algorithms/rrsacpso/`
- Shared infrastructure: `shared/`
- Experiment and analysis helpers: `scripts/`

## Repository Map

```text
.
├── algorithms/    MATLAB implementations of RRSACPSO and baselines
├── shared/        Environment, evaluation, statistics, utilities, plotting
├── scripts/       Experiment runners, analysis scripts, figure generation
├── docs/          Notes, references, archived paper material
├── paper/         Archived Elsevier-era paper assets and submission metadata
├── sage_latex_template_4_unzipped/
│                  Active SAGE manuscript directory
├── data/          Terrain and scenario assets
├── validation/    External / deployment-side validation assets
├── outputs/       Local runtime outputs (git-ignored)
└── models/        Local checkpoints and logs (git-ignored)
```

## What Is Current

- `algorithms/rrsacpso/` is the active development target.
- `sage_latex_template_4_unzipped/` is the active manuscript source.
- `paper/` should be treated as archived submission material unless explicitly revived.
- `docs/archive_old_draft/` is historical only.

## Common Commands

```matlab
% Main benchmark driver
run_comparison

% RRSACPSO component tests
run('algorithms/rrsacpso/test_rrsacpso_components.m')
```

## Benchmark Entry Point

- Use only `run_comparison.m` for benchmark execution.
- There is no second benchmark launcher anymore; scenario and group selection should be driven through `run_comparison.m` configuration and environment overrides.

## Artifact Policy

These directories are local working data, not source of truth:
- `outputs/`
- `models/`
- `docs/references/`

Keep code reviews focused on `algorithms/`, `shared/`, `scripts/`, and the active manuscript.

## Naming Note

Some files still contain older names such as `RRSACPSO`, `CrossQ`, or earlier paper terminology. Those are legacy traces, not the canonical project description.
