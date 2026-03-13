# RRSACPSO Repository

MATLAB research repository for UAV global path planning with PSO-based optimizers and RL-based parameter adaptation.

The current trustworthy story of this repo is narrower than some legacy filenames and drafts suggest:
- The main proposed method is `RRSACPSO`.
- The strongest validated contribution is the `RankResidualControl` path inside `algorithms/rrsacpso/`.
- Critic-side SAC variants remain research branches, not settled claims.

## Start Here

- Main benchmark entry point: `run_comparison.m`
- Proposed algorithm: `algorithms/rrsacpso/`
- Shared MATLAB infrastructure: `shared/`
- Experiment and analysis helpers: `scripts/`
- Current manuscript workspace: `paper/`
- Notes and archived material: `docs/`

## Repository Map

```text
.
├── algorithms/    MATLAB implementations of RRSACPSO and baselines
├── shared/        Environment, evaluation, statistics, utilities, plotting
├── scripts/       Experiment runners, analysis scripts, figure generation
├── paper/         Current manuscript workspace and paper-side figures
├── docs/          Notes, references, and archived draft material
├── data/          Terrain and scenario assets
├── validation/    External / deployment-side validation assets
├── results/       Generated benchmark/result bundles
├── outputs/       Local runtime outputs (git-ignored)
└── models/        Local checkpoints and logs (git-ignored)
```

## What Is Current

- `algorithms/rrsacpso/` is the active development target.
- `paper/` is the manuscript directory currently present in this workspace.
- `docs/archive_old_draft/` is historical only.
- `docs/notes/` holds working notes such as the SOTA summary, retraining guide, and paper change log.

## Common Commands

```matlab
% Main benchmark driver
run_comparison

% RRSACPSO component tests
run('algorithms/rrsacpso/test_rrsacpso_components.m')
```

## Benchmark Entry Point

- Use only `run_comparison.m` for benchmark execution.
- Treat `scripts/` helpers as internal support tooling around that entry point.

## Artifact Policy

These directories are local working data, not source of truth:
- `outputs/`
- `models/`
- `results/`
- `docs/references/`

Keep code reviews focused on `algorithms/`, `shared/`, `scripts/`, and `paper/`.

## Naming Note

Some files still contain older names such as `RRSACPSO`, `CrossQ`, or earlier paper terminology. Those are legacy traces, not the canonical project description.
