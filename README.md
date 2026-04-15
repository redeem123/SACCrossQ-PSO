# AFSACPSO Repository

MATLAB research repository for UAV global path planning with PSO-based optimizers and RL-based parameter adaptation.

The current trustworthy story of this repo is narrower than some legacy filenames and drafts suggest:
- The main proposed method is `AFSACPSO`.
- The strongest validated contribution is the `AttractorFieldControl` path inside `algorithms/afsacpso/`.
- Critic-side SAC variants remain research branches, not settled claims.

## Start Here

- Main benchmark entry point: `run_comparison.m`
- Proposed algorithm: `algorithms/afsacpso/`
- Shared MATLAB infrastructure: `shared/`
- Experiment and analysis helpers: `scripts/`
- Current manuscript workspace: `paper/`
- Notes and archived material: `docs/`

## Repository Map

```text
.
├── algorithms/    MATLAB implementations of AFSACPSO and baselines
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

- `algorithms/afsacpso/` is the active development target.
- `paper/` is the manuscript directory currently present in this workspace.
- `docs/archive_old_draft/` is historical only.
- `docs/notes/` holds working notes such as the SOTA summary, retraining guide, and paper change log.

## Common Commands

```matlab
% Main benchmark driver
run_comparison

% AFSACPSO component tests
run('algorithms/afsacpso/test_afsacpso_components.m')
```

## Benchmark Entry Point

- Use only `run_comparison.m` for benchmark execution.
- Treat `scripts/` helpers as internal support tooling around that entry point.
- Paper-aligned external comparison groups now use `SACPSO-Global` as the retained frozen SAC row in both `rlbased` and `others`.
- Use `VIETANH_ALGO_GROUP=afsacpso` or `VIETANH_ALGO_GROUP=afsacpso-modes` when you specifically want the attractor-field method or its same-backbone ablation.

## Artifact Policy

These directories are local working data, not source of truth:
- `outputs/`
- `models/`
- `results/`
- `docs/references/`

Keep code reviews focused on `algorithms/`, `shared/`, `scripts/`, and `paper/`.

## Naming Note

Some files still contain older names such as `AFSACPSO`, `CrossQ`, or earlier paper terminology. Those are legacy traces, not the canonical project description.
