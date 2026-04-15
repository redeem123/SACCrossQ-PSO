# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Summary

MATLAB research repository for UAV global path planning. The proposed method is **AFSACPSO** (Attractor-Field Soft Actor-Critic PSO) — an online RL agent that adapts per-particle PSO parameters via a 9D SAC action space with attractor-field geometric modulation. Target journal: Swarm and Evolutionary Computation (Elsevier).

## Commands

### Run benchmarks (the only entry point)
```bash
# From repo root — MATLAB must be on PATH
/Applications/MATLAB_R2024a.app/bin/matlab -nodisplay -nosplash -r "run_comparison; exit"

# With environment overrides
VIETANH_ALGO_GROUP=others VIETANH_NUM_RUNS=10 VIETANH_SCENARIO_FILTER="Simple" \
  /Applications/MATLAB_R2024a.app/bin/matlab -nodisplay -nosplash -r "run_comparison; exit"
```

### Environment variables for `run_comparison.m`
| Variable | Default | Values |
|---|---|---|
| `VIETANH_ALGO_GROUP` | `rlbased` | `rlbased`, `others`, `afsacpso`, `afsacpso-modes`, `ablation-sac`, `ablation-sac-features`, `all` |
| `VIETANH_NUM_RUNS` | `30` | integer |
| `VIETANH_POP_SIZE` | `100` | integer |
| `VIETANH_MAX_ITERATIONS` | `1000` | integer |
| `VIETANH_SCENARIO_FILTER` | (all 4) | substring match on scenario label |
| `VIETANH_EXECUTION_MODE` | `parallel` | `parallel`, `serial` |

### Run component tests
```matlab
run('algorithms/afsacpso/test_afsacpso_components.m')
```

### Build the paper
```bash
cd paper && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

### Clear stale resume cache (required before fresh benchmark runs)
```bash
rm -rf outputs/results/resume_checkpoints/
```

## Architecture

### Control flow
```
run_comparison.m
  → load_experiment_config()          # reads env vars, builds config struct
  → scripts/compare_algorithms.m      # benchmark engine
      → generateFixedTerrain()        # from shared/environment/
      → executeScenario()             # parfor over runs
          → callGlobalPlanningAlgorithm()   # dispatcher in algorithms/
              → globalPathPlanningAFSACPSO()  or  directGlobalPlanning()
      → performTTest(), ComparePSOVariants(), plot functions
```

### Key directories
- **`algorithms/`** — Each algorithm in its own subdirectory. All dispatched through `callGlobalPlanningAlgorithm.m`.
- **`shared/`** — Algorithm-agnostic infrastructure (environment, evaluation, neural nets, statistics, visualization). **Design rule: `shared/` never imports from `algorithms/`.**
- **`scripts/`** — Benchmark engine (`compare_algorithms.m`) and analysis helpers. Treat as internal tooling behind `run_comparison.m`.
- **`paper/`** — LaTeX manuscript (`elsarticle` class, `main.tex`).
- **`data/`** — Terrain assets (`ChrismasTerrain2.tif`, `terrainStruct_c_100.mat`).

### AFSACPSO internals (`algorithms/afsacpso/`)
- `AFSACPSO_Config.m` — All hyperparameters. Production mode is `'online'` with `paramMode = 'attractor-field'`.
- `AFSACPSO_Agent.m` — SAC agent (actor, twin critics, auto-alpha).
- `globalPathPlanningAFSACPSO.m` — PSO loop + online SAC training + post-PSO Nelder-Mead local search.
- `convertActionToPerParticleParams.m` — Maps 9D action → per-particle [w, c1, c2] via constrained bias + geometric modulation.
- `extractRichFeatures.m` — Builds 15D state from swarm statistics.

### Fitness evaluation (`shared/evaluation/`)
- `evaluatePathFitness.m` — Total cost = path length + height penalty + smoothness + safety.
- Height penalty dominates (~50% of total cost); penalizes absolute z, not altitude above terrain.

### Deterministic seeding
Seeds are `run * 1000 + scenarioIdx * 100 + algorithm.seedOffset`. `parfor` with fixed workers produces deterministic results.

## Critical Invariants

1. **Single entry point**: Always benchmark via `run_comparison.m` from repo root. Never call `compare_algorithms.m` directly.
2. **`shared/` independence**: Never add calls from `shared/` to `algorithms/`. The dependency flows one way: algorithms → shared.
3. **`paramMode` must be `'attractor-field'`**: Do not override to `'attractor-field'` or add LDIW/TVAC schedules — the novel contribution is pure SAC-driven parameter control with no schedule.
4. **Delete resume checkpoints** (`outputs/results/resume_checkpoints/`) before fresh runs to avoid stale cached results.
5. **GPU is force-disabled** on macOS in `AFSACPSO_Config.m` — all computation is CPU-only.
6. **`configOverrides` struct** in algorithm registry entries allows per-algorithm config changes without editing `AFSACPSO_Config.m`.
