# CQSAC-PSO: Cross-Q Soft Actor-Critic Particle Swarm Optimization for UAV Path Planning

> **Paper under review.** This repository contains the official MATLAB implementation of **CQSAC-PSO** (CrossQ Soft Actor-Critic PSO), an RL-augmented PSO framework for 3-D UAV global path planning on real terrain data.

---

## Table of Contents
1. [Overview](#overview)
2. [Algorithm Architecture](#algorithm-architecture)
3. [Repository Structure](#repository-structure)
4. [Requirements](#requirements)
5. [Quick Start](#quick-start)
6. [Reproducing Paper Results](#reproducing-paper-results)
7. [Model Artifacts](#model-artifacts)
8. [Benchmark Algorithms](#benchmark-algorithms)
9. [Ablation Study](#ablation-study)
10. [Results](#results)
11. [Citation](#citation)

---

## Overview

CQSAC-PSO integrates a **CrossQ Soft Actor-Critic (SAC) agent** into the Particle Swarm Optimization (PSO) loop to adaptively control per-particle inertia weight (ω) and acceleration coefficients (c₁, c₂) at each iteration. This replaces the conventional hand-tuned or simple linearly-decaying parameters with a learned policy that observes the swarm's collective state and outputs personalised control signals for every particle.

**Key novelties:**
- 🔀 **Per-particle parameter control** — the SAC actor outputs an action vector for *each* particle independently, rather than applying a single global update
- 🤖 **CrossQ critic** — uses cross-batch normalisation to stabilise off-policy learning inside the optimisation loop without a separate pre-training phase
- 🎯 **Multi-objective reward** — balances path length, terrain clearance, obstacle avoidance, and climb penalty in a single shaped scalar
- 🔭 **Transformer state encoder** — self-attention over the swarm's particle positions captures inter-particle dynamics as a compact context vector

The method is benchmarked against **8 PSO variants** (PSO-LDIW, RLAMPSO, DQN-PSO, MPSORL, SAC-SAPSO, PPO-PSO, and others) across **3 terrain scenarios** with 30 independent runs per condition.

---

## Algorithm Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    CQSAC-PSO Loop                        │
│                                                          │
│  Swarm State ──► Transformer Encoder ──► SAC Actor      │
│                       ↓                       ↓          │
│              Context vector         Per-particle (ω,c₁,c₂)
│                                               ↓          │
│              PSO Velocity & Position Update              │
│                       ↓                                  │
│              Fitness Evaluation (terrain + obstacles)    │
│                       ↓                                  │
│              CrossQ Critic Update (off-policy)           │
└─────────────────────────────────────────────────────────┘
```

**Key files:**
| Component | File |
|---|---|
| SAC Agent (actor + CrossQ critics) | `algorithms/apexpso/APEXPSO_Agent.m` |
| Cross-scale state encoder | `algorithms/apexpso/CrossScaleStateEncoder.m` |
| Reward function | `algorithms/apexpso/calculateMultiObjectiveReward.m` |
| Main PSO loop | `algorithms/apexpso/globalPathPlanningAPEXPSO.m` |
| Fitness evaluation | `shared/evaluation/evaluatePathFitness.m` |

---

## Repository Structure

```
.
├── run_comparison.m          ← ENTRY POINT: configure and launch experiments
│
├── algorithms/               ← All PSO algorithm implementations
│   ├── apexpso/              ← CQSAC-PSO (proposed method) ★
│   ├── neural_guided_pso/    ← Neural-Guided PSO baseline
│   ├── rlampso/              ← DDPG-based RLAMPSO
│   ├── dqn_pso/              ← DQN-PSO baseline
│   ├── mpsorl/               ← Multi-strategy RL-PSO
│   ├── ppopso/               ← PPO-PSO baseline
│   ├── svpso/                ← SAC-SAPSO baseline
│   ├── pso_tvac/             ← PSO-LDIW (classical baseline)
│   └── ...                   ← Other PSO variants
│
├── shared/                   ← Shared infrastructure (used by all algorithms)
│   ├── environment/          ← Terrain loading, scenario definitions
│   ├── evaluation/           ← Fitness function, collision checking
│   ├── neural_networks/      ← Shared NN utilities
│   ├── path_planning/        ← Waypoint interpolation, path utilities
│   ├── statistics/           ← Wilcoxon tests, Friedman ranking, CSV export
│   ├── utilities/            ← PSO helpers, GPU management, logging
│   └── visualization/        ← 3D path plots, convergence curves
│
├── scripts/                  ← Standalone analysis and plotting scripts
│   ├── compare_algorithms.m
│   ├── visualize_training_comparison.m
│   ├── generate_rl_paper_figures.m
│   ├── friedman_rankings.py
│   └── generate_hybrid_insets.py
│
├── paper/                    ← Paper source (Elsevier article class)
│   ├── main.tex              ← Master LaTeX source
│   ├── main.pdf              ← Compiled PDF (latest version)
│   ├── main_backup.tex       ← Pre-edit backup
│   ├── figures/              ← All figures used in the paper
│   ├── Background_RelatedWork.tex  ← Related work section draft
│   └── paper_review.md       ← Review notes
│
├── docs/                     ← Documentation and reference material
│   ├── notes/                ← Dev notes, bug fixes, retraining guides
│   ├── references/           ← Reference PDFs (cited papers)
│   └── archive_old_draft/    ← Deprecated paper draft (do not use)
│
├── data/                     ← Terrain elevation data
│   └── ChrismasTerrain2.tif  ← GeoTIFF DEM (100×100 grid, 3 scenarios)
│
├── models/                   ← Model artifacts (Git-ignored, ~88 MB each)
│   ├── APEXPSO/              ← CQSAC-PSO global logs/checkpoints
│   ├── RLAMPSO/              ← DDPG pretrained weights
│   └── DQN_PSO/              ← DQN pretrained Q-tables
│
├── outputs/                  ← All runtime outputs (Git-ignored)
│   ├── results/              ← Per-algorithm result structs (.mat)
│   ├── data/statistics/      ← CSV tables: T-test, Friedman rankings
│   ├── figures/              ← Generated plots
│   │   ├── convergence/
│   │   ├── paths/
│   │   ├── training/
│   │   ├── statistics/
│   │   └── environment/
│   └── logs/                 ← MATLAB diary logs per run
│
└── tools/                    ← Local binaries (not part of algorithm)
    └── poppler/              ← PDF utilities (pdftotext, pdfimages, etc.)
```

---

## Requirements

### MATLAB
- **MATLAB R2022b or later** (required for `dlarray`, `dlnetwork`, parallel pool)
- Toolboxes: **Deep Learning Toolbox**, **Statistics and Machine Learning Toolbox**, **Parallel Computing Toolbox**

### Python (analysis scripts only)
```bash
pip install -r requirements.txt
# or: conda env create -f environment.yml
```

### Terrain Data
The `data/ChrismasTerrain2.tif` GeoTIFF is included in the repo. Scenario obstacle densities are set in `run_comparison.m`:
- **Scenario 0** — flat terrain, no obstacles
- **Scenario 5** — moderate density
- **Scenario 10** — high obstacle density

---

## Quick Start

```matlab
% 1. Open MATLAB, set CWD to this repository root
cd('/path/to/vietanhpaper-2')

% 2. Run the comparison (default: RL-based algorithms, 30 runs, parallel)
run_comparison

% 3. Results land in:
%    outputs/results/   — raw .mat files
%    outputs/figures/   — PDF plots
%    outputs/logs/      — run logs
```

**To change the algorithm group**, edit the `algorithms` assignment near line 307 of `run_comparison.m`:
```matlab
algorithms = rlBased;       % RL-augmented PSO variants (default)
algorithms = others;         % Classical PSO baselines
algorithms = ablationRun;    % CQSAC-PSO ablation variants
```

---

## Reproducing Paper Results

### Table 2 — Main Comparison (30 runs × 3 scenarios)

```matlab
% In run_comparison.m, set:
%   config.general.executionMode = 'parallel';
%   config.general.numRuns = 30;
%   algorithms = rlBased;        % includes CQSAC-PSO and all baselines
run_comparison
```

### Table 3 — Ablation Study (Global Mode, Action Size = 3)

```matlab
% Option A: Reuse generated global ablation outputs (fast, ~30 min)
algorithms = ablationRun;

% Option B: Retrain from scratch (slow, ~8h per variant)
algorithms = ablationTrain;
```

### Statistical Tests
After running, Wilcoxon rank-sum T-tests and Friedman rankings are auto-computed and saved to `outputs/data/statistics/`.

---

## Model Artifacts

Model artifacts are **not tracked in Git** (too large). Place generated logs in `models/`:

| Model | Mode | File |
|---|---|---|
| CQSAC-PSO (baseline) | global | `models/APEXPSO/apexpso_global_log.mat` |
| Ablation: No Attention | global | `models/APEXPSO/apexpso_abl_noatt_global_log.mat` |
| Ablation: No CrossQ | global | `models/APEXPSO/apexpso_abl_nocrossq_global_log.mat` |
| Ablation: Simple Reward | global | `models/APEXPSO/apexpso_abl_simplerwd_global_log.mat` |

> **macOS users:** GPU-trained models are automatically converted to CPU arrays on load. No manual steps needed.

---

## Benchmark Algorithms

| Algorithm | Category | Reference |
|---|---|---|
| **CQSAC-PSO** | RL-PSO (proposed) | This work |
| RLAMPSO (Global) | RL-PSO | DDPG-based |
| DQN-PSO (Global) | RL-PSO | Q-learning |
| MPSORL | RL-PSO | Multi-strategy |
| SAC-SAPSO | RL-PSO | SAC baseline |
| PPO-PSO | RL-PSO | PPO-based |
| PSO-LDIW | Classical PSO | Kennedy & Eberhart 1995 |

All algorithms use the same: population size = 40, max iterations = 1500, terrain grid = 100×100×100, identical fitness function and collision constraint (`hardCollisionConstraint=true`).

---

## Ablation Study

Three ablation variants isolate contributions of CQSAC-PSO's novel components:

| Variant | What is removed | Expected effect |
|---|---|---|
| No Attention | Transformer encoder → simple MLP state | Loses inter-particle context |
| No CrossQ | Standard critic → CrossQ critic | Less stable off-policy learning |
| Simple Reward | Multi-objective reward → path-length only | Ignores safety and smoothness |

---

## Results

Key results (Scenario 2, 30 runs, mean ± std path length):

| Algorithm | Path Length (m) | Rank |
|---|---|---|
| **CQSAC-PSO (Per-Particle)** | **lowest** | **1** |
| RLAMPSO | — | 2 |
| DQN-PSO | — | 3 |
| PSO-LDIW | — | 7 |

Full tables, convergence curves, and trajectory visualisations are in `paper/main.pdf`.

---

## Citation

```bibtex
@article{cqsacpso2026,
  title   = {{CQSAC-PSO}: Cross-Q Soft Actor-Critic Particle Swarm Optimization
             for 3-D UAV Global Path Planning},
  author  = {[Authors]},
  journal = {[Journal]},
  year    = {2026},
  note    = {Under review}
}
```

---

## License

Source code: MIT License.  
Pretrained model weights: released for academic research only.
