# APEXPSO Research Log

## 2026-03-05 — Iteration R1 (Rank-residual CrossQ-SAC stabilization)

Code focus:
- Rank-residual swarm control
- Cross-scale state encoder
- Huber critic loss
- Overestimation-weighted critic loss
- Critic-disagreement uncertainty penalty
- Entropy uncertainty coupling

Run:
- `outputs/research/2026-03-05_18-13-36`
- 3 runs, 600 iterations, parallel workers=3

Outcome:
- Scenario-level wins remained fragmented.
- The main bottleneck was still Scenario3 mean fitness.

---

## 2026-03-05 — Iteration R2 (Fast-update SAC schedule)

Code focus:
- `batchSize=64`
- `gradientStepsPerTraining=2`
- `warmupPeriod=64`

Run:
- `outputs/research/2026-03-05_18-44-37`
- 3 runs, 600 iterations, parallel workers=3

Outcome:
- Faster critic adaptation helped some Scenario3 runs.
- Scenario1 and Scenario2 regressed, so fast-update was not adopted as the default schedule.

---

## 2026-03-06 — Iteration R3 (50-run LOO with resumable checkpoints)

Run:
- `outputs/research/loo50_resumable_2026-03-06_06-46-48`
- 50 runs, 600 iterations, fair-seed control, 14-worker resumable execution

Infrastructure outcome:
- Added per-run checkpoint persistence to `apexpso_checkpoint.csv`
- Added resume support via `APEXPSO_RESEARCH_RESUME_DIR`
- MATLAB `parfor` workers still hit native crashes on macOS ARM, but completed runs were preserved and the study finished after resume

50-run LOO outcome retained in the repo:
- `APEX-LOO-NoRankResidual` was the only clearly harmful removal
- `APEX-LOO-WithEntropyUncertainty` remained the only optional add-back worth preserving for retest
- no edited variant achieved `#1` mean fitness against stored baselines

Baseline comparison:
- Stored baseline best means remain:
- Scenario1: `1495.9562`
- Scenario2: `1551.0854`
- Scenario3: `1505.9552`

Component decisions from LOO:
- Keep: `RankResidualControl`
- Keep as base defaults: `CrossScaleState`, `Curiosity`, `HuberCriticLoss`, `OverestimationPenalty`, `ActorUncertaintyPenalty`
- Retest as focused candidate: `EntropyUncertaintyCoupling`
- Removed from the repository after this study: experimental replay/backup/prior/normalization branches and weak ablation branches that did not justify their complexity

Code decision:
- Reduced the shipped `APEXPSO_Config` stack to the surviving components only
- Replaced the old research family with a minimal `V7` family centered on the reduced base, the cross-scale ablation, and the entropy-coupling probe
- Current retained LOO family: `APEX-LOO-Full`, `APEX-LOO-NoRankResidual`, `APEX-LOO-WithEntropyUncertainty`, `APEX-LOO-NoCrossScaleState`
