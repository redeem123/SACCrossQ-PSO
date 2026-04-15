# Figure Notes

## Figure 2

- Source manuscript figure: `paper/main.tex`
- Purpose: illustrative repository-aligned MDP/data-flow view
- Required fidelity points:
  - 9 swarm features
  - 15D temporal-summary state
  - 9D latent action
  - attractor-field expansion to particle-wise `w_i, c_{1,i}, c_{2,i}`
  - relative-improvement reward
  - replay buffer and SAC updates marked as training-only

## External Benchmark Figures

- RL comparison figures should use the `rlbased` group from `run_comparison.m`
- Classical/adaptive figures should use the `others` group from `run_comparison.m`
- After the 2026-04-02 launcher patch, both groups should treat `SACPSO-Global` as the retained SAC row

## Internal Ablation Figures

- Use `VIETANH_ALGO_GROUP=afsacpso-modes`
- Interpret as control-interface evidence, not as proof that attractor-field is the strongest frozen deployment row
