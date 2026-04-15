# AFSACPSO UAV Paper Outline

## Working Title

Soft Actor-Critic with Attractor-Field PSO Parameter Adaptation for UAV Path Planning

## Core Claim

AFSACPSO is a low-dimensional SAC-guided control-interface design for PSO parameter adaptation. Its attractor-field expansion preserves structured particle-wise heterogeneity without direct per-particle control. In the current frozen-policy benchmark, AFSACPSO is competitive with the matched global SAC controller and stronger than the fixed-subgroup and direct per-particle matched variants on easier scenarios.

## Main Hypotheses

1. Control-interface design materially affects frozen SAC-guided PSO deployment under a matched backbone.
2. Low-dimensional latent control plus attractor-field expansion is more effective than fixed subgroup control and more sample-efficient than direct per-particle control.
3. The strongest retained frozen deployment row for external comparison is the matched global SAC controller, not the attractor-field row.

## Required Comparisons

1. Internal same-backbone ablation:
   - AFSACPSO
   - SACPSO-Global
   - SACPSO-5Subgroup
   - SACPSO-PerParticle
2. External RL baselines:
   - RLAMPSO
   - DQN-PSO
   - SAC-SAPSO
   - PPO-PSO
   - MPSORL
   - SACPSO-Global
3. External classical/adaptive baselines:
   - PSO-Standard
   - PSO-LDIW
   - FIPS
   - CLPSO
   - SAEPSO
   - SAEPSO*
   - UAPSO
   - SACPSO-Global

## Success Criteria

1. All external paper tables are reproducible from `run_comparison.m`.
2. The manuscript claim matches the code path:
   - AFSACPSO = proposed design
   - SACPSO-Global = retained external benchmark row
3. All significance claims are backed by matched tests with sample counts and corrected p-values.
4. Every paper figure/table is traceable to a command path and output directory.
