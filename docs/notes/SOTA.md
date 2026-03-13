# RRSACPSO SOTA Gap, Novelty Hypothesis, and Current Direction (2026-03-06)

## 1) Recent Related Work (Targeted)

- CrossQ (Bhatt et al., 2024): removes target networks, uses BatchNorm, low UTD=1 for better sample and wall-clock efficiency in continuous control.
  - Source: https://openreview.net/forum?id=PczQtTsTIX
- Scaling Off-Policy RL with Batch + Weight Normalization (Palenicek et al., 2025): studies normalization-driven stability in CrossQ-family off-policy RL.
  - Source: https://arxiv.org/abs/2502.07523
- SAC-based PSO adaptation (Maguire et al., 2024): SAC for PSO parameter control, but not rank-residual particle control with deterministic cross-scale swarm encoding.
  - Source: https://www.mdpi.com/2227-7390/12/22/3481
- RL-guided MOPSO for UAV (QL-MOPSO, 2025): tabular Q-learning + PSO hybrid for multi-objective UAV path planning.
  - Source: https://www.mdpi.com/2073-8994/17/8/1292
- RL-QPSO Net (2025): DRL + QPSO dual-module for robot path planning.
  - Source: https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2024.1464572/full
- Transformer-based SAC for UAV (MATRS, 2025): transformer + SAC in MARL path planning, but not RL-controlled PSO dynamics.
  - Source: https://www.mdpi.com/1424-8220/25/24/7463

## 2) Gap Analysis for This Repo

Observed gap in current benchmark behavior:
- Scenario2 can improve under SAC-layer control, but Scenario1 and Scenario3 still regress against the stored baselines.
- High variance indicates unstable credit assignment and over-sensitive swarm control on harder terrains.
- Existing RL-PSO papers are mostly either:
  - discrete/tabular control,
  - homogeneous swarm control,
  - or RL for planning policy directly rather than PSO parameter residuals.

Technical gap still not covered well by prior hybrids:
- target-free CrossQ SAC + deterministic cross-scale swarm encoding + rank-residual PSO control in one online loop.

## 3) Current Novelty Hypothesis

Hypothesis H6:
A lean CrossQ-SAC controller for rank-residual PSO updates will outperform heavier RL-PSO hybrids if the controller focuses on three things only:
1. informative swarm state compression,
2. low-dimensional but expressive residual control over `w, c1, c2`,
3. critic stabilization through a simple target-free CrossQ training setup instead of auxiliary replay or backup machinery.

## 4) Current Mathematical Direction

Let critics \(Q_1,Q_2\), policy \(\pi\), entropy temperature \(\alpha\), and done flag \(d\).

1) SAC target:
\[
y_t = r_t + \gamma(1-d_t)\left(\min(Q_1',Q_2') - \alpha\log\pi(a'|s')\right)
\]

2) Robust critic loss with asymmetric overestimation penalty:
\[
\mathcal L_Q = \mathbb E\left[\omega(e_t)\,\mathrm{Huber}(e_t)\right],\quad e_t = Q(s_t,a_t)-y_t
\]
\[
\omega(e_t)=
\begin{cases}
\tau, & e_t\ge 0 \\
1-\tau, & e_t<0
\end{cases}
\]

3) Actor objective:
\[
\mathcal L_\pi = \mathbb E\left[\alpha\log\pi(a|s) - \min(Q_1,Q_2)\right]
\]

4) Entropy schedule:
\[
\mathcal H_t^{\star} = \mathcal H_0^{\star}\left(1-\kappa\,p_t\right)
\]
where \(p_t\) is anneal progress.

## 5) What Is Currently Novel in This Repo

Compared with the cited RL-PSO hybrids, this implementation currently combines:
- CrossQ-style target-free SAC backbone,
- deterministic cross-scale swarm state encoding,
- rank-residual mapping from low-dimensional SAC actions to particle-wise PSO control,
- simple fitness-improvement reward inside the online PSO loop.

## 6) Current Status vs Termination Criterion

Not met yet.
- No edited variant is `#1` on all scenarios simultaneously.
- The strongest ablation signal is that `RankResidualControl` matters a lot, while most extra SAC add-ons did not explain the remaining performance gap.
- The next search should stay focused on state representation, residual control geometry, and critic shaping rather than adding more SAC-side machinery.
