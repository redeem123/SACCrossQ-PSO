# Cover Letter -- Swarm and Evolutionary Computation

**To:** The Editor-in-Chief, *Swarm and Evolutionary Computation*  
**Date:** March 9, 2026  
**Re:** Submission of "Rank-Residual Soft Actor-Critic for PSO Parameter Adaptation in UAV Path Planning"

Dear Editor,

I am writing to submit the manuscript titled **"Rank-Residual Soft Actor-Critic for PSO Parameter Adaptation in UAV Path Planning"** for consideration for publication in *Swarm and Evolutionary Computation*.

**Summary of the work.**  
This paper addresses a persistent issue in PSO-based UAV path planning: strong sensitivity to the inertia and acceleration coefficients that govern swarm behavior. The manuscript presents the current repository-consistent form of RRSACPSO, a SAC-guided PSO controller whose central contribution is a **rank-residual control law**. Instead of predicting either one global coefficient triplet or a full per-particle action vector, the controller emits a compact 9-dimensional latent action that is expanded into particle-wise PSO coefficients through:

1. a deterministic progress-dependent base schedule,
2. residual modulation by particle rank, velocity magnitude, and stagnation, and
3. a compact 15-dimensional temporal-summary state with a direct fitness-improvement reward.

The paper is intentionally narrower than earlier drafts. It does **not** claim a retained CrossQ, Transformer, or second validated SAC-side critic contribution. Critic-side branches remain implemented in the repository as research variants, but the manuscript makes only the claim that is currently supported by the codebase and ablation direction: **RankResidualControl is the core validated architectural contribution**.

**Why this is suitable for SWEvo.**  
The paper fits the journal's scope because it contributes a new adaptive control parameterization for swarm intelligence, grounded in an applied UAV path-planning problem. The work is relevant to researchers interested in RL-guided metaheuristics, structured adaptive parameter control, and honest ablation methodology for hybrid optimization systems. In addition to the method itself, the paper offers a useful negative-result boundary: current critic-side SAC variants are treated as exploratory rather than overstated as validated contributions.

**Confirmation.**  
This manuscript has not been published elsewhere and is not under consideration by any other journal. All data and results are original. The author has no competing interests to declare.

I appreciate your consideration.

Respectfully,  
**Viet Anh Ngo**  
School of Mechanical Engineering  
Hanoi University of Science and Technology  
No. 1 Dai Co Viet Road, Hai Ba Trung District, Hanoi, Vietnam  
Email: anh.nv215671@sis.hust.edu.vn  
ORCID: [0009-0002-5429-8554](https://orcid.org/0009-0002-5429-8554)
