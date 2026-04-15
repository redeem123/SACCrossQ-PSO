# Paper Review: CrossQ Soft Actor‑Critic for PSO Parameter Adaptation in UAV Path Planning **Target journal:** Swarm and Evolutionary Computation **Overall verdict: 🟡 Major Revision Required** --- ## Scores at a Glance | Criterion | Score | Notes |
|---|---|---|
| Novelty | 7/10 | Strong combination of known components; no single radical new idea |
| Technical Soundness | 6/10 | Multiple broken references & missing figures undermine trust |
| Writing Quality | 7/10 | Generally clear; some sections over-explained, others under-explained |
| Experimental Rigor | 6/10 | Good breadth but SAEPSO wins most head-to-head comparisons |
| Presentation | 5/10 | Placeholder metadata, broken refs, missing figures |
| Reproducibility | 4/10 | Key hyperparameters defined as macros but **never assigned values** |
| Significance | 6/10 | Useful engineering contribution; limited theoretical depth | --- ## 🔴 Critical Issues (Must Fix) ### 1. Placeholder Author Metadata
Lines 73–81 still contain **template placeholders**:
- `Bob Author`, `Christine Author`, `Derek Author`
- `Affiliation, department, city, postcode, country`
- `corresponding.author@email.example` > [!CAUTION]
> This is an instant desk-reject at any venue. Replace with real authors and affiliations. ### 2. Broken/Undefined Figure References
Three figures are **referenced in the text but never defined** as `\begin{figure}` environments: | Reference | Line | Status |
|---|---|---|
| `\ref{fig:rl_trajectories}` | 622 | ❌ **Undefined** — will render as "??" |
| `\ref{fig:training_details}` | 631 | ❌ **Undefined** — will render as "??" |
| `\ref{fig:train_rewards}` | 656 | ❌ **Undefined** — will render as "??" | These are critical for the ablation analysis. You have `Training_EpisodeRewards_Comparison.pdf` in the figures folder — likely intended for `fig:train_rewards` — but the LaTeX `\figure` block is missing. ### 3. Uncited Bibliography Entry
`\bibitem{Vaswani2017Attention}` (line 1073) is **never cited** in the text. Either cite it (e.g., when describing the Transformer encoder in §4.1) or remove it. ### 4. Hyperparameter Values Never Specified
The paper defines numerous symbol macros (`\symStatedim`, `\symWindowsize`, `\symFeaturecount`, `\symHiddendimA`, etc.) but **never assigns them concrete values**. Only `\valPopsize=40` and `\valMaxIters=600` are set. A reader cannot reproduce the work without knowing:
- Temporal window size $W$
- Feature count $N_f$
- Hidden dimensions $H_a$, $H_b$
- Batch size $B$, learning rate $\eta$
- Replay buffer capacity $C_{rb}$
- Number of training episodes $E$
- Retained reward definition and decision rule
- Penalty thresholds $\rho_{stag}, \rho_{coll}, \rho_{div}, \theta_{min}$ > [!IMPORTANT]
> Add a **Hyperparameter Table** listing all values. This is standard for SWEVO papers. --- ## 🟡 Major Concerns ### 5. AFSACPSO Does Not Beat SAEPSO
The paper's own results show that **SAEPSO wins in all 3 scenarios** for traditional comparisons (Table 3), with lower mean fitness AND lower standard deviation. The narrative tries to redirect attention to inference speed, but:
- The 2–3× speed advantage only applies to AFSACPSO, not the online variant
- For a journal like SWEVO, **optimization quality trumps runtime**
- The paper should be more honest about this limitation rather than burying it **Recommendation:** Frame the contribution more carefully. AFSACPSO is best-in-class *among RL-based methods*, competitive with (but not superior to) hand-crafted adaptive PSO. The real value is the *learning framework*, not beating SAEPSO. ### 6. Ablation Results Are Inconclusive
Table 4 shows:
- **"No Attention" wins in Scenarios 2 and 3** on mean fitness
- Only **2 out of 6 comparisons** show statistical significance ($^{*}$ or $^{**}$)
- The explanation for Scenario 3 ("artifact of fitness function") is valid but weakens the claim The paper correctly notes the variance differences, but the ablation doesn't convincingly prove the Transformer is essential. Consider running more than 30 runs to increase statistical power, or testing on more diverse scenarios. ### 7. Only 3 Benchmark Scenarios
Three scenarios is thin for a SWEVO paper. Most comparable works test on 10+ benchmark functions or 5+ diverse environments. The current scenarios seem to vary only in obstacle density. Add scenarios with:
- Different terrain types (steep mountains, canyons, flat plains)
- Different start/goal configurations
- Dynamic no-fly zones ### 8. Missing Convergence Theory
The paper has no convergence guarantee or theoretical analysis. While not strictly required for an empirical paper, SWEVO often expects at least a brief complexity/convergence discussion for new PSO variants. --- ## 🟢 Strengths 1. **Well-motivated gap analysis.** The three-gap framework (temporal encoding, target networks, reward sparsity) is clearly articulated and well-supported by the literature review. 2. **Strong Related Work section.** Comprehensive coverage of RL-PSO methods with a useful comparison table (Table 1). The categorization into value-based vs. policy gradient vs. attention-based is clean. 3. **Thoughtful ablation design.** Testing each retained component independently (No Attention, No CrossQ) is methodologically sound. 4. **Honest discussion of limitations** (§6.2). The paper acknowledges training cost, static environments, and transfer limitations. 5. **Good computational efficiency analysis** (Table 5). Reporting per-scenario timing with S3/S1 ratios is a nice touch. 6. **Friedman test** across all algorithms demonstrates statistical rigor beyond simple pairwise t-tests. --- ## 📝 Writing & Presentation Suggestions ### Structure
- The **Results Summary** (§5.4) largely repeats the Discussion (§6). Merge or eliminate one.
- The conclusion (§7) also repeats the same key findings for a third time. Trim to 1 paragraph.
- §5.3 (Training Efficiency) text references `Figure~\ref{fig:training_details}` and `Figure~\ref{fig:train_rewards}` — these are **different figures** but the text blurs them together. Clarify. ### Terminology Consistency
- The paper alternates between "AFSACPSO" and "AFSACPSO" without always being clear which is meant. Establish this distinction clearly at first mention.
- "Online learning" vs " inference" — make the distinction sharper. ### Minor Issues
- Line 30: `\hfuzz=2pt` and `\hbadness=10000` suppress overfull hbox warnings. This is fine for drafting but remove before submission — reviewers will notice.
- Line 15: Custom `\FloatBarrier` is a no-op (`\par\vskip\z@\noindent\mbox{}`). Use the `placeins` package properly or remove.
- Line 107: The Meng 2025 citation about multi-UAV multitasking feels shoehorned into the introduction. It's tangential to the paper's single-UAV focus.
- Table captions are excessively long (Table 5 caption is 5 lines). Shorten to 1–2 sentences and move details to the text. --- ## 📊 Missing Content for SWEVO Standard | Expected Element | Present? |
|---|---|
| Hyperparameter table | ❌ |
| Parameter sensitivity analysis | ❌ |
| Convergence plots for traditional PSO variants | ❌ (only RL-based shown) |
| Box plots or violin plots | ❌ |
| Wilcoxon rank-sum tests (non-parametric) | ❌ (only paired t-tests) |
| CEC benchmark functions | ❌ |
| Algorithm complexity analysis | ❌ |
| Source code availability statement | ❌ | --- ## Verdict Summary The paper presents a **technically sound engineering contribution** — combining CrossQ-SAC with a Transformer encoder and a direct fitness-improvement reward for PSO parameter control is a reasonable and well-executed idea. The writing is generally clear, and the experimental setup is decent. However, the paper needs **major revisions** before it's ready for SWEVO: 1. **Fix all broken references and placeholder metadata** — these are desk-reject issues
2. **Add a hyperparameter table** — reproducibility is non-negotiable
3. **Reframe the narrative** — don't oversell beating traditional methods when SAEPSO wins
4. **Expand benchmarks** — 3 scenarios is insufficient for the venue
5. **Add missing standard analyses** — sensitivity study, Wilcoxon tests, convergence plots for all methods The core idea is publishable. The execution gaps need to be addressed.
