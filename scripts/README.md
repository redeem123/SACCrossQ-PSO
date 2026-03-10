# scripts/

Experiment runners, analysis helpers, and figure-generation utilities.

This directory is still flat, but it should be read in four groups:

| Group | Current files |
|---|---|
| Experiment runners | `compare_algorithms.m`, `run_rrsacpso_research_cycle.m`, `run_apex_full_loop_iteration.m`, `run_rrsacpso_autosearch_loop.m` |
| Analysis / synchronization | `sync_ablation_tables_to_tex.m`, `friedman_rankings.py` |
| Figure generation | `generate_rl_paper_figures.m`, `generate_hybrid_insets.py`, `generate_others_legend.m`, `visualize_training_comparison.m` |
| Automation / support | `run_rrsacpso_autosearch_daemon.sh`, `latex_build.ps1` |

## Recommended Use

- Use `run_comparison.m` from the repo root for all benchmarks.
- Treat `compare_algorithms.m` as the internal engine that `run_comparison.m` calls, not as a separate user-facing benchmark entry point.
- Use `run_rrsacpso_research_cycle.m` for RRSACPSO component and ablation work.
- Treat PDFs and caches in this directory as generated artifacts, not source.
