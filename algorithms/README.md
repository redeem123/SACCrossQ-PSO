# algorithms/

MATLAB implementations of the proposed method and benchmark PSO variants.

## Current Focus

- `apexpso/` is the main research algorithm.
- The current RRSACPSO repo story centers on rank-residual parameter adaptation, not on legacy CrossQ-first wording.
- `RLAM-OPENSOURSE/` is third-party reference code, not part of the primary MATLAB pipeline.

## Directory Map

| Directory | Role |
|---|---|
| `apexpso/` | Proposed RRSACPSO method and component tests |
| `baseline_pso/` | Standard PSO baseline |
| `clpso/`, `dms_pso/`, `fips/`, `hpso_tvac/`, `lips/`, `pso_tvac/`, `uapso/`, `saepso/` | Classical / adaptive PSO baselines |
| `dqn_pso/`, `mpsorl/`, `ppopso/`, `svpso/`, `rlampso/` | RL-based baselines |
| `annpso/`, `igpso/`, `neural_guided_pso/`, `rlnnpso/` | Neural / hybrid baselines |
| `RLAM-OPENSOURSE/` | Vendored external reference implementation |

## Calling Convention

Cross-algorithm dispatch goes through:
- `algorithms/callGlobalPlanningAlgorithm.m`

The most important RRSACPSO entry points are:
- `algorithms/apexpso/globalPathPlanningAPEXPSO.m`
- `algorithms/apexpso/APEXPSO_Agent.m`
- `algorithms/apexpso/APEXPSO_Config.m`

## Organization Rule

When adding new work:
- put algorithm logic in the algorithm folder
- put shared helpers in `shared/`
- put experiment orchestration in `scripts/`
- keep logs and generated artifacts out of `algorithms/`
