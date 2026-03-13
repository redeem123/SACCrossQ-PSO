# shared/

Shared infrastructure used by all algorithm implementations.
These modules are algorithm-agnostic; algorithms depend on them but not on each other.

## Module Map

| Module | Responsibility |
|---|---|
| `environment/` | DEM loading (`getTerrainHeight`), scenario construction, obstacle placement |
| `evaluation/` | Path fitness function, collision detection, penalty components |
| `neural_networks/` | Shared NN builders (actor, critic networks), GPU utils |
| `path_planning/` | Waypoint interpolation, path smoothing, parametrisation |
| `statistics/` | Wilcoxon tests, Friedman ranking, CSV/table export |
| `utilities/` | PSO core helpers, parallel pool, training loggers, path resolution |
| `visualization/` | 3D path plots, convergence curves, comparison charts |

## Design Rules

1. **No algorithm cross-dependencies** — `shared/` can call nothing from `algorithms/`
2. **Single responsibility** — each `.m` file in `shared/` does one well-defined thing
3. **Documented interfaces** — every function has a `% INPUTS / OUTPUTS` block header
