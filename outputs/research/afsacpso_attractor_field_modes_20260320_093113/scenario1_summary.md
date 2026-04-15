# Scenario 1

- Label: ChrismasTerrain2 - Simple
- Terrain: /Users/hust-hwashin621m/Desktop/vietanhpaper-2/data/ChrismasTerrain2.tif
- Danger zones: 0
- Metric: global_fitness_total

## Means

| Algorithm | Mean | Std | Median |
|---|---:|---:|---:|
| AFSACPSO | 1607.2446 | 86.7935 | 1615.2004 |
| SACPSO-Global | 1569.3196 | 101.2003 | 1577.7894 |
| SACPSO-5Subgroup | 1668.7979 | 67.3982 | 1661.2012 |
| SACPSO-PerParticle | 1666.3285 | 76.9908 | 1665.0085 |

## Attractor-Field Pairwise

| Opponent | Mean diff (RR - Opp) | p raw t | p raw Wilcoxon | p Holm Wilcoxon |
|---|---:|---:|---:|---:|
| SACPSO-Global | 37.9250 | 0.124946 | 0.076914 | 0.153829 |
| SACPSO-5Subgroup | -61.5533 | 0.006432 | 0.012096 | 0.036287 |
| SACPSO-PerParticle | -59.0839 | 0.010895 | 0.007050 | 0.028202 |
