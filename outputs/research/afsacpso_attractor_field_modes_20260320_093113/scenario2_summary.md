# Scenario 2

- Label: ChrismasTerrain2 - Medium
- Terrain: /Users/hust-hwashin621m/Desktop/vietanhpaper-2/data/ChrismasTerrain2.tif
- Danger zones: 5
- Metric: global_fitness_total

## Means

| Algorithm | Mean | Std | Median |
|---|---:|---:|---:|
| AFSACPSO | 1571.0133 | 104.6410 | 1573.1556 |
| SACPSO-Global | 1557.7187 | 102.7229 | 1569.0027 |
| SACPSO-5Subgroup | 1647.4451 | 114.3900 | 1644.8797 |
| SACPSO-PerParticle | 1638.0003 | 100.0753 | 1645.6440 |

## Attractor-Field Pairwise

| Opponent | Mean diff (RR - Opp) | p raw t | p raw Wilcoxon | p Holm Wilcoxon |
|---|---:|---:|---:|---:|
| SACPSO-Global | 13.2947 | 0.633426 | 0.869301 | 1.000000 |
| SACPSO-5Subgroup | -76.4318 | 0.018978 | 0.024965 | 0.074894 |
| SACPSO-PerParticle | -66.9870 | 0.006954 | 0.005491 | 0.027455 |
