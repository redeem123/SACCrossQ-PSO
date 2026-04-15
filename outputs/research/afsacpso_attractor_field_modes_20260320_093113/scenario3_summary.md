# Scenario 3

- Label: ChrismasTerrain2 - Complex
- Terrain: /Users/hust-hwashin621m/Desktop/vietanhpaper-2/data/ChrismasTerrain2.tif
- Danger zones: 10
- Metric: global_fitness_total

## Means

| Algorithm | Mean | Std | Median |
|---|---:|---:|---:|
| AFSACPSO | 1586.7056 | 144.6182 | 1573.0769 |
| SACPSO-Global | 1500.7754 | 112.2676 | 1488.4318 |
| SACPSO-5Subgroup | 1632.8151 | 132.7027 | 1631.9680 |
| SACPSO-PerParticle | 1593.6576 | 162.4803 | 1534.2333 |

## Attractor-Field Pairwise

| Opponent | Mean diff (RR - Opp) | p raw t | p raw Wilcoxon | p Holm Wilcoxon |
|---|---:|---:|---:|---:|
| SACPSO-Global | 85.9302 | 0.021352 | 0.030798 | 0.123191 |
| SACPSO-5Subgroup | -46.1096 | 0.188632 | 0.149929 | 0.299857 |
| SACPSO-PerParticle | -6.9520 | 0.874911 | 0.983590 | 0.983590 |
