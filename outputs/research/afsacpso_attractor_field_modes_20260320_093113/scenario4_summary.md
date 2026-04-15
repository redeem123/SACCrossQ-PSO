# Scenario 4

- Label: terrainStruct_c_100 - Simple
- Terrain: /Users/hust-hwashin621m/Desktop/vietanhpaper-2/data/terrainStruct_c_100.mat
- Danger zones: 0
- Metric: global_fitness_total

## Means

| Algorithm | Mean | Std | Median |
|---|---:|---:|---:|
| AFSACPSO | 12341.1465 | 172.9646 | 12297.6980 |
| SACPSO-Global | 12325.7150 | 170.0541 | 12262.2649 |
| SACPSO-5Subgroup | 12392.5301 | 185.6024 | 12322.2474 |
| SACPSO-PerParticle | 12462.7766 | 267.7009 | 12440.6359 |

## Attractor-Field Pairwise

| Opponent | Mean diff (RR - Opp) | p raw t | p raw Wilcoxon | p Holm Wilcoxon |
|---|---:|---:|---:|---:|
| SACPSO-Global | 15.4314 | 0.736542 | 0.967187 | 0.967187 |
| SACPSO-5Subgroup | -51.3836 | 0.204106 | 0.073543 | 0.351468 |
| SACPSO-PerParticle | -121.6301 | 0.055680 | 0.070294 | 0.351468 |
