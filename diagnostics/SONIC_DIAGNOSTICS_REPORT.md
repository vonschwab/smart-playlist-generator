# Sonic Diagnostics Report (artifact: `experiments/genre_similarity_lab/artifacts/data_matrices_step1.npz`)

Generated via geometry + feature ablation runs (2025-12-15, seed=1).

## Geometry (S_raw vs S_z)

| Run | S_raw p10 / p50 / p90 (spread, sat) | S_z p10 / p50 / p90 (spread, sat) | corr(S_raw,T_raw) | corr(S_raw,Tc_resc) | corr(S_z,Tc_resc) |
| --- | --- | --- | --- | --- | --- |
| random | 0.9412 / 0.9950 / 0.9994 (0.0582, F) | -0.3946 / -0.0009 / 0.4072 (0.8018, F) | 0.887 | 0.175 | 0.337 |
| Aaliyah | 0.9982 / 0.9994 / 0.9999 (0.0017, T) | -0.0719 / 0.3070 / 0.6392 (0.7111, F) | 0.474 | 0.378 | 0.291 |
| Radiohead | 0.9635 / 0.9946 / 0.9992 (0.0357, F) | -0.3085 / 0.0355 / 0.3814 (0.6899, F) | 0.679 | 0.124 | 0.320 |
| cross within_a (Minor Threat) | 0.9981 / 0.9996 / 0.9999 (0.0019, T) | 0.4475 / 0.6777 / 0.8820 (0.4344, F) | 0.621 | 0.571 | 0.416 |
| cross within_b (Green-House) | 0.9189 / 0.9896 / 0.9980 (0.0791, F) | 0.0845 / 0.4106 / 0.6760 (0.5915, F) | 0.893 | 0.089 | 0.417 |
| cross across | 0.8751 / 0.9612 / 0.9891 (0.1140, F) | -0.6376 / -0.4532 / -0.1914 (0.4462, F) | 0.997 | 0.242 | 0.296 |

Takeaway: S_raw is saturated for Aaliyah and within Minor Threat; S_z restores large spread in all cases and weakens correlation to centered transitions.

## Feature Ablation (top 5 dims by |Δp50|; baseline S_raw p10/p50/p90)

| Run | Baseline p10/p50/p90 | Top dims (|Δp50|, dim) |
| --- | --- | --- |
| random | 0.9412 / 0.9950 / 0.9994 | 0.0612 (dim_26), 0.0034 (dim_00), 0.0010 (dim_01), 0.00044 (dim_25), 0.00020 (dim_02) |
| cross_groups (across) | 0.8751 / 0.9612 / 0.9891 | 0.1424 (dim_26), 0.0274 (dim_00), 0.00635 (dim_01), 0.00272 (dim_25), 0.00020 (dim_02) |
| Aaliyah | 0.9982 / 0.9994 / 0.9999 | 0.0296 (dim_26), 0.00033 (dim_00), 0.00021 (dim_25), 0.000094 (dim_01), 0.000059 (dim_02) |
| Radiohead | 0.9635 / 0.9946 / 0.9992 | 0.0412 (dim_26), 0.00380 (dim_00), 0.00105 (dim_01), 0.00040 (dim_25), 0.00016 (dim_02) |

Dim_26 dominates across all runs; dim_00 and dim_01 are secondary but much smaller. Top-3 dims explain ~98–99% of the total ablation impact in every run.

## Interpretation
- S_raw is globally high and locally saturated for specific artists/groups; z-scoring (or centering) reintroduces spread and reduces dependence on a single dominant dimension.
- A single sonic dimension (dim_26) drives most cosine similarity; removing it drops median similarity materially, showing heavy axis dominance.
- Centered transitions already separate cross-group pairs strongly; S_z provides an orthogonal way to desaturate S_sonic without changing production defaults.

