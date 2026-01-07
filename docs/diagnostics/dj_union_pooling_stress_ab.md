# DJ Union Pooling Stress A/B

- scenario: dj_stress_charli
- seed_track_id: 97bf21610ba410fba544940267ebe0c5
- anchors: 97bf21610ba410fba544940267ebe0c5, 5a33b78a22826c00be01634ffbf22d9e, e8dd7b07d07a1ded4c95983db39aacec, 5352c2182087b0ec214332f85772557a
- length: 30
- stress_segment_pool_max: 80

| label | pooling | waypoint_weight | tracklist_hash | bridge_raw_sonic_sim_mean | bridge_raw_sonic_sim_min | p90_arc_dev | max_jump | genre_target_sim_mean | genre_target_delta_mean | runtime_s | pool_overlap_jaccard | union_only_count | chosen_from_local_count | chosen_from_genre_count | chosen_from_toward_count |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A_baseline_low | baseline | 0.05 | `3d861181db1f` | 0.7502 | 0.5653 | 0.3505 | 0.2659 | 0.6526 | 0.1424 | 1.49 | na | na | 0 | 0 | 0 |
| A_baseline_high | baseline | 0.30 | `554101630b67` | 0.7477 | 0.5653 | 0.3505 | 0.2659 | 0.6374 | 0.1562 | 1.66 | na | na | 0 | 0 | 0 |
| B_union_low | dj_union | 0.05 | `08a2358e55d3` | 0.7595 | 0.6053 | 0.3267 | 0.2659 | 0.6541 | 0.1406 | 1.68 | 0.5526 | 25.0 | 21 | 3 | 26 |
| B_union_high | dj_union | 0.30 | `5b37fba2d29f` | 0.7570 | 0.5947 | 0.3267 | 0.2659 | 0.6389 | 0.1545 | 1.70 | 0.5526 | 25.0 | 22 | 3 | 26 |

## Per-Segment Summary (min/median/max)

### A_baseline_low
- pool_overlap_jaccard: min=0.0000 median=0.0000 max=0.0000
- chosen_from_local_count: min=0.0 median=0.0 max=0.0
- chosen_from_toward_count: min=0.0 median=0.0 max=0.0
- chosen_from_genre_count: min=0.0 median=0.0 max=0.0

### A_baseline_high
- pool_overlap_jaccard: min=0.0000 median=0.0000 max=0.0000
- chosen_from_local_count: min=0.0 median=0.0 max=0.0
- chosen_from_toward_count: min=0.0 median=0.0 max=0.0
- chosen_from_genre_count: min=0.0 median=0.0 max=0.0

### B_union_low
- pool_overlap_jaccard: min=0.2960 median=0.6200 max=0.7419
- chosen_from_local_count: min=4.0 median=8.0 max=9.0
- chosen_from_toward_count: min=8.0 median=9.0 max=9.0
- chosen_from_genre_count: min=0.0 median=0.0 max=3.0

### B_union_high
- pool_overlap_jaccard: min=0.2960 median=0.6200 max=0.7419
- chosen_from_local_count: min=4.0 median=9.0 max=9.0
- chosen_from_toward_count: min=8.0 median=9.0 max=9.0
- chosen_from_genre_count: min=0.0 median=0.0 max=3.0
