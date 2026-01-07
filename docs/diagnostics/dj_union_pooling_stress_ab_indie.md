# DJ Union Pooling Stress A/B

- scenario: dj_stress_indie_ladder
- seed_track_id: 99cbf799b947abc1efadeff958fa3a86
- anchors: 99cbf799b947abc1efadeff958fa3a86, 22d617435daa61958c67624fca3dbc23, f4282083af4d8149e32c76c2eccea5bc, 7cf9096e6cc377f4d38d4af7d8bdca69, b6bae06cc55006d73afa712edf0a28d2, 9cc4ace8edc409d5e94a08bfdf2a3615
- length: 30
- stress_segment_pool_max: 80

| label | pooling | waypoint_weight | tracklist_hash | bridge_raw_sonic_sim_mean | bridge_raw_sonic_sim_min | p90_arc_dev | max_jump | genre_target_sim_mean | genre_target_delta_mean | runtime_s | pool_overlap_jaccard | union_only_count | chosen_from_local_count | chosen_from_genre_count | chosen_from_toward_count |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A_baseline_low | baseline | 0.05 | `a4b3efb0113f` | 0.7732 | 0.5387 | 0.3359 | 0.2211 | 0.8194 | 0.0845 | 2.74 | na | na | 0 | 0 | 0 |
| A_baseline_high | baseline | 0.30 | `f7ca183a3071` | 0.7675 | 0.5387 | 0.3501 | 0.2211 | 0.8000 | 0.1115 | 2.56 | na | na | 0 | 0 | 0 |
| B_union_low | dj_union | 0.05 | `080bd88c69ea` | 0.7774 | 0.5994 | 0.3254 | 0.2211 | 0.8213 | 0.0821 | 2.74 | 0.4506 | 30.8 | 16 | 2 | 24 |
| B_union_high | dj_union | 0.30 | `952fe8da2bca` | 0.7675 | 0.5387 | 0.3359 | 0.2211 | 0.8029 | 0.1100 | 2.82 | 0.4533 | 30.6 | 14 | 0 | 24 |

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
- pool_overlap_jaccard: min=0.3613 median=0.4595 max=0.5000
- chosen_from_local_count: min=2.0 median=3.0 max=5.0
- chosen_from_toward_count: min=4.0 median=5.0 max=5.0
- chosen_from_genre_count: min=0.0 median=0.0 max=1.0

### B_union_high
- pool_overlap_jaccard: min=0.3613 median=0.4727 max=0.5000
- chosen_from_local_count: min=1.0 median=2.0 max=5.0
- chosen_from_toward_count: min=4.0 median=5.0 max=5.0
- chosen_from_genre_count: min=0.0 median=0.0 max=0.0
