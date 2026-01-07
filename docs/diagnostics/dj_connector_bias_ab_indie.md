# DJ Connector Bias A/B (Indie Scenario)

- scenario: dj_stress_indie_ladder
- seed_track_id: 99cbf799b947abc1efadeff958fa3a86
- anchors: 99cbf799b947abc1efadeff958fa3a86, 22d617435daa61958c67624fca3dbc23, f4282083af4d8149e32c76c2eccea5bc, 7cf9096e6cc377f4d38d4af7d8bdca69, b6bae06cc55006d73afa712edf0a28d2, 9cc4ace8edc409d5e94a08bfdf2a3615
- length: 30

| label | connectors | waypoint_weight | tracklist_hash | bridge_raw_sonic_sim_mean | bridge_raw_sonic_sim_min | p90_arc_dev | max_jump | genre_target_sim_mean | genre_target_delta_mean | runtime_s | pool_overlap_jaccard | connectors_injected_count | connectors_chosen_count |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A_connectors_off_low | false | 0.05 | `080bd88c69ea` | 0.7774 | 0.5994 | 0.3254 | 0.2211 | 0.8213 | 0.0821 | 2.58 | 0.1546 | na | na |
| A_connectors_off_high | false | 0.30 | `952fe8da2bca` | 0.7675 | 0.5387 | 0.3359 | 0.2211 | 0.8029 | 0.1100 | 2.71 | 0.1550 | na | na |
| B_connectors_on_low | true | 0.05 | `080bd88c69ea` | 0.7774 | 0.5994 | 0.3254 | 0.2211 | 0.8213 | 0.0821 | 2.78 | 0.1536 | 15 | 5 |
| B_connectors_on_high | true | 0.30 | `952fe8da2bca` | 0.7675 | 0.5387 | 0.3359 | 0.2211 | 0.8029 | 0.1100 | 2.72 | 0.1541 | 15 | 5 |

## Per-Segment Summary (min/median/max)

### A_connectors_off_low
- pool_overlap_jaccard: min=0.1306 median=0.1551 max=0.1800
- connectors_injected_count: min=0.0 median=0.0 max=0.0
- connectors_chosen_count: min=0.0 median=0.0 max=0.0

### A_connectors_off_high
- pool_overlap_jaccard: min=0.1306 median=0.1551 max=0.1800
- connectors_injected_count: min=0.0 median=0.0 max=0.0
- connectors_chosen_count: min=0.0 median=0.0 max=0.0

### B_connectors_on_low
- pool_overlap_jaccard: min=0.1300 median=0.1540 max=0.1787
- connectors_injected_count: min=3.0 median=3.0 max=3.0
- connectors_chosen_count: min=0.0 median=1.0 max=2.0

### B_connectors_on_high
- pool_overlap_jaccard: min=0.1300 median=0.1540 max=0.1787
- connectors_injected_count: min=3.0 median=3.0 max=3.0
- connectors_chosen_count: min=0.0 median=1.0 max=2.0
