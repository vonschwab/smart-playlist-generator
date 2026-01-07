# DJ Union Pooling A/B

## Runs

| label | pooling | waypoint_weight | waypoint_floor | waypoint_penalty | hash | bridge_raw_sonic_min | bridge_raw_sonic_mean | p90_arc_dev | max_jump | genre_target_sim_mean | genre_target_delta_mean | runtime_s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A_baseline_low | baseline | 0.05 | 0.20 | 0.05 | `9ec146a44ba4fa1aea5bf204397a9ca59282cc94d767feb41cdca7c3b35d5145` | 0.9180681043248536 | 0.9694469792540138 | 0.3180650316274316 | 0.2446094140880024 | 0.677270477780929 | 0.12895442480626312 | 2.05 |
| A_baseline_high | baseline | 0.25 | 0.10 | 0.10 | `3f77dc391d947e84de6fbdf40f433c3e4ea43d0176962266739c9fb1490eeca7` | 0.9343615525718929 | 0.9701544428082918 | 0.3127314003940941 | 0.2359191194993997 | 0.6177681879355357 | 0.15954579348149506 | 2.03 |
| B_union_low | dj_union | 0.05 | 0.20 | 0.05 | `9ec146a44ba4fa1aea5bf204397a9ca59282cc94d767feb41cdca7c3b35d5145` | 0.9180681043248536 | 0.9694469792540138 | 0.3180650316274316 | 0.2446094140880024 | 0.677270477780929 | 0.12895442480626312 | 1.46 |
| B_union_high | dj_union | 0.25 | 0.10 | 0.10 | `3f77dc391d947e84de6fbdf40f433c3e4ea43d0176962266739c9fb1490eeca7` | 0.9343615525718929 | 0.9701544428082918 | 0.3127314003940941 | 0.2359191194993997 | 0.6177681879355357 | 0.15954579348149506 | 1.43 |

## Pool Composition Summary

- A_baseline_low: {'dj_pool_source_local': 0, 'dj_pool_source_toward': 0, 'dj_pool_source_genre': 0, 'dj_pool_union_deduped': 0, 'dj_pool_union_capped': 0, 'segments_with_pool_counts': 3}
- A_baseline_high: {'dj_pool_source_local': 0, 'dj_pool_source_toward': 0, 'dj_pool_source_genre': 0, 'dj_pool_union_deduped': 0, 'dj_pool_union_capped': 0, 'segments_with_pool_counts': 3}
- B_union_low: {'dj_pool_source_local': 600.0, 'dj_pool_source_toward': 2080.0, 'dj_pool_source_genre': 2080.0, 'dj_pool_union_deduped': 1318.0, 'dj_pool_union_capped': 1318.0, 'segments_with_pool_counts': 3}
- B_union_high: {'dj_pool_source_local': 600.0, 'dj_pool_source_toward': 2080.0, 'dj_pool_source_genre': 2080.0, 'dj_pool_union_deduped': 1319.0, 'dj_pool_union_capped': 1319.0, 'segments_with_pool_counts': 3}
