# DJ Ladder Route A/B (Indie Scenario)

- variants: linear, ladder_onehot, ladder_smoothed
- scenario: dj_stress_indie_ladder
- seed_track_id: 99cbf799b947abc1efadeff958fa3a86
- anchors: 99cbf799b947abc1efadeff958fa3a86, 22d617435daa61958c67624fca3dbc23, f4282083af4d8149e32c76c2eccea5bc, 7cf9096e6cc377f4d38d4af7d8bdca69, b6bae06cc55006d73afa712edf0a28d2, 9cc4ace8edc409d5e94a08bfdf2a3615
- length: 30

| label | route_shape | waypoint_vector_mode | tracklist_hash | bridge_raw_sonic_sim_mean | bridge_raw_sonic_sim_min | p90_arc_dev | max_jump | chosen_from_local_count | chosen_from_genre_count | chosen_from_toward_count | pool_overlap_jaccard |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| linear | linear | onehot | `952fe8da2bca` | 0.7674 | 0.5384 | 0.3347 | 0.2214 | 13 | 1 | 24 | 0.4614 |
| ladder_onehot | ladder | onehot | `59969b679370` | 0.7730 | 0.5993 | 0.3276 | 0.2269 | 15 | 0 | 24 | 0.4483 |
| ladder_smoothed | ladder | smoothed | `080bd88c69ea` | 0.7773 | 0.5993 | 0.3276 | 0.2214 | 15 | 0 | 24 | 0.4452 |

## Waypoint Labels By Segment

### linear
- segment 0: route_shape=linear mode=onehot waypoint_count=0 labels=
- segment 1: route_shape=linear mode=onehot waypoint_count=0 labels=
- segment 2: route_shape=linear mode=onehot waypoint_count=0 labels=
- segment 3: route_shape=linear mode=onehot waypoint_count=0 labels=
- segment 4: route_shape=linear mode=onehot waypoint_count=0 labels=

### ladder_onehot
- segment 0: route_shape=ladder mode=onehot waypoint_count=3 labels=indie rock, alternative rock, rock
- segment 1: route_shape=ladder mode=onehot waypoint_count=3 labels=rock, alternative rock, indie rock
- segment 2: route_shape=ladder mode=onehot waypoint_count=3 labels=indie rock, alternative rock, rock
- segment 3: route_shape=ladder mode=onehot waypoint_count=3 labels=rock, alternative rock, indie rock
- segment 4: route_shape=ladder mode=onehot waypoint_count=3 labels=indie rock, shoegaze, ambient

### ladder_smoothed
- segment 0: route_shape=ladder mode=smoothed waypoint_count=3 labels=indie rock, alternative rock, rock
  - smoothed_top3: indie rock[indie rock:0.12, indie:0.11, alternative:0.11], alternative rock[alternative rock:0.12, alternative:0.11, indie rock:0.11], rock[rock:0.12, alternative rock:0.10, acid rock:0.10]
- segment 1: route_shape=ladder mode=smoothed waypoint_count=3 labels=rock, alternative rock, indie rock
  - smoothed_top3: rock[rock:0.12, alternative rock:0.10, acid rock:0.10], alternative rock[alternative rock:0.12, alternative:0.11, indie rock:0.11], indie rock[indie rock:0.12, indie:0.11, alternative:0.11]
- segment 2: route_shape=ladder mode=smoothed waypoint_count=3 labels=indie rock, alternative rock, rock
  - smoothed_top3: indie rock[indie rock:0.12, indie:0.11, alternative:0.11], alternative rock[alternative rock:0.12, alternative:0.11, indie rock:0.11], rock[rock:0.12, alternative rock:0.10, acid rock:0.10]
- segment 3: route_shape=ladder mode=smoothed waypoint_count=3 labels=rock, alternative rock, indie rock
  - smoothed_top3: rock[rock:0.12, alternative rock:0.10, acid rock:0.10], alternative rock[alternative rock:0.12, alternative:0.11, indie rock:0.11], indie rock[indie rock:0.12, indie:0.11, alternative:0.11]
- segment 4: route_shape=ladder mode=smoothed waypoint_count=3 labels=indie rock, shoegaze, ambient
  - smoothed_top3: indie rock[indie rock:0.12, indie:0.11, alternative:0.11], shoegaze[shoegaze:0.16, dream pop:0.14, noise pop:0.14], ambient[ambient:0.13, drone:0.11, classical modern ambient minimalism piano electronicnic:0.10]

## Ladder Warnings

### linear
- none

### ladder_onehot
- none

### ladder_smoothed
- none
