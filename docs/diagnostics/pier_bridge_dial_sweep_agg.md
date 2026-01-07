# Pier-Bridge Dial Sweep (Aggregated)

- total_grid_size: 11201
- sampled: False (shortlist source=docs\diagnostics\pier_bridge_dial_sweep.csv)

## Aggregated Results

| label | runs | unique_hashes | raw_mean | raw_min | p90_dev | max_jump | overlap | runtime_s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| arc_w0.75_t0.0_abs_m0.1_autoFalse_tb0.02_beam50 | 3 | 3 | 0.8391102356419741 | 0.703986005368776 | 0.37228172513540675 | 0.20062384413128784 | None | 3.401 |
| arc_w1.5_t0.05_abs_m0.2_autoFalse_tb0.02_beamNone | 3 | 3 | 0.7093370680075216 | 0.3708719159694342 | 0.297099474889575 | 0.18206213869880353 | None | 2.466666666666667 |
| arc_w1.5_t0.08_abs_m0.15_autoFalse_tb0.02_beam10 | 3 | 3 | 0.8296465392837642 | 0.6569224450148764 | 0.3334347016526031 | 0.20062384413128784 | None | 2.111 |
| arc_w1.5_t0.08_abs_m0.15_autoTrue_tb0.02_beam10 | 3 | 3 | 0.8296465392837642 | 0.6569224450148764 | 0.3334347016526031 | 0.20062384413128784 | None | 2.09 |
| arc_w1.5_t0.08_abs_m0.1_autoTrue_tbNone_beamNone | 3 | 3 | 0.711859993024222 | 0.3708719159694342 | 0.2926792342283462 | 0.1871349491941484 | None | 2.457333333333333 |
| arc_w1.5_t0.0_abs_m0.2_autoFalse_tb0.02_beamNone | 3 | 3 | 0.7045677913826708 | 0.38177523221145987 | 0.297099474889575 | 0.17269714312126294 | None | 2.447 |
| arc_w1.5_t0.12_abs_m0.15_autoFalse_tb0.02_beam10 | 3 | 3 | 0.8228461332071942 | 0.6605323383057922 | 0.31971627626577326 | 0.20062384413128784 | None | 2.0703333333333336 |
| arc_w1.5_t0.2_abs_m0.1_autoTrue_tbNone_beam25 | 3 | 3 | 0.8284381501799564 | 0.6449104052672797 | 0.3375040868703716 | 0.20062384413128784 | None | 2.5103333333333335 |
| arc_w2.0_t0.05_abs_m0.1_autoFalse_tb0.02_beam50 | 3 | 3 | 0.8323410671737118 | 0.6591627475877714 | 0.32378566148354176 | 0.20062384413128784 | None | 3.4793333333333334 |
| arc_w2.0_t0.05_abs_m0.1_autoTrue_tbNone_beamNone | 3 | 3 | 0.7015722488215363 | 0.4072533749679983 | 0.2854852056819982 | 0.18668061542664405 | None | 2.5056666666666665 |
| arc_w2.0_t0.08_abs_m0.1_autoFalse_tbNone_beamNone | 3 | 3 | 0.7094948335515547 | 0.4072533749679983 | 0.2854852056819982 | 0.19175342592198893 | None | 2.459 |
| arc_w2.0_t0.08_abs_m0.4_autoFalse_tbNone_beam50 | 3 | 3 | 0.8346234866923637 | 0.6591627475877714 | 0.3279565820806163 | 0.20679345256078097 | None | 3.4339999999999997 |
| arc_w2.0_t0.0_abs_m0.15_autoTrue_tbNone_beamNone | 3 | 3 | 0.7147918511127287 | 0.4313305086163031 | 0.297099474889575 | 0.16253349297758418 | None | 2.435 |
| arc_w2.0_t0.0_abs_m0.3_autoTrue_tbNone_beamNone | 3 | 3 | 0.7042030757094192 | 0.4285784620044013 | 0.2924959112574777 | 0.16626408575899257 | None | 2.4103333333333334 |
| arc_w2.0_t0.0_abs_m0.4_autoTrue_tb0.02_beam50 | 3 | 3 | 0.8335712814621431 | 0.6591627475877714 | 0.3279565820806163 | 0.20679345256078097 | None | 3.4 |
| arc_w2.0_t0.2_abs_m0.1_autoTrue_tbNone_beam25 | 3 | 3 | 0.8280301295915256 | 0.646638212389228 | 0.31930429686170575 | 0.17613776332327036 | None | 2.559 |
| arc_w2.0_t0.2_abs_m0.3_autoFalse_tbNone_beam50 | 3 | 3 | 0.8296548287299618 | 0.648878514962123 | 0.32378566148354176 | 0.1870937715187921 | None | 3.3653333333333335 |
| baseline | 3 | 3 | 0.7081190408705956 | 0.4658226719568275 | 0.34329849472921703 | 0.20357473438432394 | 1.0 | 2.2086666666666663 |

## Stability Shortlists

### Best Pacing (mean p90, then std p90)

- arc_w2.0_t0.05_abs_m0.1_autoTrue_tbNone_beamNone: p90_mean=0.2854852056819982, p90_std=0.019457245655738715, max_jump_mean=0.18668061542664405, bridge_min_mean=0.4072533749679983
- arc_w2.0_t0.08_abs_m0.1_autoFalse_tbNone_beamNone: p90_mean=0.2854852056819982, p90_std=0.019457245655738715, max_jump_mean=0.19175342592198893, bridge_min_mean=0.4072533749679983
- arc_w2.0_t0.0_abs_m0.3_autoTrue_tbNone_beamNone: p90_mean=0.2924959112574777, p90_std=0.017595741967181142, max_jump_mean=0.16626408575899257, bridge_min_mean=0.4285784620044013
- arc_w1.5_t0.08_abs_m0.1_autoTrue_tbNone_beamNone: p90_mean=0.2926792342283462, p90_std=0.01275684238886391, max_jump_mean=0.1871349491941484, bridge_min_mean=0.3708719159694342
- arc_w1.5_t0.05_abs_m0.2_autoFalse_tb0.02_beamNone: p90_mean=0.297099474889575, p90_std=0.011523421387498544, max_jump_mean=0.18206213869880353, bridge_min_mean=0.3708719159694342
- arc_w1.5_t0.0_abs_m0.2_autoFalse_tb0.02_beamNone: p90_mean=0.297099474889575, p90_std=0.011523421387498544, max_jump_mean=0.17269714312126294, bridge_min_mean=0.38177523221145987
- arc_w2.0_t0.0_abs_m0.15_autoTrue_tbNone_beamNone: p90_mean=0.297099474889575, p90_std=0.011523421387498544, max_jump_mean=0.16253349297758418, bridge_min_mean=0.4313305086163031
- arc_w2.0_t0.2_abs_m0.1_autoTrue_tbNone_beam25: p90_mean=0.31930429686170575, p90_std=0.007565024647269193, max_jump_mean=0.17613776332327036, bridge_min_mean=0.648878514962123
- arc_w1.5_t0.12_abs_m0.15_autoFalse_tb0.02_beam10: p90_mean=0.31971627626577326, p90_std=0.007060447879096205, max_jump_mean=0.20062384413128784, bridge_min_mean=0.6627726408786871
- arc_w2.0_t0.05_abs_m0.1_autoFalse_tb0.02_beam50: p90_mean=0.32378566148354176, p90_std=0.009963694334703755, max_jump_mean=0.20062384413128784, bridge_min_mean=0.6591627475877714

### Balanced (beats baseline on mean p90, mean max_jump, mean bridge_min)

- arc_w2.0_t0.2_abs_m0.1_autoTrue_tbNone_beam25: p90_mean=0.31930429686170575, max_jump_mean=0.17613776332327036, bridge_min_mean=0.648878514962123, overlap_mean=None
- arc_w1.5_t0.12_abs_m0.15_autoFalse_tb0.02_beam10: p90_mean=0.31971627626577326, max_jump_mean=0.20062384413128784, bridge_min_mean=0.6627726408786871, overlap_mean=None
- arc_w2.0_t0.2_abs_m0.3_autoFalse_tbNone_beam50: p90_mean=0.32378566148354176, max_jump_mean=0.1870937715187921, bridge_min_mean=0.648878514962123, overlap_mean=None
- arc_w2.0_t0.05_abs_m0.1_autoFalse_tb0.02_beam50: p90_mean=0.32378566148354176, max_jump_mean=0.20062384413128784, bridge_min_mean=0.6591627475877714, overlap_mean=None
- arc_w1.5_t0.08_abs_m0.15_autoFalse_tb0.02_beam10: p90_mean=0.3334347016526031, max_jump_mean=0.20062384413128784, bridge_min_mean=0.6591627475877714, overlap_mean=None
- arc_w1.5_t0.08_abs_m0.15_autoTrue_tb0.02_beam10: p90_mean=0.3334347016526031, max_jump_mean=0.20062384413128784, bridge_min_mean=0.6591627475877714, overlap_mean=None
- arc_w1.5_t0.2_abs_m0.1_autoTrue_tbNone_beam25: p90_mean=0.3375040868703716, max_jump_mean=0.20062384413128784, bridge_min_mean=0.6471507078401747, overlap_mean=None

### Most Seed-Unstable (highest p90 std, then max_jump std)

- baseline: p90_std=0.028171422347321704, max_jump_std=0.007838024538517481, unique_hashes=3
- arc_w0.75_t0.0_abs_m0.1_autoFalse_tb0.02_beam50: p90_std=0.027157207232578976, max_jump_std=0.034628547568061864, unique_hashes=3
- arc_w1.5_t0.08_abs_m0.15_autoFalse_tb0.02_beam10: p90_std=0.025811209525542268, max_jump_std=0.034628547568061864, unique_hashes=3
- arc_w1.5_t0.08_abs_m0.15_autoTrue_tb0.02_beam10: p90_std=0.025811209525542268, max_jump_std=0.034628547568061864, unique_hashes=3
- arc_w1.5_t0.2_abs_m0.1_autoTrue_tbNone_beam25: p90_std=0.024576262994055173, max_jump_std=0.034628547568061864, unique_hashes=3
- arc_w2.0_t0.08_abs_m0.1_autoFalse_tbNone_beamNone: p90_std=0.019457245655738715, max_jump_std=0.02414092830856333, unique_hashes=3
- arc_w2.0_t0.05_abs_m0.1_autoTrue_tbNone_beamNone: p90_std=0.019457245655738715, max_jump_std=0.02061360073814236, unique_hashes=3
- arc_w2.0_t0.0_abs_m0.3_autoTrue_tbNone_beamNone: p90_std=0.017595741967181142, max_jump_std=0.02109882002564195, unique_hashes=3
- arc_w2.0_t0.08_abs_m0.4_autoFalse_tbNone_beam50: p90_std=0.014511516635661711, max_jump_std=0.031194957635862868, unique_hashes=3
- arc_w2.0_t0.0_abs_m0.4_autoTrue_tb0.02_beam50: p90_std=0.014511516635661711, max_jump_std=0.031194957635862868, unique_hashes=3

