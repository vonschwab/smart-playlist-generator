# Pier-Bridge Dial Sweep

- total_grid_size: 11201
- sampled: False (shortlist source=docs\diagnostics\pier_bridge_dial_sweep.csv)

## Missing Metrics

- Genre cache hit rate is not present in the run audit markdown output.

## Metric Definitions

- raw_sonic_sim_*: mean/min transition similarity on final ordering using the same transition scoring space.
- bridge_raw_sonic_*: same as above, restricted to bridge edges between consecutive piers.
- p50_arc_dev/p90_arc_dev: absolute deviation from the arc target curve using post-hoc progress projection.
- max_jump: max forward progress jump within a bridge segment (post-hoc).
- monotonic_violations: count of backward steps in progress within segments.
- composite_score: z-normalized blend of raw_sonic_sim_mean/min, p90_arc_dev, max_jump, and runtime.

## Sweep Results (43 unique tracklists)

| label | seed | raw_sonic_sim_mean | raw_sonic_sim_min | bridge_raw_sonic_mean | bridge_raw_sonic_min | p90_arc_dev | max_jump | mono_viol | overlap | hash | runtime_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| arc_w0.75_t0.0_abs_m0.1_autoFalse_tb0.02_beam50 | b262f479713af81acff1f20b3a2a43d7 | 0.8452824632924976 | 0.7535776195429428 | 0.8479889664047261 | 0.7535776195429428 | 0.340432965627282 | 0.17613776332327036 | 2 | None | `d3116c8eb1f4a695fb7c41b32b641e5c0a5fbc88934be41f0b101712de1ab996` | 3.478 |
| arc_w0.75_t0.0_abs_m0.1_autoFalse_tb0.02_beam50 | 486803832f3fc21eb7fa4512a0283554 | 0.8283608081659447 | 0.7240783731482865 | 0.8283608081659447 | 0.7240783731482865 | 0.36961811830256086 | 0.24959600574732277 | 7 | None | `c40867d2d3581c5d75c4ac8ebc7c926711b1be8160755b07e6c20f130af917b2` | 3.433 |
| arc_w0.75_t0.0_abs_m0.1_autoFalse_tb0.02_beam50 | 09daedfeff5751c11bd5a39b97475425 | 0.8436874354674799 | 0.6343020234150984 | 0.8446405636933957 | 0.6343020234150984 | 0.40679409147637746 | 0.17613776332327036 | 6 | None | `e6e052df46509e66b7a159d7e158e985172e2dd4a41518b7e7486e9d4798b2f2` | 3.292 |
| arc_w1.5_t0.05_abs_m0.2_autoFalse_tb0.02_beamNone | b262f479713af81acff1f20b3a2a43d7 | 0.6944027010922482 | 0.3955060103485438 | 0.6944027010922482 | 0.3955060103485438 | 0.3106680293809252 | 0.19716760993017682 | 6 | None | `c6a1f2532476715952c7b1660941b8c91fbbe3a66358d73ceaef0271cbc7cdac` | 2.483 |
| arc_w1.5_t0.05_abs_m0.2_autoFalse_tb0.02_beamNone | 486803832f3fc21eb7fa4512a0283554 | 0.697570388231649 | 0.3382076170986094 | 0.697570388231649 | 0.3382076170986094 | 0.29813214125839743 | 0.15788084485332976 | 4 | None | `f92bae6baa7b44b88b534c4f3b6ff28117aa897666988fb0054f482cd722513d` | 2.427 |
| arc_w1.5_t0.05_abs_m0.2_autoFalse_tb0.02_beamNone | 09daedfeff5751c11bd5a39b97475425 | 0.736038114698668 | 0.37890212046114924 | 0.736038114698668 | 0.37890212046114924 | 0.2824982540294024 | 0.19113796131290406 | 6 | None | `84f9c7b15a675edbf2486e03016eb53808feb3fd3c9746bd39e7943299a0a1f4` | 2.49 |
| arc_w1.5_t0.08_abs_m0.15_autoFalse_tb0.02_beam10 | b262f479713af81acff1f20b3a2a43d7 | 0.8339251567055113 | 0.6827701884564017 | 0.8317170777768638 | 0.6827701884564017 | 0.3111718933037894 | 0.17613776332327036 | 3 | None | `0b52d779e87b8f00d8fdfd9214e5ae4695a3478c7860e74e5c8b5413b67978a5` | 2.154 |
| arc_w1.5_t0.08_abs_m0.15_autoFalse_tb0.02_beam10 | 486803832f3fc21eb7fa4512a0283554 | 0.8177797350607425 | 0.6604160308918139 | 0.8177797350607425 | 0.6604160308918139 | 0.36961811830256086 | 0.24959600574732277 | 8 | None | `24d3269545bb25e3507a1042d7b6fd2eaee880fff610b4168b3391cd84c78c65` | 2.062 |
| arc_w1.5_t0.08_abs_m0.15_autoFalse_tb0.02_beam10 | 09daedfeff5751c11bd5a39b97475425 | 0.8372347260850387 | 0.6275811156964136 | 0.8401022432184692 | 0.6343020234150984 | 0.31951409335145897 | 0.17613776332327036 | 3 | None | `7b5cce4454dff67c983760f119bf0a41c9e822d53207d3080499586666e322c0` | 2.117 |
| arc_w1.5_t0.08_abs_m0.15_autoTrue_tb0.02_beam10 | b262f479713af81acff1f20b3a2a43d7 | 0.8339251567055113 | 0.6827701884564017 | 0.8317170777768638 | 0.6827701884564017 | 0.3111718933037894 | 0.17613776332327036 | 3 | None | `0b52d779e87b8f00d8fdfd9214e5ae4695a3478c7860e74e5c8b5413b67978a5` | 2.09 |
| arc_w1.5_t0.08_abs_m0.15_autoTrue_tb0.02_beam10 | 486803832f3fc21eb7fa4512a0283554 | 0.8177797350607425 | 0.6604160308918139 | 0.8177797350607425 | 0.6604160308918139 | 0.36961811830256086 | 0.24959600574732277 | 8 | None | `24d3269545bb25e3507a1042d7b6fd2eaee880fff610b4168b3391cd84c78c65` | 2.052 |
| arc_w1.5_t0.08_abs_m0.15_autoTrue_tb0.02_beam10 | 09daedfeff5751c11bd5a39b97475425 | 0.8372347260850387 | 0.6275811156964136 | 0.8401022432184692 | 0.6343020234150984 | 0.31951409335145897 | 0.17613776332327036 | 3 | None | `7b5cce4454dff67c983760f119bf0a41c9e822d53207d3080499586666e322c0` | 2.128 |
| arc_w1.5_t0.08_abs_m0.1_autoTrue_tbNone_beamNone | b262f479713af81acff1f20b3a2a43d7 | 0.7084835333875819 | 0.3955060103485438 | 0.7084835333875819 | 0.3955060103485438 | 0.3106680293809252 | 0.21238604141621137 | 7 | None | `1061bdccc727c3c538084f067e98ec494af5b755f34ebdafeb1bf9499e8e2baf` | 2.499 |
| arc_w1.5_t0.08_abs_m0.1_autoTrue_tbNone_beamNone | 486803832f3fc21eb7fa4512a0283554 | 0.6840288410087834 | 0.3382076170986094 | 0.6840288410087834 | 0.3382076170986094 | 0.284871419274711 | 0.15788084485332976 | 2 | None | `763a27731bc2f510d862e28eaa0f63a317a76dfa9839bf04345d86b5dd206683` | 2.41 |
| arc_w1.5_t0.08_abs_m0.1_autoTrue_tbNone_beamNone | 09daedfeff5751c11bd5a39b97475425 | 0.7430676046763006 | 0.37890212046114924 | 0.7430676046763006 | 0.37890212046114924 | 0.2824982540294024 | 0.19113796131290406 | 6 | None | `1c532eeb42d9057faa12ff235d1be77b638ee49052ec762696d41ef3b74d5c65` | 2.463 |
| arc_w1.5_t0.0_abs_m0.2_autoFalse_tb0.02_beamNone | b262f479713af81acff1f20b3a2a43d7 | 0.699770235684586 | 0.4282159590746211 | 0.699770235684586 | 0.4282159590746211 | 0.3106680293809252 | 0.16907262319755495 | 5 | None | `4d8cf77a275dccb03f5aa8608bcbddfb12cd3ad87ff32e1ce6cc780c5adf31bf` | 2.487 |
| arc_w1.5_t0.0_abs_m0.2_autoFalse_tb0.02_beamNone | 486803832f3fc21eb7fa4512a0283554 | 0.697570388231649 | 0.3382076170986094 | 0.697570388231649 | 0.3382076170986094 | 0.29813214125839743 | 0.15788084485332976 | 4 | None | `f92bae6baa7b44b88b534c4f3b6ff28117aa897666988fb0054f482cd722513d` | 2.423 |
| arc_w1.5_t0.0_abs_m0.2_autoFalse_tb0.02_beamNone | 09daedfeff5751c11bd5a39b97475425 | 0.7163627502317775 | 0.37890212046114924 | 0.7163627502317775 | 0.37890212046114924 | 0.2824982540294024 | 0.19113796131290406 | 6 | None | `f295daa16cf91cb1fe3b799bf3a650b4537a7d8b1b56a54a90275b9b59d52e4c` | 2.431 |
| arc_w1.5_t0.12_abs_m0.15_autoFalse_tb0.02_beam10 | b262f479713af81acff1f20b3a2a43d7 | 0.8274632029677491 | 0.6827701884564017 | 0.8231990478498137 | 0.6827701884564017 | 0.3111718933037894 | 0.17613776332327036 | 3 | None | `28e35d57757e1fa797ddf512329affbaf0c4af10103b2730b66c66e5904309ff` | 2.119 |
| arc_w1.5_t0.12_abs_m0.15_autoFalse_tb0.02_beam10 | 486803832f3fc21eb7fa4512a0283554 | 0.8120486683572822 | 0.6712457107645613 | 0.8120486683572822 | 0.6712457107645613 | 0.32846284214207133 | 0.24959600574732277 | 6 | None | `0447de8d17b1b520ca044760c8aeb27003b80989868aebec2edbf3ee803ad704` | 2.038 |
| arc_w1.5_t0.12_abs_m0.15_autoFalse_tb0.02_beam10 | 09daedfeff5751c11bd5a39b97475425 | 0.8290265282965514 | 0.6275811156964136 | 0.8292823461336449 | 0.6343020234150984 | 0.31951409335145897 | 0.17613776332327036 | 4 | None | `e13ee6f2fba63164875f3735603b01b69e42b1dd07ac6257234cf44f1b4ba299` | 2.054 |
| arc_w1.5_t0.2_abs_m0.1_autoTrue_tbNone_beam25 | b262f479713af81acff1f20b3a2a43d7 | 0.8312523315122123 | 0.6519174905794566 | 0.8294947017852595 | 0.6519174905794566 | 0.309935955091587 | 0.17613776332327036 | 1 | None | `214b0e9deb4bcedcc27fd5f010f2b3dc5e13cae3df643fa4d0787a6e41b53afa` | 2.556 |
| arc_w1.5_t0.2_abs_m0.1_autoTrue_tbNone_beam25 | 486803832f3fc21eb7fa4512a0283554 | 0.8264950850928356 | 0.6552326095259688 | 0.8264950850928356 | 0.6552326095259688 | 0.36961811830256086 | 0.24959600574732277 | 7 | None | `a20a1f4dd022d87592be1870896f1dcfc8340f04fa105ce34a78b5b3adaca87f` | 2.507 |
| arc_w1.5_t0.2_abs_m0.1_autoTrue_tbNone_beam25 | 09daedfeff5751c11bd5a39b97475425 | 0.8275670339348211 | 0.6275811156964136 | 0.8273584672022735 | 0.6343020234150984 | 0.33295818721696696 | 0.17613776332327036 | 4 | None | `5c13d3b0cc747b007f33d51b41fc476289136f7a855f67464338648df93dfb8d` | 2.468 |
| arc_w2.0_t0.05_abs_m0.1_autoFalse_tb0.02_beam50 | b262f479713af81acff1f20b3a2a43d7 | 0.8409335858197502 | 0.6827701884564017 | 0.8400989739873791 | 0.6827701884564017 | 0.309935955091587 | 0.17613776332327036 | 4 | None | `789e86b9fdaaeb382901d9c84a01c8067ea4fc1629987dccdd39a2c7e818d29d` | 3.604 |
| arc_w2.0_t0.05_abs_m0.1_autoFalse_tb0.02_beam50 | 486803832f3fc21eb7fa4512a0283554 | 0.8148557882792767 | 0.6604160308918139 | 0.8148557882792767 | 0.6604160308918139 | 0.32846284214207133 | 0.24959600574732277 | 7 | None | `9b90a97cf9b35c67040ad2a9c16bc52bf45afb6ed5da7a7f9cd1325da3255cca` | 3.505 |
| arc_w2.0_t0.05_abs_m0.1_autoFalse_tb0.02_beam50 | 09daedfeff5751c11bd5a39b97475425 | 0.8412338274221085 | 0.6343020234150984 | 0.8414062621790425 | 0.6343020234150984 | 0.33295818721696696 | 0.17613776332327036 | 3 | None | `dfe0fba7fe7d584b18c57bb58afb24f81cf014cf5fb5d50420106e25e9ffb973` | 3.329 |
| arc_w2.0_t0.05_abs_m0.1_autoTrue_tbNone_beamNone | b262f479713af81acff1f20b3a2a43d7 | 0.6847157791975265 | 0.3955060103485438 | 0.6847157791975265 | 0.3955060103485438 | 0.3106680293809252 | 0.19716760993017682 | 6 | None | `72e98d3273a9e7423a6eeda63aeefe47abfc182a569dab4e13c7b8202aadb068` | 2.477 |
| arc_w2.0_t0.05_abs_m0.1_autoTrue_tbNone_beamNone | 486803832f3fc21eb7fa4512a0283554 | 0.6847853882214111 | 0.3382076170986094 | 0.6847853882214111 | 0.3382076170986094 | 0.26328933363566703 | 0.15788084485332976 | 2 | None | `d80e2ff8571061b9e567399d9467cb359c9fb8c0469a3ab03cecaeafb67bf387` | 2.595 |
| arc_w2.0_t0.05_abs_m0.1_autoTrue_tbNone_beamNone | 09daedfeff5751c11bd5a39b97475425 | 0.7352155790456714 | 0.48804649745684164 | 0.7352155790456714 | 0.48804649745684164 | 0.2824982540294024 | 0.2049933914964256 | 7 | None | `f9967a4cf7d1a8428cb494920839cda996a3c21b36dec56c3f6f5daef3c926cd` | 2.445 |
| arc_w2.0_t0.08_abs_m0.1_autoFalse_tbNone_beamNone | b262f479713af81acff1f20b3a2a43d7 | 0.7084835333875819 | 0.3955060103485438 | 0.7084835333875819 | 0.3955060103485438 | 0.3106680293809252 | 0.21238604141621137 | 7 | None | `1061bdccc727c3c538084f067e98ec494af5b755f34ebdafeb1bf9499e8e2baf` | 2.458 |
| arc_w2.0_t0.08_abs_m0.1_autoFalse_tbNone_beamNone | 486803832f3fc21eb7fa4512a0283554 | 0.6847853882214111 | 0.3382076170986094 | 0.6847853882214111 | 0.3382076170986094 | 0.26328933363566703 | 0.15788084485332976 | 2 | None | `d80e2ff8571061b9e567399d9467cb359c9fb8c0469a3ab03cecaeafb67bf387` | 2.48 |
| arc_w2.0_t0.08_abs_m0.1_autoFalse_tbNone_beamNone | 09daedfeff5751c11bd5a39b97475425 | 0.7352155790456714 | 0.48804649745684164 | 0.7352155790456714 | 0.48804649745684164 | 0.2824982540294024 | 0.2049933914964256 | 7 | None | `f9967a4cf7d1a8428cb494920839cda996a3c21b36dec56c3f6f5daef3c926cd` | 2.439 |
| arc_w2.0_t0.08_abs_m0.4_autoFalse_tbNone_beam50 | b262f479713af81acff1f20b3a2a43d7 | 0.8409335858197502 | 0.6827701884564017 | 0.8400989739873791 | 0.6827701884564017 | 0.309935955091587 | 0.17613776332327036 | 4 | None | `789e86b9fdaaeb382901d9c84a01c8067ea4fc1629987dccdd39a2c7e818d29d` | 3.525 |
| arc_w2.0_t0.08_abs_m0.4_autoFalse_tbNone_beam50 | 486803832f3fc21eb7fa4512a0283554 | 0.818026101550464 | 0.6604160308918139 | 0.818026101550464 | 0.6604160308918139 | 0.32846284214207133 | 0.24959600574732277 | 7 | None | `cacb480844940559819e44612b8e5de4f0b21951e58dfeb80856bc62db40a5f7` | 3.455 |
| arc_w2.0_t0.08_abs_m0.4_autoFalse_tbNone_beam50 | 09daedfeff5751c11bd5a39b97475425 | 0.8449107727068772 | 0.6343020234150984 | 0.8462531445998741 | 0.6343020234150984 | 0.3454709490081905 | 0.19464658861174977 | 6 | None | `0536ca9d531004055fa13454ad8590fd2d67b5e5dd961c4dcd8b50c7545eaa03` | 3.322 |
| arc_w2.0_t0.0_abs_m0.15_autoTrue_tbNone_beamNone | b262f479713af81acff1f20b3a2a43d7 | 0.6900833137898643 | 0.4282159590746211 | 0.6900833137898643 | 0.4282159590746211 | 0.3106680293809252 | 0.13913386909208875 | 5 | None | `0aedefbb12cd3185e36f5be713ccc1d93b62ea1ad73271fec0b60302c136704e` | 2.47 |
| arc_w2.0_t0.0_abs_m0.15_autoTrue_tbNone_beamNone | 486803832f3fc21eb7fa4512a0283554 | 0.697570388231649 | 0.3382076170986094 | 0.697570388231649 | 0.3382076170986094 | 0.29813214125839743 | 0.15788084485332976 | 4 | None | `f92bae6baa7b44b88b534c4f3b6ff28117aa897666988fb0054f482cd722513d` | 2.424 |
| arc_w2.0_t0.0_abs_m0.15_autoTrue_tbNone_beamNone | 09daedfeff5751c11bd5a39b97475425 | 0.7567218513166727 | 0.5275679496756788 | 0.7567218513166727 | 0.5275679496756788 | 0.2824982540294024 | 0.19058576498733404 | 6 | None | `f890784b8f217153d5be96ba113e89b1fa07f01f12512b5e9e1919dd588a5f2f` | 2.411 |
| arc_w2.0_t0.0_abs_m0.3_autoTrue_tbNone_beamNone | b262f479713af81acff1f20b3a2a43d7 | 0.6900833137898643 | 0.4282159590746211 | 0.6900833137898643 | 0.4282159590746211 | 0.3106680293809252 | 0.13913386909208875 | 5 | None | `0aedefbb12cd3185e36f5be713ccc1d93b62ea1ad73271fec0b60302c136704e` | 2.465 |
| arc_w2.0_t0.0_abs_m0.3_autoTrue_tbNone_beamNone | 486803832f3fc21eb7fa4512a0283554 | 0.683737435037302 | 0.3382076170986094 | 0.683737435037302 | 0.3382076170986094 | 0.29813214125839743 | 0.16907262319755495 | 6 | None | `d49ac2b9e99e464f0cd940e69262016b77a81e1341a7bfbf16ffda8f3ab62d8f` | 2.358 |
| arc_w2.0_t0.0_abs_m0.3_autoTrue_tbNone_beamNone | 09daedfeff5751c11bd5a39b97475425 | 0.7387884783010913 | 0.5193118098399735 | 0.7387884783010913 | 0.5193118098399735 | 0.26868756313311043 | 0.19058576498733404 | 5 | None | `9d4b48480d26318798eb5efa7523318e04b2136b386edaedab9d49c432561d40` | 2.408 |
| arc_w2.0_t0.0_abs_m0.4_autoTrue_tb0.02_beam50 | b262f479713af81acff1f20b3a2a43d7 | 0.8409335858197502 | 0.6827701884564017 | 0.8400989739873791 | 0.6827701884564017 | 0.309935955091587 | 0.17613776332327036 | 4 | None | `789e86b9fdaaeb382901d9c84a01c8067ea4fc1629987dccdd39a2c7e818d29d` | 3.458 |
| arc_w2.0_t0.0_abs_m0.4_autoTrue_tb0.02_beam50 | 486803832f3fc21eb7fa4512a0283554 | 0.8134127455307343 | 0.6604160308918139 | 0.8134127455307343 | 0.6604160308918139 | 0.32846284214207133 | 0.24959600574732277 | 7 | None | `b6c1bb87090f85faac66c222acb67d1ec03598c6848b0cd3ee93cfbc7608cba5` | 3.436 |
| arc_w2.0_t0.0_abs_m0.4_autoTrue_tb0.02_beam50 | 09daedfeff5751c11bd5a39b97475425 | 0.8463675130359449 | 0.6343020234150984 | 0.8481733932154633 | 0.6343020234150984 | 0.3454709490081905 | 0.19464658861174977 | 6 | None | `812d1b12113dfa226c13570b2fd15c44c129ed166d773f33d875b63d8993e0de` | 3.306 |
| arc_w2.0_t0.2_abs_m0.1_autoTrue_tbNone_beam25 | b262f479713af81acff1f20b3a2a43d7 | 0.8331180545853215 | 0.6519174905794566 | 0.8319540640179942 | 0.6519174905794566 | 0.309935955091587 | 0.17613776332327036 | 1 | None | `32f1333d0c0552e196c82266ed91950c7d4430bed0a7e8faa7fc157b257f092c` | 2.623 |
| arc_w2.0_t0.2_abs_m0.1_autoTrue_tbNone_beam25 | 486803832f3fc21eb7fa4512a0283554 | 0.8174492463622135 | 0.6604160308918139 | 0.8174492463622135 | 0.6604160308918139 | 0.32846284214207133 | 0.17613776332327036 | 6 | None | `bd8ea5d4e1c5763c79aeb4bb145b2ba939c7d585923d498703ca9a358e5c6378` | 2.595 |
| arc_w2.0_t0.2_abs_m0.1_autoTrue_tbNone_beam25 | 09daedfeff5751c11bd5a39b97475425 | 0.8335230878270418 | 0.6275811156964136 | 0.8352096291511095 | 0.6343020234150984 | 0.31951409335145897 | 0.17613776332327036 | 4 | None | `9761e586a90c875399d22fac3385e8c4996abfcbc6e67dbec07c7d687b2ed59e` | 2.459 |
| arc_w2.0_t0.2_abs_m0.3_autoFalse_tbNone_beam50 | b262f479713af81acff1f20b3a2a43d7 | 0.8386148348137181 | 0.6519174905794566 | 0.8377404601078269 | 0.6519174905794566 | 0.309935955091587 | 0.17613776332327036 | 1 | None | `c7f6e0efdb0ad6532cb6a46502d5208ca707e491170113bae38e888811eadf9c` | 3.477 |
| arc_w2.0_t0.2_abs_m0.3_autoFalse_tbNone_beam50 | 486803832f3fc21eb7fa4512a0283554 | 0.816006203613671 | 0.6604160308918139 | 0.816006203613671 | 0.6604160308918139 | 0.32846284214207133 | 0.17613776332327036 | 6 | None | `ea5bbde926138bc14c86fc4a106a2ebcc1210622972c457d4b7351bb34565486` | 3.39 |
| arc_w2.0_t0.2_abs_m0.3_autoFalse_tbNone_beam50 | 09daedfeff5751c11bd5a39b97475425 | 0.8343434477624964 | 0.6343020234150984 | 0.8289147123319346 | 0.6343020234150984 | 0.33295818721696696 | 0.2090057879098356 | 5 | None | `f15cb5589859ccf3bd2880d0909ea5a13725bc4f5ff81bfbe3810af963159685` | 3.229 |
| baseline | b262f479713af81acff1f20b3a2a43d7 | 0.6763449106998393 | 0.42184005046144696 | 0.6763449106998393 | 0.42184005046144696 | 0.32289920741268724 | 0.21238604141621137 | 8 | 1.0 | `384ae4e8513834b89a92f57e9eb878add2ca5ab5180dc70eae21e9e30aeea929` | 2.209 |
| baseline | 486803832f3fc21eb7fa4512a0283554 | 0.7009704655351322 | 0.44735900435803805 | 0.7009704655351322 | 0.44735900435803805 | 0.3831350303186547 | 0.19334477024033492 | 7 | 1.0 | `58cfb01b5d1d15aa9912a07cc924e30744571642859a7c8b0bf030259c257532` | 2.131 |
| baseline | 09daedfeff5751c11bd5a39b97475425 | 0.7470417463768154 | 0.5282689610509975 | 0.7470417463768154 | 0.5282689610509975 | 0.3238612464563091 | 0.2049933914964256 | 9 | 1.0 | `50a1079806aacc471777be193c72d9d555a460c3e0edc48542d5a334537a860c` | 2.286 |

## Top Configs: Smoothness

- arc_w0.75_t0.0_abs_m0.1_autoFalse_tb0.02_beam50: raw_sonic_sim_mean=0.8452824632924976, raw_sonic_sim_min=0.7535776195429428, p90_arc_dev=0.340432965627282, max_jump=0.17613776332327036, overlap=None
- arc_w0.75_t0.0_abs_m0.1_autoFalse_tb0.02_beam50: raw_sonic_sim_mean=0.8283608081659447, raw_sonic_sim_min=0.7240783731482865, p90_arc_dev=0.36961811830256086, max_jump=0.24959600574732277, overlap=None
- arc_w2.0_t0.05_abs_m0.1_autoFalse_tb0.02_beam50: raw_sonic_sim_mean=0.8409335858197502, raw_sonic_sim_min=0.6827701884564017, p90_arc_dev=0.309935955091587, max_jump=0.17613776332327036, overlap=None
- arc_w2.0_t0.08_abs_m0.4_autoFalse_tbNone_beam50: raw_sonic_sim_mean=0.8409335858197502, raw_sonic_sim_min=0.6827701884564017, p90_arc_dev=0.309935955091587, max_jump=0.17613776332327036, overlap=None
- arc_w2.0_t0.0_abs_m0.4_autoTrue_tb0.02_beam50: raw_sonic_sim_mean=0.8409335858197502, raw_sonic_sim_min=0.6827701884564017, p90_arc_dev=0.309935955091587, max_jump=0.17613776332327036, overlap=None

## Top Configs: Pacing

- arc_w2.0_t0.05_abs_m0.1_autoTrue_tbNone_beamNone: raw_sonic_sim_mean=0.6847853882214111, raw_sonic_sim_min=0.3382076170986094, p90_arc_dev=0.26328933363566703, max_jump=0.15788084485332976, overlap=None
- arc_w2.0_t0.08_abs_m0.1_autoFalse_tbNone_beamNone: raw_sonic_sim_mean=0.6847853882214111, raw_sonic_sim_min=0.3382076170986094, p90_arc_dev=0.26328933363566703, max_jump=0.15788084485332976, overlap=None
- arc_w2.0_t0.0_abs_m0.3_autoTrue_tbNone_beamNone: raw_sonic_sim_mean=0.7387884783010913, raw_sonic_sim_min=0.5193118098399735, p90_arc_dev=0.26868756313311043, max_jump=0.19058576498733404, overlap=None
- arc_w2.0_t0.0_abs_m0.15_autoTrue_tbNone_beamNone: raw_sonic_sim_mean=0.7567218513166727, raw_sonic_sim_min=0.5275679496756788, p90_arc_dev=0.2824982540294024, max_jump=0.19058576498733404, overlap=None
- arc_w1.5_t0.05_abs_m0.2_autoFalse_tb0.02_beamNone: raw_sonic_sim_mean=0.736038114698668, raw_sonic_sim_min=0.37890212046114924, p90_arc_dev=0.2824982540294024, max_jump=0.19113796131290406, overlap=None

## Top Configs: Balanced

- arc_w1.5_t0.08_abs_m0.15_autoTrue_tb0.02_beam10: raw_sonic_sim_mean=0.8339251567055113, raw_sonic_sim_min=0.6827701884564017, p90_arc_dev=0.3111718933037894, max_jump=0.17613776332327036, overlap=None
- arc_w1.5_t0.08_abs_m0.15_autoFalse_tb0.02_beam10: raw_sonic_sim_mean=0.8339251567055113, raw_sonic_sim_min=0.6827701884564017, p90_arc_dev=0.3111718933037894, max_jump=0.17613776332327036, overlap=None
- arc_w1.5_t0.12_abs_m0.15_autoFalse_tb0.02_beam10: raw_sonic_sim_mean=0.8274632029677491, raw_sonic_sim_min=0.6827701884564017, p90_arc_dev=0.3111718933037894, max_jump=0.17613776332327036, overlap=None
- arc_w1.5_t0.08_abs_m0.15_autoFalse_tb0.02_beam10: raw_sonic_sim_mean=0.8372347260850387, raw_sonic_sim_min=0.6275811156964136, p90_arc_dev=0.31951409335145897, max_jump=0.17613776332327036, overlap=None
- arc_w1.5_t0.08_abs_m0.15_autoTrue_tb0.02_beam10: raw_sonic_sim_mean=0.8372347260850387, raw_sonic_sim_min=0.6275811156964136, p90_arc_dev=0.31951409335145897, max_jump=0.17613776332327036, overlap=None

## Separation Table

| dial | value | unique_tracklists |
| --- | --- | ---: |
| progress_arc_weight | 2.0 | 22 |
| progress_arc_weight | 1.5 | 17 |
| progress_arc_weight | 0.75 | 3 |
| progress_arc_weight | None | 3 |
| progress_arc_loss | abs | 40 |
| progress_arc_loss | None | 3 |
| progress_arc_max_step | 0.1 | 18 |
| progress_arc_max_step | 0.15 | 9 |
| progress_arc_max_step | 0.3 | 6 |
| progress_arc_max_step | 0.2 | 5 |
| progress_arc_max_step | 0.4 | 5 |
| progress_arc_max_step | None | 3 |
| progress_arc_tolerance | 0.0 | 13 |
| progress_arc_tolerance | 0.08 | 11 |
| progress_arc_tolerance | 0.05 | 9 |
| progress_arc_tolerance | 0.2 | 9 |
| progress_arc_tolerance | 0.12 | 3 |
| progress_arc_tolerance | None | 3 |
| progress_arc_autoscale_enabled | False | 25 |
| progress_arc_autoscale_enabled | True | 23 |
| progress_arc_autoscale_enabled | None | 3 |
| genre_tie_break_band | None | 26 |
| genre_tie_break_band | 0.02 | 19 |
| beam_width | None | 18 |
| beam_width | 50 | 13 |
| beam_width | 10 | 6 |
| beam_width | 25 | 6 |
| arc_strength_bucket | extreme | 37 |
| arc_strength_bucket | high | 3 |
| arc_strength_bucket | off | 3 |
| jumpiness_bucket | tight | 27 |
| jumpiness_bucket | moderate | 11 |
| jumpiness_bucket | loose | 5 |
| jumpiness_bucket | unbounded | 3 |

## What Changed vs Baseline

### seed b262f479713af81acff1f20b3a2a43d7
- baseline hash: `384ae4e8513834b89a92f57e9eb878add2ca5ab5180dc70eae21e9e30aeea929`

- hash `d3116c8eb1f4a695fb7c41b32b641e5c0a5fbc88934be41f0b101712de1ab996` label `arc_w0.75_t0.0_abs_m0.1_autoFalse_tb0.02_beam50` overlap=None
  - pos 2: King Krule - The Cadet Leaps -> St. Vincent - The Bed | base_T=0.7159177022106813 cur_T=0.8682706332347809 base_dev=0.4891376911737034 cur_dev=0.48980799972154376
  - pos 3: Mount Eerie - (soft air) -> King Krule - The Cadet Leaps | base_T=0.6195541647965571 cur_T=0.8130570399659223 base_dev=0.3329677950033614 cur_dev=0.38075131551133384
  - pos 4: Godspeed You! Black Emperor - Peasantry or ‘Light! Inside of Light!’ -> Alex Somers - Looking After | base_T=0.6135803414188412 cur_T=0.8694433726555622 base_dev=0.17513777839063632 cur_dev=0.27866531070496714
  - pos 5: Jónsi & Alex Somers - All The Big Trees (2019 Remaster) -> Tim Hecker - Abduction | base_T=0.6889105761394906 cur_T=0.8464609468686048 base_dev=0.004146027090951188 cur_dev=0.10092497375923676
  - pos 6: Sigur Rós - Andrá -> Mary Lattimore & Walt Mcclements - The Poppies, the Wild Mustard, the Blue-Eyed Grass | base_T=0.6484424356689317 cur_T=0.7602331050266803 base_dev=0.07191639938006988 cur_dev=0.07626982022851603
  - pos 7: Tim Hecker - Abduction -> Sigur Rós - Andrá | base_T=0.4299009771685955 cur_T=0.7535776195429428 base_dev=0.25262841683403703 cur_dev=0.2341280737907988
  - pos 8: Mary Lattimore & Walt Mcclements - The Poppies, the Wild Mustard, the Blue-Eyed Grass -> M83 - Another Wave from You | base_T=0.7602331050266803 cur_T=0.78742005345353 base_dev=0.3468678703016146 cur_dev=0.31354826082438414
  - pos 10: Jonny Nash - October Song -> Cornelius - Surfing on Mind Wave, Pt. 2 | base_T=0.7848286992525537 cur_T=0.8034951146243114 base_dev=0.1286568087201729 cur_dev=0.2072080595593752
  - pos 11: Labradford - V -> Microphones - Here With Summer | base_T=0.7699099714964436 cur_T=0.8398301230820251 base_dev=0.13082749692531817 cur_dev=0.0800313041791113
  - pos 12: TV On The Radio - Tonight -> PaPerCuts - Mattress on the Floor | base_T=0.6403422925030239 cur_T=0.8967572984165522 base_dev=0.059283005278525897 cur_dev=0.05568463355117209
  - pos 13: Animal Collective - No More Runnin -> Dirty Beaches - Sweet 17 | base_T=0.6221370689484218 cur_T=0.8756926284369257 base_dev=0.0642896440993998 cur_dev=0.02039670325270082
  - pos 14: Orchid Mantis - Fog -> Television Personalities - David Hockney's Diary | base_T=0.42184005046144696 cur_T=0.9142592763980442 base_dev=0.0824767435334669 cur_dev=0.17267414574460205
  - pos 15: Wild Nothing - Ride -> Ramones - Glad to See You Go (40th Anniversary Mix) | base_T=0.5511360093665991 cur_T=0.829455849593771 base_dev=0.06725810102744806 cur_dev=0.3090328786583877
  - pos 16: 00110100 01010100 - 0000 871 0017 -> David Bowie - Suffragette City | base_T=0.4464113325140045 cur_T=0.7586708660369917 base_dev=0.04419643724598976 cur_dev=None
  - pos 17: David Bowie - Move On -> No Age - Flutter Freer | base_T=0.5323843106013314 cur_T=0.8874990957333945 base_dev=0.05692913719206438 cur_dev=0.25178264270438616
  - pos 18: The Beach Boys - Please Let Me Wonder -> Here We Go Magic - Tunnelvision | base_T=0.6492639969941926 cur_T=0.9125919289384837 base_dev=0.0029714226868551297 cur_dev=0.1745801465890185
  - pos 19: The Beatles - Little Queenie -> The Flaming Lips - Look... the Sun Rising | base_T=0.6120196198266036 cur_T=0.8853380996895184 base_dev=0.16769208289827908 cur_dev=0.02528938066264863
  - pos 20: Ariel Pink's Haunted Graffiti - Early Birds of Babylon -> Sonic Youth w. Lydia Lunch - Death Valley '69 | base_T=0.76452337442273 cur_T=0.8999231595348618 base_dev=0.11634548897856112 cur_dev=0.06893628574899446
  - pos 21: King Krule - Emergency Blimp -> The Jimi Hendrix Experience - Voodoo Child (Slight Return) | base_T=0.6910258897661363 cur_T=0.8745940561886648 base_dev=0.2734558711528715 cur_dev=0.14922095930894064
  - pos 22: The Fall - Room to Live -> The Stooges - T.V. Eye | base_T=0.7332345871241958 cur_T=0.8266032228365947 base_dev=0.3125249117150196 cur_dev=0.3303533781562691

- hash `c6a1f2532476715952c7b1660941b8c91fbbe3a66358d73ceaef0271cbc7cdac` label `arc_w1.5_t0.05_abs_m0.2_autoFalse_tb0.02_beamNone` overlap=None
  - pos 2: King Krule - The Cadet Leaps -> Godspeed You Black Emperor! - Moya | base_T=0.7159177022106813 cur_T=0.6827701884564017 base_dev=0.4891376911737034 cur_dev=0.3088111470468307
  - pos 5: Jónsi & Alex Somers - All The Big Trees (2019 Remaster) -> King Krule - The Cadet Leaps | base_T=0.6889105761394906 cur_T=0.6955376759430785 base_dev=0.004146027090951188 cur_dev=0.027197924918060112
  - pos 7: Tim Hecker - Abduction -> Foxes in Fiction - Insomnia Keys | base_T=0.4299009771685955 cur_T=0.7853552801920686 base_dev=0.25262841683403703 cur_dev=0.18482013225565375
  - pos 8: Mary Lattimore & Walt Mcclements - The Poppies, the Wild Mustard, the Blue-Eyed Grass -> Spacemen 3 - Things'll Never Be the Same | base_T=0.7602331050266803 cur_T=0.6712457107645613 base_dev=0.3468678703016146 cur_dev=0.31972795188539627
  - pos 10: Jonny Nash - October Song -> Labradford - V | base_T=0.7848286992525537 cur_T=0.6946432657992841 base_dev=0.1286568087201729 cur_dev=0.1678070190650204
  - pos 11: Labradford - V -> Jonny Nash - October Song | base_T=0.7699099714964436 cur_T=0.8502745376085842 base_dev=0.13082749692531817 cur_dev=0.09167728658047067
  - pos 13: Animal Collective - No More Runnin -> Orchid Mantis - Fog | base_T=0.6221370689484218 cur_T=0.6218079268021359 base_dev=0.0642896440993998 cur_dev=0.1772797749040546
  - pos 14: Orchid Mantis - Fog -> Animal Collective - No More Runnin | base_T=0.42184005046144696 cur_T=0.6731474696382737 base_dev=0.0824767435334669 cur_dev=0.03051338727118791
  - pos 15: Wild Nothing - Ride -> Toro Y Moi - Pavement | base_T=0.5511360093665991 cur_T=0.5650068566729678 base_dev=0.06725810102744806 cur_dev=0.12323588637544547
  - pos 17: David Bowie - Move On -> Beak> - Windmill Hill | base_T=0.5323843106013314 cur_T=0.4282159590746211 base_dev=0.05692913719206438 cur_dev=0.01632303513205824
  - pos 18: The Beach Boys - Please Let Me Wonder -> David Bowie - Move On | base_T=0.6492639969941926 cur_T=0.5825412052017636 base_dev=0.0029714226868551297 cur_dev=0.04875226538855748
  - pos 19: The Beatles - Little Queenie -> The Beach Boys - Please Let Me Wonder | base_T=0.6120196198266036 cur_T=0.6492639969941926 base_dev=0.16769208289827908 cur_dev=0.09183160868373252
  - pos 20: Ariel Pink's Haunted Graffiti - Early Birds of Babylon -> The Clientele - What Goes Up | base_T=0.76452337442273 cur_T=0.6516270384407491 base_dev=0.11634548897856112 cur_dev=0.14635109947455416
  - pos 24: Talking Heads - Life During Wartime -> No Age - Flutter Freer | base_T=0.7815785734114493 cur_T=0.8874990957333945 base_dev=0.3128306198220131 cur_dev=0.25178264270438616
  - pos 25: Fleetwood Mac - Go Your Own Way -> Talking Heads - Life During Wartime | base_T=0.6328555843932391 cur_T=0.7718437194466157 base_dev=0.24536285286463566 cur_dev=0.17409108680017032

- hash `0b52d779e87b8f00d8fdfd9214e5ae4695a3478c7860e74e5c8b5413b67978a5` label `arc_w1.5_t0.08_abs_m0.15_autoFalse_tb0.02_beam10` overlap=None
  - pos 2: King Krule - The Cadet Leaps -> Godspeed You Black Emperor! - Moya | base_T=0.7159177022106813 cur_T=0.6827701884564017 base_dev=0.4891376911737034 cur_dev=0.3088111470468307
  - pos 3: Mount Eerie - (soft air) -> Ezra Feinberg - Ovation | base_T=0.6195541647965571 cur_T=0.7039687690298055 base_dev=0.3329677950033614 cur_dev=0.17660515047671138
  - pos 4: Godspeed You! Black Emperor - Peasantry or ‘Light! Inside of Light!’ -> Dressed Up Animals - Mondtanz | base_T=0.6135803414188412 cur_T=0.8258495046404635 base_dev=0.17513777839063632 cur_dev=0.09885559488118478
  - pos 5: Jónsi & Alex Somers - All The Big Trees (2019 Remaster) -> King Krule - The Cadet Leaps | base_T=0.6889105761394906 cur_T=0.8067052528326448 base_dev=0.004146027090951188 cur_dev=0.027197924918060112
  - pos 7: Tim Hecker - Abduction -> M83 - Another Wave from You | base_T=0.4299009771685955 cur_T=0.78742005345353 base_dev=0.25262841683403703 cur_dev=0.2051618851620145
  - pos 8: Mary Lattimore & Walt Mcclements - The Poppies, the Wild Mustard, the Blue-Eyed Grass -> Spacemen 3 - Things'll Never Be the Same | base_T=0.7602331050266803 cur_T=0.8185518966682961 base_dev=0.3468678703016146 cur_dev=0.31972795188539627
  - pos 10: Jonny Nash - October Song -> Cornelius - Surfing on Mind Wave, Pt. 2 | base_T=0.7848286992525537 cur_T=0.8034951146243114 base_dev=0.1286568087201729 cur_dev=0.2072080595593752
  - pos 11: Labradford - V -> Microphones - Here With Summer | base_T=0.7699099714964436 cur_T=0.8398301230820251 base_dev=0.13082749692531817 cur_dev=0.0800313041791113
  - pos 12: TV On The Radio - Tonight -> PaPerCuts - Mattress on the Floor | base_T=0.6403422925030239 cur_T=0.8967572984165522 base_dev=0.059283005278525897 cur_dev=0.05568463355117209
  - pos 13: Animal Collective - No More Runnin -> Dirty Beaches - Sweet 17 | base_T=0.6221370689484218 cur_T=0.8756926284369257 base_dev=0.0642896440993998 cur_dev=0.02039670325270082
  - pos 14: Orchid Mantis - Fog -> Television Personalities - David Hockney's Diary | base_T=0.42184005046144696 cur_T=0.9142592763980442 base_dev=0.0824767435334669 cur_dev=0.17267414574460205
  - pos 15: Wild Nothing - Ride -> Ramones - Glad to See You Go (40th Anniversary Mix) | base_T=0.5511360093665991 cur_T=0.829455849593771 base_dev=0.06725810102744806 cur_dev=0.3090328786583877
  - pos 16: 00110100 01010100 - 0000 871 0017 -> David Bowie - Suffragette City | base_T=0.4464113325140045 cur_T=0.7586708660369917 base_dev=0.04419643724598976 cur_dev=None
  - pos 17: David Bowie - Move On -> No Age - Flutter Freer | base_T=0.5323843106013314 cur_T=0.8874990957333945 base_dev=0.05692913719206438 cur_dev=0.25178264270438616
  - pos 18: The Beach Boys - Please Let Me Wonder -> Here We Go Magic - Tunnelvision | base_T=0.6492639969941926 cur_T=0.9125919289384837 base_dev=0.0029714226868551297 cur_dev=0.1745801465890185
  - pos 19: The Beatles - Little Queenie -> The Flaming Lips - Look... the Sun Rising | base_T=0.6120196198266036 cur_T=0.8853380996895184 base_dev=0.16769208289827908 cur_dev=0.02528938066264863
  - pos 20: Ariel Pink's Haunted Graffiti - Early Birds of Babylon -> Sonic Youth w. Lydia Lunch - Death Valley '69 | base_T=0.76452337442273 cur_T=0.8999231595348618 base_dev=0.11634548897856112 cur_dev=0.06893628574899446
  - pos 21: King Krule - Emergency Blimp -> The Jimi Hendrix Experience - Voodoo Child (Slight Return) | base_T=0.6910258897661363 cur_T=0.8745940561886648 base_dev=0.2734558711528715 cur_dev=0.14922095930894064
  - pos 22: The Fall - Room to Live -> The Stooges - T.V. Eye | base_T=0.7332345871241958 cur_T=0.8266032228365947 base_dev=0.3125249117150196 cur_dev=0.3303533781562691
  - pos 23: David Bowie - Suffragette City -> David Bowie - Let's Dance | base_T=0.5975676238219465 cur_T=0.8660740780525283 base_dev=None cur_dev=None

- hash `0b52d779e87b8f00d8fdfd9214e5ae4695a3478c7860e74e5c8b5413b67978a5` label `arc_w1.5_t0.08_abs_m0.15_autoTrue_tb0.02_beam10` overlap=None
  - pos 2: King Krule - The Cadet Leaps -> Godspeed You Black Emperor! - Moya | base_T=0.7159177022106813 cur_T=0.6827701884564017 base_dev=0.4891376911737034 cur_dev=0.3088111470468307
  - pos 3: Mount Eerie - (soft air) -> Ezra Feinberg - Ovation | base_T=0.6195541647965571 cur_T=0.7039687690298055 base_dev=0.3329677950033614 cur_dev=0.17660515047671138
  - pos 4: Godspeed You! Black Emperor - Peasantry or ‘Light! Inside of Light!’ -> Dressed Up Animals - Mondtanz | base_T=0.6135803414188412 cur_T=0.8258495046404635 base_dev=0.17513777839063632 cur_dev=0.09885559488118478
  - pos 5: Jónsi & Alex Somers - All The Big Trees (2019 Remaster) -> King Krule - The Cadet Leaps | base_T=0.6889105761394906 cur_T=0.8067052528326448 base_dev=0.004146027090951188 cur_dev=0.027197924918060112
  - pos 7: Tim Hecker - Abduction -> M83 - Another Wave from You | base_T=0.4299009771685955 cur_T=0.78742005345353 base_dev=0.25262841683403703 cur_dev=0.2051618851620145
  - pos 8: Mary Lattimore & Walt Mcclements - The Poppies, the Wild Mustard, the Blue-Eyed Grass -> Spacemen 3 - Things'll Never Be the Same | base_T=0.7602331050266803 cur_T=0.8185518966682961 base_dev=0.3468678703016146 cur_dev=0.31972795188539627
  - pos 10: Jonny Nash - October Song -> Cornelius - Surfing on Mind Wave, Pt. 2 | base_T=0.7848286992525537 cur_T=0.8034951146243114 base_dev=0.1286568087201729 cur_dev=0.2072080595593752
  - pos 11: Labradford - V -> Microphones - Here With Summer | base_T=0.7699099714964436 cur_T=0.8398301230820251 base_dev=0.13082749692531817 cur_dev=0.0800313041791113
  - pos 12: TV On The Radio - Tonight -> PaPerCuts - Mattress on the Floor | base_T=0.6403422925030239 cur_T=0.8967572984165522 base_dev=0.059283005278525897 cur_dev=0.05568463355117209
  - pos 13: Animal Collective - No More Runnin -> Dirty Beaches - Sweet 17 | base_T=0.6221370689484218 cur_T=0.8756926284369257 base_dev=0.0642896440993998 cur_dev=0.02039670325270082
  - pos 14: Orchid Mantis - Fog -> Television Personalities - David Hockney's Diary | base_T=0.42184005046144696 cur_T=0.9142592763980442 base_dev=0.0824767435334669 cur_dev=0.17267414574460205
  - pos 15: Wild Nothing - Ride -> Ramones - Glad to See You Go (40th Anniversary Mix) | base_T=0.5511360093665991 cur_T=0.829455849593771 base_dev=0.06725810102744806 cur_dev=0.3090328786583877
  - pos 16: 00110100 01010100 - 0000 871 0017 -> David Bowie - Suffragette City | base_T=0.4464113325140045 cur_T=0.7586708660369917 base_dev=0.04419643724598976 cur_dev=None
  - pos 17: David Bowie - Move On -> No Age - Flutter Freer | base_T=0.5323843106013314 cur_T=0.8874990957333945 base_dev=0.05692913719206438 cur_dev=0.25178264270438616
  - pos 18: The Beach Boys - Please Let Me Wonder -> Here We Go Magic - Tunnelvision | base_T=0.6492639969941926 cur_T=0.9125919289384837 base_dev=0.0029714226868551297 cur_dev=0.1745801465890185
  - pos 19: The Beatles - Little Queenie -> The Flaming Lips - Look... the Sun Rising | base_T=0.6120196198266036 cur_T=0.8853380996895184 base_dev=0.16769208289827908 cur_dev=0.02528938066264863
  - pos 20: Ariel Pink's Haunted Graffiti - Early Birds of Babylon -> Sonic Youth w. Lydia Lunch - Death Valley '69 | base_T=0.76452337442273 cur_T=0.8999231595348618 base_dev=0.11634548897856112 cur_dev=0.06893628574899446
  - pos 21: King Krule - Emergency Blimp -> The Jimi Hendrix Experience - Voodoo Child (Slight Return) | base_T=0.6910258897661363 cur_T=0.8745940561886648 base_dev=0.2734558711528715 cur_dev=0.14922095930894064
  - pos 22: The Fall - Room to Live -> The Stooges - T.V. Eye | base_T=0.7332345871241958 cur_T=0.8266032228365947 base_dev=0.3125249117150196 cur_dev=0.3303533781562691
  - pos 23: David Bowie - Suffragette City -> David Bowie - Let's Dance | base_T=0.5975676238219465 cur_T=0.8660740780525283 base_dev=None cur_dev=None

- hash `1061bdccc727c3c538084f067e98ec494af5b755f34ebdafeb1bf9499e8e2baf` label `arc_w1.5_t0.08_abs_m0.1_autoTrue_tbNone_beamNone` overlap=None
  - pos 2: King Krule - The Cadet Leaps -> Godspeed You Black Emperor! - Moya | base_T=0.7159177022106813 cur_T=0.6827701884564017 base_dev=0.4891376911737034 cur_dev=0.3088111470468307
  - pos 5: Jónsi & Alex Somers - All The Big Trees (2019 Remaster) -> King Krule - The Cadet Leaps | base_T=0.6889105761394906 cur_T=0.6955376759430785 base_dev=0.004146027090951188 cur_dev=0.027197924918060112
  - pos 7: Tim Hecker - Abduction -> Foxes in Fiction - Insomnia Keys | base_T=0.4299009771685955 cur_T=0.7853552801920686 base_dev=0.25262841683403703 cur_dev=0.18482013225565375
  - pos 8: Mary Lattimore & Walt Mcclements - The Poppies, the Wild Mustard, the Blue-Eyed Grass -> Spacemen 3 - Things'll Never Be the Same | base_T=0.7602331050266803 cur_T=0.6712457107645613 base_dev=0.3468678703016146 cur_dev=0.31972795188539627
  - pos 10: Jonny Nash - October Song -> Labradford - V | base_T=0.7848286992525537 cur_T=0.6946432657992841 base_dev=0.1286568087201729 cur_dev=0.1678070190650204
  - pos 11: Labradford - V -> Jonny Nash - October Song | base_T=0.7699099714964436 cur_T=0.8502745376085842 base_dev=0.13082749692531817 cur_dev=0.09167728658047067
  - pos 13: Animal Collective - No More Runnin -> Orchid Mantis - Fog | base_T=0.6221370689484218 cur_T=0.6218079268021359 base_dev=0.0642896440993998 cur_dev=0.1772797749040546
  - pos 14: Orchid Mantis - Fog -> Animal Collective - No More Runnin | base_T=0.42184005046144696 cur_T=0.6731474696382737 base_dev=0.0824767435334669 cur_dev=0.03051338727118791
  - pos 15: Wild Nothing - Ride -> Toro Y Moi - Pavement | base_T=0.5511360093665991 cur_T=0.5650068566729678 base_dev=0.06725810102744806 cur_dev=0.12323588637544547
  - pos 19: The Beatles - Little Queenie -> Ariel Pink's Haunted Graffiti - Early Birds of Babylon | base_T=0.6120196198266036 cur_T=0.6154926046838209 base_dev=0.16769208289827908 cur_dev=0.03717464867391296
  - pos 20: Ariel Pink's Haunted Graffiti - Early Birds of Babylon -> The Clientele - What Goes Up | base_T=0.76452337442273 cur_T=0.7327194244057653 base_dev=0.11634548897856112 cur_dev=0.14635109947455416
  - pos 24: Talking Heads - Life During Wartime -> No Age - Flutter Freer | base_T=0.7815785734114493 cur_T=0.8874990957333945 base_dev=0.3128306198220131 cur_dev=0.25178264270438616
  - pos 25: Fleetwood Mac - Go Your Own Way -> Talking Heads - Life During Wartime | base_T=0.6328555843932391 cur_T=0.7718437194466157 base_dev=0.24536285286463566 cur_dev=0.17409108680017032
  - pos 26: Art Feynman - Early Signs of Rhythm -> The Fall - No Xmas For John Key (Peel Session) | base_T=0.8753313515770027 cur_T=0.8430552344999207 base_dev=0.1426792760465157 cur_dev=0.09241799435977271

- hash `4d8cf77a275dccb03f5aa8608bcbddfb12cd3ad87ff32e1ce6cc780c5adf31bf` label `arc_w1.5_t0.0_abs_m0.2_autoFalse_tb0.02_beamNone` overlap=None
  - pos 2: King Krule - The Cadet Leaps -> Godspeed You Black Emperor! - Moya | base_T=0.7159177022106813 cur_T=0.6827701884564017 base_dev=0.4891376911737034 cur_dev=0.3088111470468307
  - pos 5: Jónsi & Alex Somers - All The Big Trees (2019 Remaster) -> King Krule - The Cadet Leaps | base_T=0.6889105761394906 cur_T=0.6955376759430785 base_dev=0.004146027090951188 cur_dev=0.027197924918060112
  - pos 7: Tim Hecker - Abduction -> Foxes in Fiction - Insomnia Keys | base_T=0.4299009771685955 cur_T=0.7853552801920686 base_dev=0.25262841683403703 cur_dev=0.18482013225565375
  - pos 8: Mary Lattimore & Walt Mcclements - The Poppies, the Wild Mustard, the Blue-Eyed Grass -> Spacemen 3 - Things'll Never Be the Same | base_T=0.7602331050266803 cur_T=0.6712457107645613 base_dev=0.3468678703016146 cur_dev=0.31972795188539627
  - pos 10: Jonny Nash - October Song -> Labradford - V | base_T=0.7848286992525537 cur_T=0.6946432657992841 base_dev=0.1286568087201729 cur_dev=0.1678070190650204
  - pos 11: Labradford - V -> Jonny Nash - October Song | base_T=0.7699099714964436 cur_T=0.8502745376085842 base_dev=0.13082749692531817 cur_dev=0.09167728658047067
  - pos 14: Orchid Mantis - Fog -> Toro Y Moi - Pavement | base_T=0.42184005046144696 cur_T=0.5650068566729678 base_dev=0.0824767435334669 cur_dev=0.017554483794823605
  - pos 15: Wild Nothing - Ride -> Autechre - r cazt | base_T=0.5511360093665991 cur_T=0.7838957812645864 base_dev=0.06725810102744806 cur_dev=0.041108802196430705
  - pos 17: David Bowie - Move On -> Beak> - Windmill Hill | base_T=0.5323843106013314 cur_T=0.4282159590746211 base_dev=0.05692913719206438 cur_dev=0.01632303513205824
  - pos 18: The Beach Boys - Please Let Me Wonder -> David Bowie - Move On | base_T=0.6492639969941926 cur_T=0.5825412052017636 base_dev=0.0029714226868551297 cur_dev=0.04875226538855748
  - pos 19: The Beatles - Little Queenie -> The Beach Boys - Please Let Me Wonder | base_T=0.6120196198266036 cur_T=0.6492639969941926 base_dev=0.16769208289827908 cur_dev=0.09183160868373252
  - pos 20: Ariel Pink's Haunted Graffiti - Early Birds of Babylon -> The Clientele - What Goes Up | base_T=0.76452337442273 cur_T=0.6516270384407491 base_dev=0.11634548897856112 cur_dev=0.14635109947455416
  - pos 24: Talking Heads - Life During Wartime -> No Age - Flutter Freer | base_T=0.7815785734114493 cur_T=0.8874990957333945 base_dev=0.3128306198220131 cur_dev=0.25178264270438616
  - pos 25: Fleetwood Mac - Go Your Own Way -> Talking Heads - Life During Wartime | base_T=0.6328555843932391 cur_T=0.7718437194466157 base_dev=0.24536285286463566 cur_dev=0.17409108680017032

- hash `28e35d57757e1fa797ddf512329affbaf0c4af10103b2730b66c66e5904309ff` label `arc_w1.5_t0.12_abs_m0.15_autoFalse_tb0.02_beam10` overlap=None
  - pos 2: King Krule - The Cadet Leaps -> Godspeed You Black Emperor! - Moya | base_T=0.7159177022106813 cur_T=0.6827701884564017 base_dev=0.4891376911737034 cur_dev=0.3088111470468307
  - pos 3: Mount Eerie - (soft air) -> Ezra Feinberg - Ovation | base_T=0.6195541647965571 cur_T=0.7039687690298055 base_dev=0.3329677950033614 cur_dev=0.17660515047671138
  - pos 4: Godspeed You! Black Emperor - Peasantry or ‘Light! Inside of Light!’ -> Dressed Up Animals - Mondtanz | base_T=0.6135803414188412 cur_T=0.8258495046404635 base_dev=0.17513777839063632 cur_dev=0.09885559488118478
  - pos 5: Jónsi & Alex Somers - All The Big Trees (2019 Remaster) -> King Krule - The Cadet Leaps | base_T=0.6889105761394906 cur_T=0.8067052528326448 base_dev=0.004146027090951188 cur_dev=0.027197924918060112
  - pos 7: Tim Hecker - Abduction -> M83 - Another Wave from You | base_T=0.4299009771685955 cur_T=0.78742005345353 base_dev=0.25262841683403703 cur_dev=0.2051618851620145
  - pos 8: Mary Lattimore & Walt Mcclements - The Poppies, the Wild Mustard, the Blue-Eyed Grass -> Spacemen 3 - Things'll Never Be the Same | base_T=0.7602331050266803 cur_T=0.8185518966682961 base_dev=0.3468678703016146 cur_dev=0.31972795188539627
  - pos 10: Jonny Nash - October Song -> Cornelius - Surfing on Mind Wave, Pt. 2 | base_T=0.7848286992525537 cur_T=0.8034951146243114 base_dev=0.1286568087201729 cur_dev=0.2072080595593752
  - pos 11: Labradford - V -> Microphones - Here With Summer | base_T=0.7699099714964436 cur_T=0.8398301230820251 base_dev=0.13082749692531817 cur_dev=0.0800313041791113
  - pos 12: TV On The Radio - Tonight -> PaPerCuts - Mattress on the Floor | base_T=0.6403422925030239 cur_T=0.8967572984165522 base_dev=0.059283005278525897 cur_dev=0.05568463355117209
  - pos 13: Animal Collective - No More Runnin -> Dirty Beaches - Sweet 17 | base_T=0.6221370689484218 cur_T=0.8756926284369257 base_dev=0.0642896440993998 cur_dev=0.02039670325270082
  - pos 14: Orchid Mantis - Fog -> Television Personalities - David Hockney's Diary | base_T=0.42184005046144696 cur_T=0.9142592763980442 base_dev=0.0824767435334669 cur_dev=0.17267414574460205
  - pos 15: Wild Nothing - Ride -> Ramones - Glad to See You Go (40th Anniversary Mix) | base_T=0.5511360093665991 cur_T=0.829455849593771 base_dev=0.06725810102744806 cur_dev=0.3090328786583877
  - pos 16: 00110100 01010100 - 0000 871 0017 -> David Bowie - Suffragette City | base_T=0.4464113325140045 cur_T=0.7586708660369917 base_dev=0.04419643724598976 cur_dev=None
  - pos 17: David Bowie - Move On -> No Age - Flutter Freer | base_T=0.5323843106013314 cur_T=0.8874990957333945 base_dev=0.05692913719206438 cur_dev=0.25178264270438616
  - pos 18: The Beach Boys - Please Let Me Wonder -> Talking Heads - Life During Wartime | base_T=0.6492639969941926 cur_T=0.7718437194466157 base_dev=0.0029714226868551297 cur_dev=0.17409108680017032
  - pos 19: The Beatles - Little Queenie -> The Fall - No Xmas For John Key (Peel Session) | base_T=0.6120196198266036 cur_T=0.8430552344999207 base_dev=0.16769208289827908 cur_dev=0.09241799435977271
  - pos 20: Ariel Pink's Haunted Graffiti - Early Birds of Babylon -> Sonic Youth w. Lydia Lunch - Death Valley '69 | base_T=0.76452337442273 cur_T=0.8955575758212249 base_dev=0.11634548897856112 cur_dev=0.06893628574899446
  - pos 21: King Krule - Emergency Blimp -> The Jimi Hendrix Experience - Voodoo Child (Slight Return) | base_T=0.6910258897661363 cur_T=0.8745940561886648 base_dev=0.2734558711528715 cur_dev=0.14922095930894064
  - pos 22: The Fall - Room to Live -> The Stooges - T.V. Eye | base_T=0.7332345871241958 cur_T=0.8266032228365947 base_dev=0.3125249117150196 cur_dev=0.3303533781562691
  - pos 23: David Bowie - Suffragette City -> David Bowie - Let's Dance | base_T=0.5975676238219465 cur_T=0.8660740780525283 base_dev=None cur_dev=None

- hash `214b0e9deb4bcedcc27fd5f010f2b3dc5e13cae3df643fa4d0787a6e41b53afa` label `arc_w1.5_t0.2_abs_m0.1_autoTrue_tbNone_beam25` overlap=None
  - pos 2: King Krule - The Cadet Leaps -> Ezra Feinberg - Ovation | base_T=0.7159177022106813 cur_T=0.6519174905794566 base_dev=0.4891376911737034 cur_dev=0.28499152613908096
  - pos 3: Mount Eerie - (soft air) -> Dressed Up Animals - Mondtanz | base_T=0.6195541647965571 cur_T=0.8258495046404635 base_dev=0.3329677950033614 cur_dev=0.26106726929191365
  - pos 4: Godspeed You! Black Emperor - Peasantry or ‘Light! Inside of Light!’ -> King Krule - The Cadet Leaps | base_T=0.6135803414188412 cur_T=0.8067052528326448 base_dev=0.17513777839063632 cur_dev=0.21853964110060498
  - pos 5: Jónsi & Alex Somers - All The Big Trees (2019 Remaster) -> Alex Somers - Looking After | base_T=0.6889105761394906 cur_T=0.8694433726555622 base_dev=0.004146027090951188 cur_dev=0.08732359452242228
  - pos 6: Sigur Rós - Andrá -> Tim Hecker - Abduction | base_T=0.6484424356689317 cur_T=0.8464609468686048 base_dev=0.07191639938006988 cur_dev=0.0904167424233081
  - pos 7: Tim Hecker - Abduction -> Sigur Rós - Andrá | base_T=0.4299009771685955 cur_T=0.7452949556047597 base_dev=0.25262841683403703 cur_dev=0.2341280737907988
  - pos 8: Mary Lattimore & Walt Mcclements - The Poppies, the Wild Mustard, the Blue-Eyed Grass -> M83 - Another Wave from You | base_T=0.7602331050266803 cur_T=0.78742005345353 base_dev=0.3468678703016146 cur_dev=0.31354826082438414
  - pos 10: Jonny Nash - October Song -> Microphones - Here With Summer | base_T=0.7848286992525537 cur_T=0.6552326095259688 base_dev=0.1286568087201729 cur_dev=0.21877083720095408
  - pos 11: Labradford - V -> PaPerCuts - Mattress on the Floor | base_T=0.7699099714964436 cur_T=0.8967572984165522 base_dev=0.13082749692531817 cur_dev=0.25616906750238166
  - pos 12: TV On The Radio - Tonight -> No Age - A Ceiling Dreams of a Floor | base_T=0.6403422925030239 cur_T=0.9139005290702163 base_dev=0.059283005278525897 cur_dev=0.10578747777961578
  - pos 13: Animal Collective - No More Runnin -> Dirty Beaches - Sweet 17 | base_T=0.6221370689484218 cur_T=0.8957787584269118 base_dev=0.0642896440993998 cur_dev=0.02039670325270082
  - pos 14: Orchid Mantis - Fog -> Television Personalities - David Hockney's Diary | base_T=0.42184005046144696 cur_T=0.9142592763980442 base_dev=0.0824767435334669 cur_dev=0.17267414574460205
  - pos 15: Wild Nothing - Ride -> Ramones - Glad to See You Go (40th Anniversary Mix) | base_T=0.5511360093665991 cur_T=0.829455849593771 base_dev=0.06725810102744806 cur_dev=0.3090328786583877
  - pos 16: 00110100 01010100 - 0000 871 0017 -> David Bowie - Suffragette City | base_T=0.4464113325140045 cur_T=0.7586708660369917 base_dev=0.04419643724598976 cur_dev=None
  - pos 17: David Bowie - Move On -> No Age - Flutter Freer | base_T=0.5323843106013314 cur_T=0.8874990957333945 base_dev=0.05692913719206438 cur_dev=0.25178264270438616
  - pos 18: The Beach Boys - Please Let Me Wonder -> Talking Heads - Life During Wartime | base_T=0.6492639969941926 cur_T=0.7718437194466157 base_dev=0.0029714226868551297 cur_dev=0.17409108680017032
  - pos 19: The Beatles - Little Queenie -> The Fall - No Xmas For John Key (Peel Session) | base_T=0.6120196198266036 cur_T=0.8430552344999207 base_dev=0.16769208289827908 cur_dev=0.09241799435977271
  - pos 20: Ariel Pink's Haunted Graffiti - Early Birds of Babylon -> Sonic Youth w. Lydia Lunch - Death Valley '69 | base_T=0.76452337442273 cur_T=0.8955575758212249 base_dev=0.11634548897856112 cur_dev=0.06893628574899446
  - pos 21: King Krule - Emergency Blimp -> The Jimi Hendrix Experience - Voodoo Child (Slight Return) | base_T=0.6910258897661363 cur_T=0.8745940561886648 base_dev=0.2734558711528715 cur_dev=0.14922095930894064
  - pos 22: The Fall - Room to Live -> The Stooges - T.V. Eye | base_T=0.7332345871241958 cur_T=0.8266032228365947 base_dev=0.3125249117150196 cur_dev=0.3303533781562691

- hash `789e86b9fdaaeb382901d9c84a01c8067ea4fc1629987dccdd39a2c7e818d29d` label `arc_w2.0_t0.05_abs_m0.1_autoFalse_tb0.02_beam50` overlap=None
  - pos 2: King Krule - The Cadet Leaps -> Godspeed You Black Emperor! - Moya | base_T=0.7159177022106813 cur_T=0.6827701884564017 base_dev=0.4891376911737034 cur_dev=0.3088111470468307
  - pos 3: Mount Eerie - (soft air) -> Ezra Feinberg - Ovation | base_T=0.6195541647965571 cur_T=0.7039687690298055 base_dev=0.3329677950033614 cur_dev=0.17660515047671138
  - pos 4: Godspeed You! Black Emperor - Peasantry or ‘Light! Inside of Light!’ -> Dressed Up Animals - Mondtanz | base_T=0.6135803414188412 cur_T=0.8258495046404635 base_dev=0.17513777839063632 cur_dev=0.09885559488118478
  - pos 5: Jónsi & Alex Somers - All The Big Trees (2019 Remaster) -> King Krule - The Cadet Leaps | base_T=0.6889105761394906 cur_T=0.8067052528326448 base_dev=0.004146027090951188 cur_dev=0.027197924918060112
  - pos 6: Sigur Rós - Andrá -> Ólafur Arnalds - The Bottom Line | base_T=0.6484424356689317 cur_T=0.8535193140570737 base_dev=0.07191639938006988 cur_dev=0.04651712824524801
  - pos 7: Tim Hecker - Abduction -> Sigur Rós - Andrá | base_T=0.4299009771685955 cur_T=0.8192691191030009 base_dev=0.25262841683403703 cur_dev=0.2341280737907988
  - pos 8: Mary Lattimore & Walt Mcclements - The Poppies, the Wild Mustard, the Blue-Eyed Grass -> M83 - Another Wave from You | base_T=0.7602331050266803 cur_T=0.78742005345353 base_dev=0.3468678703016146 cur_dev=0.31354826082438414
  - pos 10: Jonny Nash - October Song -> Cornelius - Surfing on Mind Wave, Pt. 2 | base_T=0.7848286992525537 cur_T=0.8034951146243114 base_dev=0.1286568087201729 cur_dev=0.2072080595593752
  - pos 11: Labradford - V -> Microphones - Here With Summer | base_T=0.7699099714964436 cur_T=0.8398301230820251 base_dev=0.13082749692531817 cur_dev=0.0800313041791113
  - pos 12: TV On The Radio - Tonight -> PaPerCuts - Mattress on the Floor | base_T=0.6403422925030239 cur_T=0.8967572984165522 base_dev=0.059283005278525897 cur_dev=0.05568463355117209
  - pos 13: Animal Collective - No More Runnin -> Dirty Beaches - Sweet 17 | base_T=0.6221370689484218 cur_T=0.8756926284369257 base_dev=0.0642896440993998 cur_dev=0.02039670325270082
  - pos 14: Orchid Mantis - Fog -> Television Personalities - David Hockney's Diary | base_T=0.42184005046144696 cur_T=0.9142592763980442 base_dev=0.0824767435334669 cur_dev=0.17267414574460205
  - pos 15: Wild Nothing - Ride -> Ramones - Glad to See You Go (40th Anniversary Mix) | base_T=0.5511360093665991 cur_T=0.829455849593771 base_dev=0.06725810102744806 cur_dev=0.3090328786583877
  - pos 16: 00110100 01010100 - 0000 871 0017 -> David Bowie - Suffragette City | base_T=0.4464113325140045 cur_T=0.7586708660369917 base_dev=0.04419643724598976 cur_dev=None
  - pos 17: David Bowie - Move On -> New Order - Sooner Than You Think | base_T=0.5323843106013314 cur_T=0.8302902189542899 base_dev=0.05692913719206438 cur_dev=0.28003484210896434
  - pos 18: The Beach Boys - Please Let Me Wonder -> No Age - Flutter Freer | base_T=0.6492639969941926 cur_T=0.9424741729013337 base_dev=0.0029714226868551297 cur_dev=0.11304310968254339
  - pos 19: The Beatles - Little Queenie -> Sonic Youth - Hot Wire My Heart | base_T=0.6120196198266036 cur_T=0.9353409635913639 base_dev=0.16769208289827908 cur_dev=0.048094857108766154
  - pos 20: Ariel Pink's Haunted Graffiti - Early Birds of Babylon -> Sonic Youth w. Lydia Lunch - Death Valley '69 | base_T=0.76452337442273 cur_T=0.9226276644427387 base_dev=0.11634548897856112 cur_dev=0.06893628574899446
  - pos 21: King Krule - Emergency Blimp -> The Jimi Hendrix Experience - Voodoo Child (Slight Return) | base_T=0.6910258897661363 cur_T=0.8745940561886648 base_dev=0.2734558711528715 cur_dev=0.14922095930894064
  - pos 22: The Fall - Room to Live -> The Stooges - T.V. Eye | base_T=0.7332345871241958 cur_T=0.8266032228365947 base_dev=0.3125249117150196 cur_dev=0.3303533781562691

- hash `72e98d3273a9e7423a6eeda63aeefe47abfc182a569dab4e13c7b8202aadb068` label `arc_w2.0_t0.05_abs_m0.1_autoTrue_tbNone_beamNone` overlap=None
  - pos 2: King Krule - The Cadet Leaps -> Godspeed You Black Emperor! - Moya | base_T=0.7159177022106813 cur_T=0.6827701884564017 base_dev=0.4891376911737034 cur_dev=0.3088111470468307
  - pos 5: Jónsi & Alex Somers - All The Big Trees (2019 Remaster) -> King Krule - The Cadet Leaps | base_T=0.6889105761394906 cur_T=0.6955376759430785 base_dev=0.004146027090951188 cur_dev=0.027197924918060112
  - pos 7: Tim Hecker - Abduction -> Foxes in Fiction - Insomnia Keys | base_T=0.4299009771685955 cur_T=0.7853552801920686 base_dev=0.25262841683403703 cur_dev=0.18482013225565375
  - pos 8: Mary Lattimore & Walt Mcclements - The Poppies, the Wild Mustard, the Blue-Eyed Grass -> Spacemen 3 - Things'll Never Be the Same | base_T=0.7602331050266803 cur_T=0.6712457107645613 base_dev=0.3468678703016146 cur_dev=0.31972795188539627
  - pos 10: Jonny Nash - October Song -> Labradford - V | base_T=0.7848286992525537 cur_T=0.6946432657992841 base_dev=0.1286568087201729 cur_dev=0.1678070190650204
  - pos 11: Labradford - V -> Jonny Nash - October Song | base_T=0.7699099714964436 cur_T=0.8502745376085842 base_dev=0.13082749692531817 cur_dev=0.09167728658047067
  - pos 13: Animal Collective - No More Runnin -> Orchid Mantis - Fog | base_T=0.6221370689484218 cur_T=0.6218079268021359 base_dev=0.0642896440993998 cur_dev=0.1772797749040546
  - pos 14: Orchid Mantis - Fog -> Animal Collective - No More Runnin | base_T=0.42184005046144696 cur_T=0.6731474696382737 base_dev=0.0824767435334669 cur_dev=0.03051338727118791
  - pos 15: Wild Nothing - Ride -> Toro Y Moi - Pavement | base_T=0.5511360093665991 cur_T=0.5650068566729678 base_dev=0.06725810102744806 cur_dev=0.12323588637544547
  - pos 17: David Bowie - Move On -> Beak> - Windmill Hill | base_T=0.5323843106013314 cur_T=0.4282159590746211 base_dev=0.05692913719206438 cur_dev=0.01632303513205824
  - pos 18: The Beach Boys - Please Let Me Wonder -> David Bowie - Move On | base_T=0.6492639969941926 cur_T=0.5825412052017636 base_dev=0.0029714226868551297 cur_dev=0.04875226538855748
  - pos 19: The Beatles - Little Queenie -> The Beach Boys - Please Let Me Wonder | base_T=0.6120196198266036 cur_T=0.6492639969941926 base_dev=0.16769208289827908 cur_dev=0.09183160868373252
  - pos 20: Ariel Pink's Haunted Graffiti - Early Birds of Babylon -> The Clientele - What Goes Up | base_T=0.76452337442273 cur_T=0.6516270384407491 base_dev=0.11634548897856112 cur_dev=0.14635109947455416
  - pos 24: Talking Heads - Life During Wartime -> No Age - Flutter Freer | base_T=0.7815785734114493 cur_T=0.8874990957333945 base_dev=0.3128306198220131 cur_dev=0.25178264270438616
  - pos 25: Fleetwood Mac - Go Your Own Way -> Talking Heads - Life During Wartime | base_T=0.6328555843932391 cur_T=0.7718437194466157 base_dev=0.24536285286463566 cur_dev=0.17409108680017032
  - pos 26: Art Feynman - Early Signs of Rhythm -> Fleetwood Mac - Go Your Own Way | base_T=0.8753313515770027 cur_T=0.6328555843932391 base_dev=0.1426792760465157 cur_dev=0.04487841891342609

- hash `1061bdccc727c3c538084f067e98ec494af5b755f34ebdafeb1bf9499e8e2baf` label `arc_w2.0_t0.08_abs_m0.1_autoFalse_tbNone_beamNone` overlap=None
  - pos 2: King Krule - The Cadet Leaps -> Godspeed You Black Emperor! - Moya | base_T=0.7159177022106813 cur_T=0.6827701884564017 base_dev=0.4891376911737034 cur_dev=0.3088111470468307
  - pos 5: Jónsi & Alex Somers - All The Big Trees (2019 Remaster) -> King Krule - The Cadet Leaps | base_T=0.6889105761394906 cur_T=0.6955376759430785 base_dev=0.004146027090951188 cur_dev=0.027197924918060112
  - pos 7: Tim Hecker - Abduction -> Foxes in Fiction - Insomnia Keys | base_T=0.4299009771685955 cur_T=0.7853552801920686 base_dev=0.25262841683403703 cur_dev=0.18482013225565375
  - pos 8: Mary Lattimore & Walt Mcclements - The Poppies, the Wild Mustard, the Blue-Eyed Grass -> Spacemen 3 - Things'll Never Be the Same | base_T=0.7602331050266803 cur_T=0.6712457107645613 base_dev=0.3468678703016146 cur_dev=0.31972795188539627
  - pos 10: Jonny Nash - October Song -> Labradford - V | base_T=0.7848286992525537 cur_T=0.6946432657992841 base_dev=0.1286568087201729 cur_dev=0.1678070190650204
  - pos 11: Labradford - V -> Jonny Nash - October Song | base_T=0.7699099714964436 cur_T=0.8502745376085842 base_dev=0.13082749692531817 cur_dev=0.09167728658047067
  - pos 13: Animal Collective - No More Runnin -> Orchid Mantis - Fog | base_T=0.6221370689484218 cur_T=0.6218079268021359 base_dev=0.0642896440993998 cur_dev=0.1772797749040546
  - pos 14: Orchid Mantis - Fog -> Animal Collective - No More Runnin | base_T=0.42184005046144696 cur_T=0.6731474696382737 base_dev=0.0824767435334669 cur_dev=0.03051338727118791
  - pos 15: Wild Nothing - Ride -> Toro Y Moi - Pavement | base_T=0.5511360093665991 cur_T=0.5650068566729678 base_dev=0.06725810102744806 cur_dev=0.12323588637544547
  - pos 19: The Beatles - Little Queenie -> Ariel Pink's Haunted Graffiti - Early Birds of Babylon | base_T=0.6120196198266036 cur_T=0.6154926046838209 base_dev=0.16769208289827908 cur_dev=0.03717464867391296
  - pos 20: Ariel Pink's Haunted Graffiti - Early Birds of Babylon -> The Clientele - What Goes Up | base_T=0.76452337442273 cur_T=0.7327194244057653 base_dev=0.11634548897856112 cur_dev=0.14635109947455416
  - pos 24: Talking Heads - Life During Wartime -> No Age - Flutter Freer | base_T=0.7815785734114493 cur_T=0.8874990957333945 base_dev=0.3128306198220131 cur_dev=0.25178264270438616
  - pos 25: Fleetwood Mac - Go Your Own Way -> Talking Heads - Life During Wartime | base_T=0.6328555843932391 cur_T=0.7718437194466157 base_dev=0.24536285286463566 cur_dev=0.17409108680017032
  - pos 26: Art Feynman - Early Signs of Rhythm -> The Fall - No Xmas For John Key (Peel Session) | base_T=0.8753313515770027 cur_T=0.8430552344999207 base_dev=0.1426792760465157 cur_dev=0.09241799435977271

- hash `789e86b9fdaaeb382901d9c84a01c8067ea4fc1629987dccdd39a2c7e818d29d` label `arc_w2.0_t0.08_abs_m0.4_autoFalse_tbNone_beam50` overlap=None
  - pos 2: King Krule - The Cadet Leaps -> Godspeed You Black Emperor! - Moya | base_T=0.7159177022106813 cur_T=0.6827701884564017 base_dev=0.4891376911737034 cur_dev=0.3088111470468307
  - pos 3: Mount Eerie - (soft air) -> Ezra Feinberg - Ovation | base_T=0.6195541647965571 cur_T=0.7039687690298055 base_dev=0.3329677950033614 cur_dev=0.17660515047671138
  - pos 4: Godspeed You! Black Emperor - Peasantry or ‘Light! Inside of Light!’ -> Dressed Up Animals - Mondtanz | base_T=0.6135803414188412 cur_T=0.8258495046404635 base_dev=0.17513777839063632 cur_dev=0.09885559488118478
  - pos 5: Jónsi & Alex Somers - All The Big Trees (2019 Remaster) -> King Krule - The Cadet Leaps | base_T=0.6889105761394906 cur_T=0.8067052528326448 base_dev=0.004146027090951188 cur_dev=0.027197924918060112
  - pos 6: Sigur Rós - Andrá -> Ólafur Arnalds - The Bottom Line | base_T=0.6484424356689317 cur_T=0.8535193140570737 base_dev=0.07191639938006988 cur_dev=0.04651712824524801
  - pos 7: Tim Hecker - Abduction -> Sigur Rós - Andrá | base_T=0.4299009771685955 cur_T=0.8192691191030009 base_dev=0.25262841683403703 cur_dev=0.2341280737907988
  - pos 8: Mary Lattimore & Walt Mcclements - The Poppies, the Wild Mustard, the Blue-Eyed Grass -> M83 - Another Wave from You | base_T=0.7602331050266803 cur_T=0.78742005345353 base_dev=0.3468678703016146 cur_dev=0.31354826082438414
  - pos 10: Jonny Nash - October Song -> Cornelius - Surfing on Mind Wave, Pt. 2 | base_T=0.7848286992525537 cur_T=0.8034951146243114 base_dev=0.1286568087201729 cur_dev=0.2072080595593752
  - pos 11: Labradford - V -> Microphones - Here With Summer | base_T=0.7699099714964436 cur_T=0.8398301230820251 base_dev=0.13082749692531817 cur_dev=0.0800313041791113
  - pos 12: TV On The Radio - Tonight -> PaPerCuts - Mattress on the Floor | base_T=0.6403422925030239 cur_T=0.8967572984165522 base_dev=0.059283005278525897 cur_dev=0.05568463355117209
  - pos 13: Animal Collective - No More Runnin -> Dirty Beaches - Sweet 17 | base_T=0.6221370689484218 cur_T=0.8756926284369257 base_dev=0.0642896440993998 cur_dev=0.02039670325270082
  - pos 14: Orchid Mantis - Fog -> Television Personalities - David Hockney's Diary | base_T=0.42184005046144696 cur_T=0.9142592763980442 base_dev=0.0824767435334669 cur_dev=0.17267414574460205
  - pos 15: Wild Nothing - Ride -> Ramones - Glad to See You Go (40th Anniversary Mix) | base_T=0.5511360093665991 cur_T=0.829455849593771 base_dev=0.06725810102744806 cur_dev=0.3090328786583877
  - pos 16: 00110100 01010100 - 0000 871 0017 -> David Bowie - Suffragette City | base_T=0.4464113325140045 cur_T=0.7586708660369917 base_dev=0.04419643724598976 cur_dev=None
  - pos 17: David Bowie - Move On -> New Order - Sooner Than You Think | base_T=0.5323843106013314 cur_T=0.8302902189542899 base_dev=0.05692913719206438 cur_dev=0.28003484210896434
  - pos 18: The Beach Boys - Please Let Me Wonder -> No Age - Flutter Freer | base_T=0.6492639969941926 cur_T=0.9424741729013337 base_dev=0.0029714226868551297 cur_dev=0.11304310968254339
  - pos 19: The Beatles - Little Queenie -> Sonic Youth - Hot Wire My Heart | base_T=0.6120196198266036 cur_T=0.9353409635913639 base_dev=0.16769208289827908 cur_dev=0.048094857108766154
  - pos 20: Ariel Pink's Haunted Graffiti - Early Birds of Babylon -> Sonic Youth w. Lydia Lunch - Death Valley '69 | base_T=0.76452337442273 cur_T=0.9226276644427387 base_dev=0.11634548897856112 cur_dev=0.06893628574899446
  - pos 21: King Krule - Emergency Blimp -> The Jimi Hendrix Experience - Voodoo Child (Slight Return) | base_T=0.6910258897661363 cur_T=0.8745940561886648 base_dev=0.2734558711528715 cur_dev=0.14922095930894064
  - pos 22: The Fall - Room to Live -> The Stooges - T.V. Eye | base_T=0.7332345871241958 cur_T=0.8266032228365947 base_dev=0.3125249117150196 cur_dev=0.3303533781562691

- hash `0aedefbb12cd3185e36f5be713ccc1d93b62ea1ad73271fec0b60302c136704e` label `arc_w2.0_t0.0_abs_m0.15_autoTrue_tbNone_beamNone` overlap=None
  - pos 2: King Krule - The Cadet Leaps -> Godspeed You Black Emperor! - Moya | base_T=0.7159177022106813 cur_T=0.6827701884564017 base_dev=0.4891376911737034 cur_dev=0.3088111470468307
  - pos 5: Jónsi & Alex Somers - All The Big Trees (2019 Remaster) -> King Krule - The Cadet Leaps | base_T=0.6889105761394906 cur_T=0.6955376759430785 base_dev=0.004146027090951188 cur_dev=0.027197924918060112
  - pos 7: Tim Hecker - Abduction -> Foxes in Fiction - Insomnia Keys | base_T=0.4299009771685955 cur_T=0.7853552801920686 base_dev=0.25262841683403703 cur_dev=0.18482013225565375
  - pos 8: Mary Lattimore & Walt Mcclements - The Poppies, the Wild Mustard, the Blue-Eyed Grass -> Spacemen 3 - Things'll Never Be the Same | base_T=0.7602331050266803 cur_T=0.6712457107645613 base_dev=0.3468678703016146 cur_dev=0.31972795188539627
  - pos 10: Jonny Nash - October Song -> Labradford - V | base_T=0.7848286992525537 cur_T=0.6946432657992841 base_dev=0.1286568087201729 cur_dev=0.1678070190650204
  - pos 11: Labradford - V -> Jonny Nash - October Song | base_T=0.7699099714964436 cur_T=0.8502745376085842 base_dev=0.13082749692531817 cur_dev=0.09167728658047067
  - pos 14: Orchid Mantis - Fog -> Toro Y Moi - Pavement | base_T=0.42184005046144696 cur_T=0.5650068566729678 base_dev=0.0824767435334669 cur_dev=0.017554483794823605
  - pos 15: Wild Nothing - Ride -> Autechre - r cazt | base_T=0.5511360093665991 cur_T=0.7838957812645864 base_dev=0.06725810102744806 cur_dev=0.041108802196430705
  - pos 17: David Bowie - Move On -> Beak> - Windmill Hill | base_T=0.5323843106013314 cur_T=0.4282159590746211 base_dev=0.05692913719206438 cur_dev=0.01632303513205824
  - pos 18: The Beach Boys - Please Let Me Wonder -> David Bowie - Move On | base_T=0.6492639969941926 cur_T=0.5825412052017636 base_dev=0.0029714226868551297 cur_dev=0.04875226538855748
  - pos 19: The Beatles - Little Queenie -> The Beach Boys - Please Let Me Wonder | base_T=0.6120196198266036 cur_T=0.6492639969941926 base_dev=0.16769208289827908 cur_dev=0.09183160868373252
  - pos 20: Ariel Pink's Haunted Graffiti - Early Birds of Babylon -> The Clientele - What Goes Up | base_T=0.76452337442273 cur_T=0.6516270384407491 base_dev=0.11634548897856112 cur_dev=0.14635109947455416
  - pos 24: Talking Heads - Life During Wartime -> No Age - Flutter Freer | base_T=0.7815785734114493 cur_T=0.8874990957333945 base_dev=0.3128306198220131 cur_dev=0.25178264270438616
  - pos 25: Fleetwood Mac - Go Your Own Way -> Talking Heads - Life During Wartime | base_T=0.6328555843932391 cur_T=0.7718437194466157 base_dev=0.24536285286463566 cur_dev=0.17409108680017032
  - pos 26: Art Feynman - Early Signs of Rhythm -> Fleetwood Mac - Go Your Own Way | base_T=0.8753313515770027 cur_T=0.6328555843932391 base_dev=0.1426792760465157 cur_dev=0.04487841891342609

- hash `0aedefbb12cd3185e36f5be713ccc1d93b62ea1ad73271fec0b60302c136704e` label `arc_w2.0_t0.0_abs_m0.3_autoTrue_tbNone_beamNone` overlap=None
  - pos 2: King Krule - The Cadet Leaps -> Godspeed You Black Emperor! - Moya | base_T=0.7159177022106813 cur_T=0.6827701884564017 base_dev=0.4891376911737034 cur_dev=0.3088111470468307
  - pos 5: Jónsi & Alex Somers - All The Big Trees (2019 Remaster) -> King Krule - The Cadet Leaps | base_T=0.6889105761394906 cur_T=0.6955376759430785 base_dev=0.004146027090951188 cur_dev=0.027197924918060112
  - pos 7: Tim Hecker - Abduction -> Foxes in Fiction - Insomnia Keys | base_T=0.4299009771685955 cur_T=0.7853552801920686 base_dev=0.25262841683403703 cur_dev=0.18482013225565375
  - pos 8: Mary Lattimore & Walt Mcclements - The Poppies, the Wild Mustard, the Blue-Eyed Grass -> Spacemen 3 - Things'll Never Be the Same | base_T=0.7602331050266803 cur_T=0.6712457107645613 base_dev=0.3468678703016146 cur_dev=0.31972795188539627
  - pos 10: Jonny Nash - October Song -> Labradford - V | base_T=0.7848286992525537 cur_T=0.6946432657992841 base_dev=0.1286568087201729 cur_dev=0.1678070190650204
  - pos 11: Labradford - V -> Jonny Nash - October Song | base_T=0.7699099714964436 cur_T=0.8502745376085842 base_dev=0.13082749692531817 cur_dev=0.09167728658047067
  - pos 14: Orchid Mantis - Fog -> Toro Y Moi - Pavement | base_T=0.42184005046144696 cur_T=0.5650068566729678 base_dev=0.0824767435334669 cur_dev=0.017554483794823605
  - pos 15: Wild Nothing - Ride -> Autechre - r cazt | base_T=0.5511360093665991 cur_T=0.7838957812645864 base_dev=0.06725810102744806 cur_dev=0.041108802196430705
  - pos 17: David Bowie - Move On -> Beak> - Windmill Hill | base_T=0.5323843106013314 cur_T=0.4282159590746211 base_dev=0.05692913719206438 cur_dev=0.01632303513205824
  - pos 18: The Beach Boys - Please Let Me Wonder -> David Bowie - Move On | base_T=0.6492639969941926 cur_T=0.5825412052017636 base_dev=0.0029714226868551297 cur_dev=0.04875226538855748
  - pos 19: The Beatles - Little Queenie -> The Beach Boys - Please Let Me Wonder | base_T=0.6120196198266036 cur_T=0.6492639969941926 base_dev=0.16769208289827908 cur_dev=0.09183160868373252
  - pos 20: Ariel Pink's Haunted Graffiti - Early Birds of Babylon -> The Clientele - What Goes Up | base_T=0.76452337442273 cur_T=0.6516270384407491 base_dev=0.11634548897856112 cur_dev=0.14635109947455416
  - pos 24: Talking Heads - Life During Wartime -> No Age - Flutter Freer | base_T=0.7815785734114493 cur_T=0.8874990957333945 base_dev=0.3128306198220131 cur_dev=0.25178264270438616
  - pos 25: Fleetwood Mac - Go Your Own Way -> Talking Heads - Life During Wartime | base_T=0.6328555843932391 cur_T=0.7718437194466157 base_dev=0.24536285286463566 cur_dev=0.17409108680017032
  - pos 26: Art Feynman - Early Signs of Rhythm -> Fleetwood Mac - Go Your Own Way | base_T=0.8753313515770027 cur_T=0.6328555843932391 base_dev=0.1426792760465157 cur_dev=0.04487841891342609

- hash `789e86b9fdaaeb382901d9c84a01c8067ea4fc1629987dccdd39a2c7e818d29d` label `arc_w2.0_t0.0_abs_m0.4_autoTrue_tb0.02_beam50` overlap=None
  - pos 2: King Krule - The Cadet Leaps -> Godspeed You Black Emperor! - Moya | base_T=0.7159177022106813 cur_T=0.6827701884564017 base_dev=0.4891376911737034 cur_dev=0.3088111470468307
  - pos 3: Mount Eerie - (soft air) -> Ezra Feinberg - Ovation | base_T=0.6195541647965571 cur_T=0.7039687690298055 base_dev=0.3329677950033614 cur_dev=0.17660515047671138
  - pos 4: Godspeed You! Black Emperor - Peasantry or ‘Light! Inside of Light!’ -> Dressed Up Animals - Mondtanz | base_T=0.6135803414188412 cur_T=0.8258495046404635 base_dev=0.17513777839063632 cur_dev=0.09885559488118478
  - pos 5: Jónsi & Alex Somers - All The Big Trees (2019 Remaster) -> King Krule - The Cadet Leaps | base_T=0.6889105761394906 cur_T=0.8067052528326448 base_dev=0.004146027090951188 cur_dev=0.027197924918060112
  - pos 6: Sigur Rós - Andrá -> Ólafur Arnalds - The Bottom Line | base_T=0.6484424356689317 cur_T=0.8535193140570737 base_dev=0.07191639938006988 cur_dev=0.04651712824524801
  - pos 7: Tim Hecker - Abduction -> Sigur Rós - Andrá | base_T=0.4299009771685955 cur_T=0.8192691191030009 base_dev=0.25262841683403703 cur_dev=0.2341280737907988
  - pos 8: Mary Lattimore & Walt Mcclements - The Poppies, the Wild Mustard, the Blue-Eyed Grass -> M83 - Another Wave from You | base_T=0.7602331050266803 cur_T=0.78742005345353 base_dev=0.3468678703016146 cur_dev=0.31354826082438414
  - pos 10: Jonny Nash - October Song -> Cornelius - Surfing on Mind Wave, Pt. 2 | base_T=0.7848286992525537 cur_T=0.8034951146243114 base_dev=0.1286568087201729 cur_dev=0.2072080595593752
  - pos 11: Labradford - V -> Microphones - Here With Summer | base_T=0.7699099714964436 cur_T=0.8398301230820251 base_dev=0.13082749692531817 cur_dev=0.0800313041791113
  - pos 12: TV On The Radio - Tonight -> PaPerCuts - Mattress on the Floor | base_T=0.6403422925030239 cur_T=0.8967572984165522 base_dev=0.059283005278525897 cur_dev=0.05568463355117209
  - pos 13: Animal Collective - No More Runnin -> Dirty Beaches - Sweet 17 | base_T=0.6221370689484218 cur_T=0.8756926284369257 base_dev=0.0642896440993998 cur_dev=0.02039670325270082
  - pos 14: Orchid Mantis - Fog -> Television Personalities - David Hockney's Diary | base_T=0.42184005046144696 cur_T=0.9142592763980442 base_dev=0.0824767435334669 cur_dev=0.17267414574460205
  - pos 15: Wild Nothing - Ride -> Ramones - Glad to See You Go (40th Anniversary Mix) | base_T=0.5511360093665991 cur_T=0.829455849593771 base_dev=0.06725810102744806 cur_dev=0.3090328786583877
  - pos 16: 00110100 01010100 - 0000 871 0017 -> David Bowie - Suffragette City | base_T=0.4464113325140045 cur_T=0.7586708660369917 base_dev=0.04419643724598976 cur_dev=None
  - pos 17: David Bowie - Move On -> New Order - Sooner Than You Think | base_T=0.5323843106013314 cur_T=0.8302902189542899 base_dev=0.05692913719206438 cur_dev=0.28003484210896434
  - pos 18: The Beach Boys - Please Let Me Wonder -> No Age - Flutter Freer | base_T=0.6492639969941926 cur_T=0.9424741729013337 base_dev=0.0029714226868551297 cur_dev=0.11304310968254339
  - pos 19: The Beatles - Little Queenie -> Sonic Youth - Hot Wire My Heart | base_T=0.6120196198266036 cur_T=0.9353409635913639 base_dev=0.16769208289827908 cur_dev=0.048094857108766154
  - pos 20: Ariel Pink's Haunted Graffiti - Early Birds of Babylon -> Sonic Youth w. Lydia Lunch - Death Valley '69 | base_T=0.76452337442273 cur_T=0.9226276644427387 base_dev=0.11634548897856112 cur_dev=0.06893628574899446
  - pos 21: King Krule - Emergency Blimp -> The Jimi Hendrix Experience - Voodoo Child (Slight Return) | base_T=0.6910258897661363 cur_T=0.8745940561886648 base_dev=0.2734558711528715 cur_dev=0.14922095930894064
  - pos 22: The Fall - Room to Live -> The Stooges - T.V. Eye | base_T=0.7332345871241958 cur_T=0.8266032228365947 base_dev=0.3125249117150196 cur_dev=0.3303533781562691

- hash `32f1333d0c0552e196c82266ed91950c7d4430bed0a7e8faa7fc157b257f092c` label `arc_w2.0_t0.2_abs_m0.1_autoTrue_tbNone_beam25` overlap=None
  - pos 2: King Krule - The Cadet Leaps -> Ezra Feinberg - Ovation | base_T=0.7159177022106813 cur_T=0.6519174905794566 base_dev=0.4891376911737034 cur_dev=0.28499152613908096
  - pos 3: Mount Eerie - (soft air) -> Dressed Up Animals - Mondtanz | base_T=0.6195541647965571 cur_T=0.8258495046404635 base_dev=0.3329677950033614 cur_dev=0.26106726929191365
  - pos 4: Godspeed You! Black Emperor - Peasantry or ‘Light! Inside of Light!’ -> King Krule - The Cadet Leaps | base_T=0.6135803414188412 cur_T=0.8067052528326448 base_dev=0.17513777839063632 cur_dev=0.21853964110060498
  - pos 5: Jónsi & Alex Somers - All The Big Trees (2019 Remaster) -> Alex Somers - Looking After | base_T=0.6889105761394906 cur_T=0.8694433726555622 base_dev=0.004146027090951188 cur_dev=0.08732359452242228
  - pos 6: Sigur Rós - Andrá -> Tim Hecker - Abduction | base_T=0.6484424356689317 cur_T=0.8464609468686048 base_dev=0.07191639938006988 cur_dev=0.0904167424233081
  - pos 7: Tim Hecker - Abduction -> Sigur Rós - Andrá | base_T=0.4299009771685955 cur_T=0.7452949556047597 base_dev=0.25262841683403703 cur_dev=0.2341280737907988
  - pos 8: Mary Lattimore & Walt Mcclements - The Poppies, the Wild Mustard, the Blue-Eyed Grass -> M83 - Another Wave from You | base_T=0.7602331050266803 cur_T=0.78742005345353 base_dev=0.3468678703016146 cur_dev=0.31354826082438414
  - pos 10: Jonny Nash - October Song -> Cornelius - Surfing on Mind Wave, Pt. 2 | base_T=0.7848286992525537 cur_T=0.8034951146243114 base_dev=0.1286568087201729 cur_dev=0.2072080595593752
  - pos 11: Labradford - V -> Microphones - Here With Summer | base_T=0.7699099714964436 cur_T=0.8398301230820251 base_dev=0.13082749692531817 cur_dev=0.0800313041791113
  - pos 12: TV On The Radio - Tonight -> PaPerCuts - Mattress on the Floor | base_T=0.6403422925030239 cur_T=0.8967572984165522 base_dev=0.059283005278525897 cur_dev=0.05568463355117209
  - pos 13: Animal Collective - No More Runnin -> Dirty Beaches - Sweet 17 | base_T=0.6221370689484218 cur_T=0.8756926284369257 base_dev=0.0642896440993998 cur_dev=0.02039670325270082
  - pos 14: Orchid Mantis - Fog -> Television Personalities - David Hockney's Diary | base_T=0.42184005046144696 cur_T=0.9142592763980442 base_dev=0.0824767435334669 cur_dev=0.17267414574460205
  - pos 15: Wild Nothing - Ride -> Ramones - Glad to See You Go (40th Anniversary Mix) | base_T=0.5511360093665991 cur_T=0.829455849593771 base_dev=0.06725810102744806 cur_dev=0.3090328786583877
  - pos 16: 00110100 01010100 - 0000 871 0017 -> David Bowie - Suffragette City | base_T=0.4464113325140045 cur_T=0.7586708660369917 base_dev=0.04419643724598976 cur_dev=None
  - pos 17: David Bowie - Move On -> No Age - Flutter Freer | base_T=0.5323843106013314 cur_T=0.8874990957333945 base_dev=0.05692913719206438 cur_dev=0.25178264270438616
  - pos 18: The Beach Boys - Please Let Me Wonder -> Talking Heads - Life During Wartime | base_T=0.6492639969941926 cur_T=0.7718437194466157 base_dev=0.0029714226868551297 cur_dev=0.17409108680017032
  - pos 19: The Beatles - Little Queenie -> The Fall - No Xmas For John Key (Peel Session) | base_T=0.6120196198266036 cur_T=0.8430552344999207 base_dev=0.16769208289827908 cur_dev=0.09241799435977271
  - pos 20: Ariel Pink's Haunted Graffiti - Early Birds of Babylon -> Sonic Youth w. Lydia Lunch - Death Valley '69 | base_T=0.76452337442273 cur_T=0.8955575758212249 base_dev=0.11634548897856112 cur_dev=0.06893628574899446
  - pos 21: King Krule - Emergency Blimp -> The Jimi Hendrix Experience - Voodoo Child (Slight Return) | base_T=0.6910258897661363 cur_T=0.8745940561886648 base_dev=0.2734558711528715 cur_dev=0.14922095930894064
  - pos 22: The Fall - Room to Live -> The Stooges - T.V. Eye | base_T=0.7332345871241958 cur_T=0.8266032228365947 base_dev=0.3125249117150196 cur_dev=0.3303533781562691

- hash `c7f6e0efdb0ad6532cb6a46502d5208ca707e491170113bae38e888811eadf9c` label `arc_w2.0_t0.2_abs_m0.3_autoFalse_tbNone_beam50` overlap=None
  - pos 2: King Krule - The Cadet Leaps -> Ezra Feinberg - Ovation | base_T=0.7159177022106813 cur_T=0.6519174905794566 base_dev=0.4891376911737034 cur_dev=0.28499152613908096
  - pos 3: Mount Eerie - (soft air) -> Dressed Up Animals - Mondtanz | base_T=0.6195541647965571 cur_T=0.8258495046404635 base_dev=0.3329677950033614 cur_dev=0.26106726929191365
  - pos 4: Godspeed You! Black Emperor - Peasantry or ‘Light! Inside of Light!’ -> King Krule - The Cadet Leaps | base_T=0.6135803414188412 cur_T=0.8067052528326448 base_dev=0.17513777839063632 cur_dev=0.21853964110060498
  - pos 5: Jónsi & Alex Somers - All The Big Trees (2019 Remaster) -> Alex Somers - Looking After | base_T=0.6889105761394906 cur_T=0.8694433726555622 base_dev=0.004146027090951188 cur_dev=0.08732359452242228
  - pos 6: Sigur Rós - Andrá -> Tim Hecker - Abduction | base_T=0.6484424356689317 cur_T=0.8464609468686048 base_dev=0.07191639938006988 cur_dev=0.0904167424233081
  - pos 7: Tim Hecker - Abduction -> Sigur Rós - Andrá | base_T=0.4299009771685955 cur_T=0.7452949556047597 base_dev=0.25262841683403703 cur_dev=0.2341280737907988
  - pos 8: Mary Lattimore & Walt Mcclements - The Poppies, the Wild Mustard, the Blue-Eyed Grass -> M83 - Another Wave from You | base_T=0.7602331050266803 cur_T=0.78742005345353 base_dev=0.3468678703016146 cur_dev=0.31354826082438414
  - pos 10: Jonny Nash - October Song -> Cornelius - Surfing on Mind Wave, Pt. 2 | base_T=0.7848286992525537 cur_T=0.8034951146243114 base_dev=0.1286568087201729 cur_dev=0.2072080595593752
  - pos 11: Labradford - V -> Microphones - Here With Summer | base_T=0.7699099714964436 cur_T=0.8398301230820251 base_dev=0.13082749692531817 cur_dev=0.0800313041791113
  - pos 12: TV On The Radio - Tonight -> PaPerCuts - Mattress on the Floor | base_T=0.6403422925030239 cur_T=0.8967572984165522 base_dev=0.059283005278525897 cur_dev=0.05568463355117209
  - pos 13: Animal Collective - No More Runnin -> Dirty Beaches - Sweet 17 | base_T=0.6221370689484218 cur_T=0.8756926284369257 base_dev=0.0642896440993998 cur_dev=0.02039670325270082
  - pos 14: Orchid Mantis - Fog -> Television Personalities - David Hockney's Diary | base_T=0.42184005046144696 cur_T=0.9142592763980442 base_dev=0.0824767435334669 cur_dev=0.17267414574460205
  - pos 15: Wild Nothing - Ride -> Ramones - Glad to See You Go (40th Anniversary Mix) | base_T=0.5511360093665991 cur_T=0.829455849593771 base_dev=0.06725810102744806 cur_dev=0.3090328786583877
  - pos 16: 00110100 01010100 - 0000 871 0017 -> David Bowie - Suffragette City | base_T=0.4464113325140045 cur_T=0.7586708660369917 base_dev=0.04419643724598976 cur_dev=None
  - pos 17: David Bowie - Move On -> No Age - Flutter Freer | base_T=0.5323843106013314 cur_T=0.8874990957333945 base_dev=0.05692913719206438 cur_dev=0.25178264270438616
  - pos 18: The Beach Boys - Please Let Me Wonder -> The B-52's - Party Out Of Bounds | base_T=0.6492639969941926 cur_T=0.8997810645466968 base_dev=0.0029714226868551297 cur_dev=0.2140457257301595
  - pos 19: The Beatles - Little Queenie -> Art Feynman - Early Signs of Rhythm | base_T=0.6120196198266036 cur_T=0.9182195260972482 base_dev=0.16769208289827908 cur_dev=0.1426792760465157
  - pos 20: Ariel Pink's Haunted Graffiti - Early Birds of Babylon -> Sonic Youth w. Lydia Lunch - Death Valley '69 | base_T=0.76452337442273 cur_T=0.8197566531001381 base_dev=0.11634548897856112 cur_dev=0.06893628574899446
  - pos 21: King Krule - Emergency Blimp -> The Jimi Hendrix Experience - Voodoo Child (Slight Return) | base_T=0.6910258897661363 cur_T=0.8745940561886648 base_dev=0.2734558711528715 cur_dev=0.14922095930894064
  - pos 22: The Fall - Room to Live -> The Stooges - T.V. Eye | base_T=0.7332345871241958 cur_T=0.8266032228365947 base_dev=0.3125249117150196 cur_dev=0.3303533781562691

### seed 486803832f3fc21eb7fa4512a0283554
- baseline hash: `58cfb01b5d1d15aa9912a07cc924e30744571642859a7c8b0bf030259c257532`

- hash `c40867d2d3581c5d75c4ac8ebc7c926711b1be8160755b07e6c20f130af917b2` label `arc_w0.75_t0.0_abs_m0.1_autoFalse_tb0.02_beam50` overlap=None
  - pos 1: David Bowie - Suffragette City -> David Bowie - Life On Mars? | base_T=None cur_T=None base_dev=None cur_dev=None
  - pos 2: Fleetwood Mac - Go Your Own Way -> St. Vincent - The Bed | base_T=0.7261979944443382 cur_T=0.8682706332347809 base_dev=0.39555771819091223 cur_dev=0.48980799972154376
  - pos 3: Art Feynman - Early Signs of Rhythm -> King Krule - The Cadet Leaps | base_T=0.8753313515770027 cur_T=0.8130570399659223 base_dev=0.38497219966163226 cur_dev=0.38075131551133384
  - pos 4: Sonic Youth w. Lydia Lunch - Death Valley '69 -> Autechre - Altichyre | base_T=0.8197566531001381 cur_T=0.7761435841139289 base_dev=0.23366589741170762 cur_dev=0.22187967119634222
  - pos 5: The Jimi Hendrix Experience - Voodoo Child (Slight Return) -> Shabason, Krgovich, Sage - Joe | base_T=0.8745940561886648 cur_T=0.8471807111701921 base_dev=0.1625239416204261 cur_dev=0.016499777079848565
  - pos 6: The Stooges - T.V. Eye -> Lambchop - Flower | base_T=0.8266032228365947 cur_T=0.8532018867324963 base_dev=0.07121066038760437 cur_dev=0.07456150619847413
  - pos 7: David Bowie - Cracked Actor -> Sigur Rós - Andrá | base_T=0.8824453880191951 cur_T=0.7412178047301338 base_dev=0.25962938178114026 cur_dev=0.2341280737907988
  - pos 8: Prince - Baby I'm a Star ("Take Me with U" 7" B-Side Edit) (Take Me with U 7" B-Side Edit; 2017 Remaster) -> M83 - Another Wave from You | base_T=0.8793224970039951 cur_T=0.78742005345353 base_dev=0.3108689861654296 cur_dev=0.31354826082438414
  - pos 9: David Bowie - Let's Dance -> David Bowie - Warszawa | base_T=0.7845443668679988 cur_T=0.8865096925932843 base_dev=None cur_dev=None
  - pos 10: Fleetwood Mac - Blue Letter -> Cornelius - Surfing on Mind Wave, Pt. 2 | base_T=0.5734668558132143 cur_T=0.8034951146243114 base_dev=0.18860675779001007 cur_dev=0.2072080595593752
  - pos 11: Unknown Mortal Orchestra - Swim and Sleep (Like a Shark) -> Microphones - Here With Summer | base_T=0.7524002636145654 cur_T=0.8398301230820251 base_dev=0.25883326316655203 cur_dev=0.0800313041791113
  - pos 12: Prince - Take Me with U (2015 Paisley Park Remaster) -> PaPerCuts - Mattress on the Floor | base_T=0.5663842533065095 cur_T=0.8967572984165522 base_dev=0.0025446620136969217 cur_dev=0.05568463355117209
  - pos 13: The Supremes - Ask Any Girl -> Dirty Beaches - Sweet 17 | base_T=0.5122538172333997 cur_T=0.8756926284369257 base_dev=0.002159490343563275 cur_dev=0.02039670325270082
  - pos 14: Stevie Wonder - Never Had A Dream Come True -> Television Personalities - David Hockney's Diary | base_T=0.832064059755911 cur_T=0.9142592763980442 base_dev=0.005997006971160401 cur_dev=0.17267414574460205
  - pos 15: James Brown - Santa Claus, Santa Claus -> Ramones - Glad to See You Go (40th Anniversary Mix) | base_T=0.5153161003897999 cur_T=0.829455849593771 base_dev=0.0044046067747152695 cur_dev=0.3090328786583877
  - pos 16: Marvin Gaye - I Wanna Be Where You Are -> David Bowie - Suffragette City | base_T=0.7758543087250149 cur_T=0.7586708660369917 base_dev=0.1458928281625957 cur_dev=None
  - pos 17: Video Age - Blushing -> The Fall - Perverted by Languagelive at Electric Ballroom, London 8 December 1983 | base_T=0.44735900435803805 cur_T=0.8926473957940299 base_dev=0.063808524900418 cur_dev=0.2692867594119475
  - pos 18: Deerhoof - Fraction Anthem -> Art Feynman - Early Signs of Rhythm | base_T=0.4748340800811629 cur_T=0.8374564949764935 base_dev=0.0695872440346178 cur_dev=0.48190324301956805
  - pos 19: Porches - Glow -> Talking Heads - Cities | base_T=0.7223826143454176 cur_T=0.8609959869184659 base_dev=0.18416985195786595 cur_dev=0.3584849210937879
  - pos 20: David Bowie - Changes -> The Beach Boys - Our Car Club (Mono) | base_T=0.8127477255057459 cur_T=0.8627276767248913 base_dev=0.19226278364176386 cur_dev=0.21503277677762772

- hash `f92bae6baa7b44b88b534c4f3b6ff28117aa897666988fb0054f482cd722513d` label `arc_w1.5_t0.05_abs_m0.2_autoFalse_tb0.02_beamNone` overlap=None
  - pos 2: Fleetwood Mac - Go Your Own Way -> No Age - Flutter Freer | base_T=0.7261979944443382 cur_T=0.8874990957333945 base_dev=0.39555771819091223 cur_dev=0.26323797500881996
  - pos 3: Art Feynman - Early Signs of Rhythm -> Talking Heads - Life During Wartime | base_T=0.8753313515770027 cur_T=0.7718437194466157 base_dev=0.38497219966163226 cur_dev=0.21589957646407731
  - pos 4: Sonic Youth w. Lydia Lunch - Death Valley '69 -> Fleetwood Mac - Go Your Own Way | base_T=0.8197566531001381 cur_T=0.6328555843932391 base_dev=0.23366589741170762 cur_dev=0.12495966811781378
  - pos 5: The Jimi Hendrix Experience - Voodoo Child (Slight Return) -> Art Feynman - Early Signs of Rhythm | base_T=0.8745940561886648 cur_T=0.8753313515770027 base_dev=0.1625239416204261 cur_dev=0.03141880906835853
  - pos 6: The Stooges - T.V. Eye -> Prince - Baby I'm a Star ("Take Me with U" 7" B-Side Edit) (Take Me with U 7" B-Side Edit; 2017 Remaster) | base_T=0.8266032228365947 cur_T=0.8888348523258907 base_dev=0.07121066038760437 cur_dev=0.04027093609233101
  - pos 7: David Bowie - Cracked Actor -> The Jimi Hendrix Experience - Voodoo Child (Slight Return) | base_T=0.8824453880191951 cur_T=0.5314857512094759 base_dev=0.25962938178114026 cur_dev=0.19102944897284768
  - pos 8: Prince - Baby I'm a Star ("Take Me with U" 7" B-Side Edit) (Take Me with U 7" B-Side Edit; 2017 Remaster) -> The Stooges - T.V. Eye | base_T=0.8793224970039951 cur_T=0.8266032228365947 base_dev=0.3108689861654296 cur_dev=0.34180871046070294
  - pos 10: Fleetwood Mac - Blue Letter -> Talking Heads - Mind | base_T=0.5734668558132143 cur_T=0.8634542784698765 base_dev=0.18860675779001007 cur_dev=0.1089753127366695
  - pos 11: Unknown Mortal Orchestra - Swim and Sleep (Like a Shark) -> The Beatles - Yellow Submarine (2022 Mix) | base_T=0.7524002636145654 cur_T=0.7964319722259314 base_dev=0.25883326316655203 cur_dev=0.179700202007851
  - pos 13: The Supremes - Ask Any Girl -> Fleetwood Mac - Blue Letter | base_T=0.5122538172333997 cur_T=0.6820691965383506 base_dev=0.002159490343563275 cur_dev=0.012887702628465059
  - pos 14: Stevie Wonder - Never Had A Dream Come True -> Black Moth Super Rainbow - Last House In the Enchanted Forest | base_T=0.832064059755911 cur_T=0.8747167831653818 base_dev=0.005997006971160401 cur_dev=0.0063320178530172355
  - pos 15: James Brown - Santa Claus, Santa Claus -> Hyldon - Cor De Maçã | base_T=0.5153161003897999 cur_T=0.599202548108818 base_dev=0.0044046067747152695 cur_dev=0.07287166623764857
  - pos 17: Video Age - Blushing -> Norah Jones - Don't Know Why | base_T=0.44735900435803805 cur_T=0.6177673307821179 base_dev=0.063808524900418 cur_dev=0.16778030878295835
  - pos 18: Deerhoof - Fraction Anthem -> The Beach Boys - I Know There's An Answer | base_T=0.4748340800811629 cur_T=0.6245758584405523 base_dev=0.0695872440346178 cur_dev=0.11558086651025046
  - pos 19: Porches - Glow -> Deerhoof - Fraction Anthem | base_T=0.7223826143454176 cur_T=0.3382076170986094 base_dev=0.18416985195786595 cur_dev=0.16439027540520545
  - pos 20: David Bowie - Changes -> Porches - Glow | base_T=0.8127477255057459 cur_T=0.7223826143454176 base_dev=0.19226278364176386 cur_dev=0.2633406922625141
  - pos 24: Godspeed You! Black Emperor - Peasantry or ‘Light! Inside of Light!’ -> Moses Sumney - Gagarin | base_T=0.7247934731539941 cur_T=0.6210902344886864 base_dev=0.434280496159301 cur_dev=0.33604727834881687
  - pos 25: Mount Eerie - Night Palace -> Stevie Wonder - I Beleive (When I Fall In Love It Will Be Forever) | base_T=0.6996319683550355 cur_T=0.6535986334562918 base_dev=0.3812978609756771 cur_dev=0.28799166293583245

- hash `24d3269545bb25e3507a1042d7b6fd2eaee880fff610b4168b3391cd84c78c65` label `arc_w1.5_t0.08_abs_m0.15_autoFalse_tb0.02_beam10` overlap=None
  - pos 1: David Bowie - Suffragette City -> David Bowie - Life On Mars? | base_T=None cur_T=None base_dev=None cur_dev=None
  - pos 2: Fleetwood Mac - Go Your Own Way -> St. Vincent - The Bed | base_T=0.7261979944443382 cur_T=0.8682706332347809 base_dev=0.39555771819091223 cur_dev=0.48980799972154376
  - pos 3: Art Feynman - Early Signs of Rhythm -> King Krule - The Cadet Leaps | base_T=0.8753313515770027 cur_T=0.8130570399659223 base_dev=0.38497219966163226 cur_dev=0.38075131551133384
  - pos 4: Sonic Youth w. Lydia Lunch - Death Valley '69 -> Autechre - Altichyre | base_T=0.8197566531001381 cur_T=0.7761435841139289 base_dev=0.23366589741170762 cur_dev=0.22187967119634222
  - pos 5: The Jimi Hendrix Experience - Voodoo Child (Slight Return) -> Shabason, Krgovich, Sage - Joe | base_T=0.8745940561886648 cur_T=0.8471807111701921 base_dev=0.1625239416204261 cur_dev=0.016499777079848565
  - pos 6: The Stooges - T.V. Eye -> Sigur Rós - Andrá | base_T=0.8266032228365947 cur_T=0.7630046383535722 base_dev=0.07121066038760437 cur_dev=0.07191639938006988
  - pos 7: David Bowie - Cracked Actor -> Foxes in Fiction - Insomnia Keys | base_T=0.8824453880191951 cur_T=0.7853552801920686 base_dev=0.25962938178114026 cur_dev=0.18482013225565375
  - pos 8: Prince - Baby I'm a Star ("Take Me with U" 7" B-Side Edit) (Take Me with U 7" B-Side Edit; 2017 Remaster) -> M83 - Another Wave from You | base_T=0.8793224970039951 cur_T=0.6604160308918139 base_dev=0.3108689861654296 cur_dev=0.31354826082438414
  - pos 9: David Bowie - Let's Dance -> David Bowie - Warszawa | base_T=0.7845443668679988 cur_T=0.8865096925932843 base_dev=None cur_dev=None
  - pos 10: Fleetwood Mac - Blue Letter -> Cornelius - Surfing on Mind Wave, Pt. 2 | base_T=0.5734668558132143 cur_T=0.8034951146243114 base_dev=0.18860675779001007 cur_dev=0.2072080595593752
  - pos 11: Unknown Mortal Orchestra - Swim and Sleep (Like a Shark) -> Microphones - Here With Summer | base_T=0.7524002636145654 cur_T=0.8398301230820251 base_dev=0.25883326316655203 cur_dev=0.0800313041791113
  - pos 12: Prince - Take Me with U (2015 Paisley Park Remaster) -> PaPerCuts - Mattress on the Floor | base_T=0.5663842533065095 cur_T=0.8967572984165522 base_dev=0.0025446620136969217 cur_dev=0.05568463355117209
  - pos 13: The Supremes - Ask Any Girl -> Dirty Beaches - Sweet 17 | base_T=0.5122538172333997 cur_T=0.8756926284369257 base_dev=0.002159490343563275 cur_dev=0.02039670325270082
  - pos 14: Stevie Wonder - Never Had A Dream Come True -> Television Personalities - David Hockney's Diary | base_T=0.832064059755911 cur_T=0.9142592763980442 base_dev=0.005997006971160401 cur_dev=0.17267414574460205
  - pos 15: James Brown - Santa Claus, Santa Claus -> Ramones - Glad to See You Go (40th Anniversary Mix) | base_T=0.5153161003897999 cur_T=0.829455849593771 base_dev=0.0044046067747152695 cur_dev=0.3090328786583877
  - pos 16: Marvin Gaye - I Wanna Be Where You Are -> David Bowie - Suffragette City | base_T=0.7758543087250149 cur_T=0.7586708660369917 base_dev=0.1458928281625957 cur_dev=None
  - pos 17: Video Age - Blushing -> The Fall - Perverted by Languagelive at Electric Ballroom, London 8 December 1983 | base_T=0.44735900435803805 cur_T=0.8926473957940299 base_dev=0.063808524900418 cur_dev=0.2692867594119475
  - pos 18: Deerhoof - Fraction Anthem -> Art Feynman - Early Signs of Rhythm | base_T=0.4748340800811629 cur_T=0.8374564949764935 base_dev=0.0695872440346178 cur_dev=0.48190324301956805
  - pos 19: Porches - Glow -> Talking Heads - Cities | base_T=0.7223826143454176 cur_T=0.8609959869184659 base_dev=0.18416985195786595 cur_dev=0.3584849210937879
  - pos 20: David Bowie - Changes -> The Beach Boys - Our Car Club (Mono) | base_T=0.8127477255057459 cur_T=0.8627276767248913 base_dev=0.19226278364176386 cur_dev=0.21503277677762772

- hash `24d3269545bb25e3507a1042d7b6fd2eaee880fff610b4168b3391cd84c78c65` label `arc_w1.5_t0.08_abs_m0.15_autoTrue_tb0.02_beam10` overlap=None
  - pos 1: David Bowie - Suffragette City -> David Bowie - Life On Mars? | base_T=None cur_T=None base_dev=None cur_dev=None
  - pos 2: Fleetwood Mac - Go Your Own Way -> St. Vincent - The Bed | base_T=0.7261979944443382 cur_T=0.8682706332347809 base_dev=0.39555771819091223 cur_dev=0.48980799972154376
  - pos 3: Art Feynman - Early Signs of Rhythm -> King Krule - The Cadet Leaps | base_T=0.8753313515770027 cur_T=0.8130570399659223 base_dev=0.38497219966163226 cur_dev=0.38075131551133384
  - pos 4: Sonic Youth w. Lydia Lunch - Death Valley '69 -> Autechre - Altichyre | base_T=0.8197566531001381 cur_T=0.7761435841139289 base_dev=0.23366589741170762 cur_dev=0.22187967119634222
  - pos 5: The Jimi Hendrix Experience - Voodoo Child (Slight Return) -> Shabason, Krgovich, Sage - Joe | base_T=0.8745940561886648 cur_T=0.8471807111701921 base_dev=0.1625239416204261 cur_dev=0.016499777079848565
  - pos 6: The Stooges - T.V. Eye -> Sigur Rós - Andrá | base_T=0.8266032228365947 cur_T=0.7630046383535722 base_dev=0.07121066038760437 cur_dev=0.07191639938006988
  - pos 7: David Bowie - Cracked Actor -> Foxes in Fiction - Insomnia Keys | base_T=0.8824453880191951 cur_T=0.7853552801920686 base_dev=0.25962938178114026 cur_dev=0.18482013225565375
  - pos 8: Prince - Baby I'm a Star ("Take Me with U" 7" B-Side Edit) (Take Me with U 7" B-Side Edit; 2017 Remaster) -> M83 - Another Wave from You | base_T=0.8793224970039951 cur_T=0.6604160308918139 base_dev=0.3108689861654296 cur_dev=0.31354826082438414
  - pos 9: David Bowie - Let's Dance -> David Bowie - Warszawa | base_T=0.7845443668679988 cur_T=0.8865096925932843 base_dev=None cur_dev=None
  - pos 10: Fleetwood Mac - Blue Letter -> Cornelius - Surfing on Mind Wave, Pt. 2 | base_T=0.5734668558132143 cur_T=0.8034951146243114 base_dev=0.18860675779001007 cur_dev=0.2072080595593752
  - pos 11: Unknown Mortal Orchestra - Swim and Sleep (Like a Shark) -> Microphones - Here With Summer | base_T=0.7524002636145654 cur_T=0.8398301230820251 base_dev=0.25883326316655203 cur_dev=0.0800313041791113
  - pos 12: Prince - Take Me with U (2015 Paisley Park Remaster) -> PaPerCuts - Mattress on the Floor | base_T=0.5663842533065095 cur_T=0.8967572984165522 base_dev=0.0025446620136969217 cur_dev=0.05568463355117209
  - pos 13: The Supremes - Ask Any Girl -> Dirty Beaches - Sweet 17 | base_T=0.5122538172333997 cur_T=0.8756926284369257 base_dev=0.002159490343563275 cur_dev=0.02039670325270082
  - pos 14: Stevie Wonder - Never Had A Dream Come True -> Television Personalities - David Hockney's Diary | base_T=0.832064059755911 cur_T=0.9142592763980442 base_dev=0.005997006971160401 cur_dev=0.17267414574460205
  - pos 15: James Brown - Santa Claus, Santa Claus -> Ramones - Glad to See You Go (40th Anniversary Mix) | base_T=0.5153161003897999 cur_T=0.829455849593771 base_dev=0.0044046067747152695 cur_dev=0.3090328786583877
  - pos 16: Marvin Gaye - I Wanna Be Where You Are -> David Bowie - Suffragette City | base_T=0.7758543087250149 cur_T=0.7586708660369917 base_dev=0.1458928281625957 cur_dev=None
  - pos 17: Video Age - Blushing -> The Fall - Perverted by Languagelive at Electric Ballroom, London 8 December 1983 | base_T=0.44735900435803805 cur_T=0.8926473957940299 base_dev=0.063808524900418 cur_dev=0.2692867594119475
  - pos 18: Deerhoof - Fraction Anthem -> Art Feynman - Early Signs of Rhythm | base_T=0.4748340800811629 cur_T=0.8374564949764935 base_dev=0.0695872440346178 cur_dev=0.48190324301956805
  - pos 19: Porches - Glow -> Talking Heads - Cities | base_T=0.7223826143454176 cur_T=0.8609959869184659 base_dev=0.18416985195786595 cur_dev=0.3584849210937879
  - pos 20: David Bowie - Changes -> The Beach Boys - Our Car Club (Mono) | base_T=0.8127477255057459 cur_T=0.8627276767248913 base_dev=0.19226278364176386 cur_dev=0.21503277677762772

- hash `763a27731bc2f510d862e28eaa0f63a317a76dfa9839bf04345d86b5dd206683` label `arc_w1.5_t0.08_abs_m0.1_autoTrue_tbNone_beamNone` overlap=None
  - pos 2: Fleetwood Mac - Go Your Own Way -> No Age - Flutter Freer | base_T=0.7261979944443382 cur_T=0.8874990957333945 base_dev=0.39555771819091223 cur_dev=0.26323797500881996
  - pos 3: Art Feynman - Early Signs of Rhythm -> Talking Heads - Life During Wartime | base_T=0.8753313515770027 cur_T=0.7718437194466157 base_dev=0.38497219966163226 cur_dev=0.21589957646407731
  - pos 4: Sonic Youth w. Lydia Lunch - Death Valley '69 -> Fleetwood Mac - Go Your Own Way | base_T=0.8197566531001381 cur_T=0.6328555843932391 base_dev=0.23366589741170762 cur_dev=0.12495966811781378
  - pos 5: The Jimi Hendrix Experience - Voodoo Child (Slight Return) -> Art Feynman - Early Signs of Rhythm | base_T=0.8745940561886648 cur_T=0.8753313515770027 base_dev=0.1625239416204261 cur_dev=0.03141880906835853
  - pos 6: The Stooges - T.V. Eye -> David Bowie - Cracked Actor | base_T=0.8266032228365947 cur_T=0.7742549371410699 base_dev=0.07121066038760437 cur_dev=0.09741770737041133
  - pos 7: David Bowie - Cracked Actor -> Prince - Baby I'm a Star ("Take Me with U" 7" B-Side Edit) (Take Me with U 7" B-Side Edit; 2017 Remaster) | base_T=0.8824453880191951 cur_T=0.8793224970039951 base_dev=0.25962938178114026 cur_dev=0.20248261050305993
  - pos 8: Prince - Baby I'm a Star ("Take Me with U" 7" B-Side Edit) (Take Me with U 7" B-Side Edit; 2017 Remaster) -> The Jimi Hendrix Experience - Voodoo Child (Slight Return) | base_T=0.8793224970039951 cur_T=0.5314857512094759 base_dev=0.3108689861654296 cur_dev=0.2994158246352173
  - pos 10: Fleetwood Mac - Blue Letter -> Talking Heads - Mind | base_T=0.5734668558132143 cur_T=0.8634542784698765 base_dev=0.18860675779001007 cur_dev=0.1089753127366695
  - pos 11: Unknown Mortal Orchestra - Swim and Sleep (Like a Shark) -> The Beatles - Yellow Submarine (2022 Mix) | base_T=0.7524002636145654 cur_T=0.7964319722259314 base_dev=0.25883326316655203 cur_dev=0.179700202007851
  - pos 13: The Supremes - Ask Any Girl -> Fleetwood Mac - Blue Letter | base_T=0.5122538172333997 cur_T=0.6820691965383506 base_dev=0.002159490343563275 cur_dev=0.012887702628465059
  - pos 14: Stevie Wonder - Never Had A Dream Come True -> Black Moth Super Rainbow - Last House In the Enchanted Forest | base_T=0.832064059755911 cur_T=0.8747167831653818 base_dev=0.005997006971160401 cur_dev=0.0063320178530172355
  - pos 15: James Brown - Santa Claus, Santa Claus -> Hyldon - Cor De Maçã | base_T=0.5153161003897999 cur_T=0.599202548108818 base_dev=0.0044046067747152695 cur_dev=0.07287166623764857
  - pos 17: Video Age - Blushing -> Norah Jones - Don't Know Why | base_T=0.44735900435803805 cur_T=0.6177673307821179 base_dev=0.063808524900418 cur_dev=0.16778030878295835
  - pos 18: Deerhoof - Fraction Anthem -> The Beach Boys - I Know There's An Answer | base_T=0.4748340800811629 cur_T=0.6245758584405523 base_dev=0.0695872440346178 cur_dev=0.11558086651025046
  - pos 19: Porches - Glow -> Deerhoof - Fraction Anthem | base_T=0.7223826143454176 cur_T=0.3382076170986094 base_dev=0.18416985195786595 cur_dev=0.16439027540520545
  - pos 20: David Bowie - Changes -> Porches - Glow | base_T=0.8127477255057459 cur_T=0.7223826143454176 base_dev=0.19226278364176386 cur_dev=0.2633406922625141
  - pos 24: Godspeed You! Black Emperor - Peasantry or ‘Light! Inside of Light!’ -> Moses Sumney - Gagarin | base_T=0.7247934731539941 cur_T=0.6210902344886864 base_dev=0.434280496159301 cur_dev=0.33604727834881687
  - pos 25: Mount Eerie - Night Palace -> Stevie Wonder - I Beleive (When I Fall In Love It Will Be Forever) | base_T=0.6996319683550355 cur_T=0.6535986334562918 base_dev=0.3812978609756771 cur_dev=0.28799166293583245
  - pos 26: King Krule - The Cadet Leaps -> Godspeed You! Black Emperor - Peasantry or ‘Light! Inside of Light!’ | base_T=0.535524171086011 cur_T=0.46595641418519207 base_dev=0.13845839189621728 cur_dev=0.09505652918624863
  - pos 27: Sigur Rós - Andrá -> King Krule - The Cadet Leaps | base_T=0.8160552428817576 cur_T=0.6955376759430785 base_dev=0.008164849824317755 cur_dev=0.08406254206009711

- hash `f92bae6baa7b44b88b534c4f3b6ff28117aa897666988fb0054f482cd722513d` label `arc_w1.5_t0.0_abs_m0.2_autoFalse_tb0.02_beamNone` overlap=None
  - pos 2: Fleetwood Mac - Go Your Own Way -> No Age - Flutter Freer | base_T=0.7261979944443382 cur_T=0.8874990957333945 base_dev=0.39555771819091223 cur_dev=0.26323797500881996
  - pos 3: Art Feynman - Early Signs of Rhythm -> Talking Heads - Life During Wartime | base_T=0.8753313515770027 cur_T=0.7718437194466157 base_dev=0.38497219966163226 cur_dev=0.21589957646407731
  - pos 4: Sonic Youth w. Lydia Lunch - Death Valley '69 -> Fleetwood Mac - Go Your Own Way | base_T=0.8197566531001381 cur_T=0.6328555843932391 base_dev=0.23366589741170762 cur_dev=0.12495966811781378
  - pos 5: The Jimi Hendrix Experience - Voodoo Child (Slight Return) -> Art Feynman - Early Signs of Rhythm | base_T=0.8745940561886648 cur_T=0.8753313515770027 base_dev=0.1625239416204261 cur_dev=0.03141880906835853
  - pos 6: The Stooges - T.V. Eye -> Prince - Baby I'm a Star ("Take Me with U" 7" B-Side Edit) (Take Me with U 7" B-Side Edit; 2017 Remaster) | base_T=0.8266032228365947 cur_T=0.8888348523258907 base_dev=0.07121066038760437 cur_dev=0.04027093609233101
  - pos 7: David Bowie - Cracked Actor -> The Jimi Hendrix Experience - Voodoo Child (Slight Return) | base_T=0.8824453880191951 cur_T=0.5314857512094759 base_dev=0.25962938178114026 cur_dev=0.19102944897284768
  - pos 8: Prince - Baby I'm a Star ("Take Me with U" 7" B-Side Edit) (Take Me with U 7" B-Side Edit; 2017 Remaster) -> The Stooges - T.V. Eye | base_T=0.8793224970039951 cur_T=0.8266032228365947 base_dev=0.3108689861654296 cur_dev=0.34180871046070294
  - pos 10: Fleetwood Mac - Blue Letter -> Talking Heads - Mind | base_T=0.5734668558132143 cur_T=0.8634542784698765 base_dev=0.18860675779001007 cur_dev=0.1089753127366695
  - pos 11: Unknown Mortal Orchestra - Swim and Sleep (Like a Shark) -> The Beatles - Yellow Submarine (2022 Mix) | base_T=0.7524002636145654 cur_T=0.7964319722259314 base_dev=0.25883326316655203 cur_dev=0.179700202007851
  - pos 13: The Supremes - Ask Any Girl -> Fleetwood Mac - Blue Letter | base_T=0.5122538172333997 cur_T=0.6820691965383506 base_dev=0.002159490343563275 cur_dev=0.012887702628465059
  - pos 14: Stevie Wonder - Never Had A Dream Come True -> Black Moth Super Rainbow - Last House In the Enchanted Forest | base_T=0.832064059755911 cur_T=0.8747167831653818 base_dev=0.005997006971160401 cur_dev=0.0063320178530172355
  - pos 15: James Brown - Santa Claus, Santa Claus -> Hyldon - Cor De Maçã | base_T=0.5153161003897999 cur_T=0.599202548108818 base_dev=0.0044046067747152695 cur_dev=0.07287166623764857
  - pos 17: Video Age - Blushing -> Norah Jones - Don't Know Why | base_T=0.44735900435803805 cur_T=0.6177673307821179 base_dev=0.063808524900418 cur_dev=0.16778030878295835
  - pos 18: Deerhoof - Fraction Anthem -> The Beach Boys - I Know There's An Answer | base_T=0.4748340800811629 cur_T=0.6245758584405523 base_dev=0.0695872440346178 cur_dev=0.11558086651025046
  - pos 19: Porches - Glow -> Deerhoof - Fraction Anthem | base_T=0.7223826143454176 cur_T=0.3382076170986094 base_dev=0.18416985195786595 cur_dev=0.16439027540520545
  - pos 20: David Bowie - Changes -> Porches - Glow | base_T=0.8127477255057459 cur_T=0.7223826143454176 base_dev=0.19226278364176386 cur_dev=0.2633406922625141
  - pos 24: Godspeed You! Black Emperor - Peasantry or ‘Light! Inside of Light!’ -> Moses Sumney - Gagarin | base_T=0.7247934731539941 cur_T=0.6210902344886864 base_dev=0.434280496159301 cur_dev=0.33604727834881687
  - pos 25: Mount Eerie - Night Palace -> Stevie Wonder - I Beleive (When I Fall In Love It Will Be Forever) | base_T=0.6996319683550355 cur_T=0.6535986334562918 base_dev=0.3812978609756771 cur_dev=0.28799166293583245

- hash `0447de8d17b1b520ca044760c8aeb27003b80989868aebec2edbf3ee803ad704` label `arc_w1.5_t0.12_abs_m0.15_autoFalse_tb0.02_beam10` overlap=None
  - pos 1: David Bowie - Suffragette City -> David Bowie - Life On Mars? | base_T=None cur_T=None base_dev=None cur_dev=None
  - pos 2: Fleetwood Mac - Go Your Own Way -> teen suicide - My Little World | base_T=0.7261979944443382 cur_T=0.8061904896633129 base_dev=0.39555771819091223 cur_dev=0.3433774234597585
  - pos 3: Art Feynman - Early Signs of Rhythm -> Tom Waits - It's Over (Remastered) | base_T=0.8753313515770027 cur_T=0.7695236674405489 base_dev=0.38497219966163226 cur_dev=0.27705930792705674
  - pos 4: Sonic Youth w. Lydia Lunch - Death Valley '69 -> King Krule - The Cadet Leaps | base_T=0.8197566531001381 cur_T=0.7482779710978253 base_dev=0.23366589741170762 cur_dev=0.21853964110060498
  - pos 5: The Jimi Hendrix Experience - Voodoo Child (Slight Return) -> Sigur Rós - Andrá | base_T=0.8745940561886648 cur_T=0.8160552428817576 base_dev=0.1625239416204261 cur_dev=0.11942531680247498
  - pos 6: The Stooges - T.V. Eye -> Foxes in Fiction - Insomnia Keys | base_T=0.8266032228365947 cur_T=0.7853552801920686 base_dev=0.07121066038760437 cur_dev=0.022608457844924823
  - pos 7: David Bowie - Cracked Actor -> Spacemen 3 - Things'll Never Be the Same | base_T=0.8824453880191951 cur_T=0.6712457107645613 base_dev=0.25962938178114026 cur_dev=0.21134157622302663
  - pos 8: Prince - Baby I'm a Star ("Take Me with U" 7" B-Side Edit) (Take Me with U 7" B-Side Edit; 2017 Remaster) -> M83 - Another Wave from You | base_T=0.8793224970039951 cur_T=0.7505786214818555 base_dev=0.3108689861654296 cur_dev=0.31354826082438414
  - pos 9: David Bowie - Let's Dance -> David Bowie - Warszawa | base_T=0.7845443668679988 cur_T=0.8865096925932843 base_dev=None cur_dev=None
  - pos 10: Fleetwood Mac - Blue Letter -> Cornelius - Surfing on Mind Wave, Pt. 2 | base_T=0.5734668558132143 cur_T=0.8034951146243114 base_dev=0.18860675779001007 cur_dev=0.2072080595593752
  - pos 11: Unknown Mortal Orchestra - Swim and Sleep (Like a Shark) -> Microphones - Here With Summer | base_T=0.7524002636145654 cur_T=0.8398301230820251 base_dev=0.25883326316655203 cur_dev=0.0800313041791113
  - pos 12: Prince - Take Me with U (2015 Paisley Park Remaster) -> PaPerCuts - Mattress on the Floor | base_T=0.5663842533065095 cur_T=0.8967572984165522 base_dev=0.0025446620136969217 cur_dev=0.05568463355117209
  - pos 13: The Supremes - Ask Any Girl -> Dirty Beaches - Sweet 17 | base_T=0.5122538172333997 cur_T=0.8756926284369257 base_dev=0.002159490343563275 cur_dev=0.02039670325270082
  - pos 14: Stevie Wonder - Never Had A Dream Come True -> Television Personalities - David Hockney's Diary | base_T=0.832064059755911 cur_T=0.9142592763980442 base_dev=0.005997006971160401 cur_dev=0.17267414574460205
  - pos 15: James Brown - Santa Claus, Santa Claus -> Ramones - Glad to See You Go (40th Anniversary Mix) | base_T=0.5153161003897999 cur_T=0.829455849593771 base_dev=0.0044046067747152695 cur_dev=0.3090328786583877
  - pos 16: Marvin Gaye - I Wanna Be Where You Are -> David Bowie - Suffragette City | base_T=0.7758543087250149 cur_T=0.7586708660369917 base_dev=0.1458928281625957 cur_dev=None
  - pos 17: Video Age - Blushing -> The Fall - Perverted by Languagelive at Electric Ballroom, London 8 December 1983 | base_T=0.44735900435803805 cur_T=0.8926473957940299 base_dev=0.063808524900418 cur_dev=0.2692867594119475
  - pos 18: Deerhoof - Fraction Anthem -> Art Feynman - Early Signs of Rhythm | base_T=0.4748340800811629 cur_T=0.8374564949764935 base_dev=0.0695872440346178 cur_dev=0.48190324301956805
  - pos 19: Porches - Glow -> Talking Heads - Cities | base_T=0.7223826143454176 cur_T=0.8609959869184659 base_dev=0.18416985195786595 cur_dev=0.3584849210937879
  - pos 20: David Bowie - Changes -> The Beach Boys - Our Car Club (Mono) | base_T=0.8127477255057459 cur_T=0.8627276767248913 base_dev=0.19226278364176386 cur_dev=0.21503277677762772

- hash `a20a1f4dd022d87592be1870896f1dcfc8340f04fa105ce34a78b5b3adaca87f` label `arc_w1.5_t0.2_abs_m0.1_autoTrue_tbNone_beam25` overlap=None
  - pos 1: David Bowie - Suffragette City -> David Bowie - Life On Mars? | base_T=None cur_T=None base_dev=None cur_dev=None
  - pos 2: Fleetwood Mac - Go Your Own Way -> St. Vincent - The Bed | base_T=0.7261979944443382 cur_T=0.8682706332347809 base_dev=0.39555771819091223 cur_dev=0.48980799972154376
  - pos 3: Art Feynman - Early Signs of Rhythm -> King Krule - The Cadet Leaps | base_T=0.8753313515770027 cur_T=0.8130570399659223 base_dev=0.38497219966163226 cur_dev=0.38075131551133384
  - pos 4: Sonic Youth w. Lydia Lunch - Death Valley '69 -> Autechre - Altichyre | base_T=0.8197566531001381 cur_T=0.7761435841139289 base_dev=0.23366589741170762 cur_dev=0.22187967119634222
  - pos 5: The Jimi Hendrix Experience - Voodoo Child (Slight Return) -> Shabason, Krgovich, Sage - Joe | base_T=0.8745940561886648 cur_T=0.8471807111701921 base_dev=0.1625239416204261 cur_dev=0.016499777079848565
  - pos 6: The Stooges - T.V. Eye -> Lambchop - Flower | base_T=0.8266032228365947 cur_T=0.8532018867324963 base_dev=0.07121066038760437 cur_dev=0.07456150619847413
  - pos 7: David Bowie - Cracked Actor -> Sigur Rós - Andrá | base_T=0.8824453880191951 cur_T=0.7412178047301338 base_dev=0.25962938178114026 cur_dev=0.2341280737907988
  - pos 8: Prince - Baby I'm a Star ("Take Me with U" 7" B-Side Edit) (Take Me with U 7" B-Side Edit; 2017 Remaster) -> M83 - Another Wave from You | base_T=0.8793224970039951 cur_T=0.78742005345353 base_dev=0.3108689861654296 cur_dev=0.31354826082438414
  - pos 9: David Bowie - Let's Dance -> David Bowie - Warszawa | base_T=0.7845443668679988 cur_T=0.8865096925932843 base_dev=None cur_dev=None
  - pos 10: Fleetwood Mac - Blue Letter -> Microphones - Here With Summer | base_T=0.5734668558132143 cur_T=0.6552326095259688 base_dev=0.18860675779001007 cur_dev=0.21877083720095408
  - pos 11: Unknown Mortal Orchestra - Swim and Sleep (Like a Shark) -> PaPerCuts - Mattress on the Floor | base_T=0.7524002636145654 cur_T=0.8967572984165522 base_dev=0.25883326316655203 cur_dev=0.25616906750238166
  - pos 12: Prince - Take Me with U (2015 Paisley Park Remaster) -> No Age - A Ceiling Dreams of a Floor | base_T=0.5663842533065095 cur_T=0.9139005290702163 base_dev=0.0025446620136969217 cur_dev=0.10578747777961578
  - pos 13: The Supremes - Ask Any Girl -> Dirty Beaches - Sweet 17 | base_T=0.5122538172333997 cur_T=0.8957787584269118 base_dev=0.002159490343563275 cur_dev=0.02039670325270082
  - pos 14: Stevie Wonder - Never Had A Dream Come True -> Television Personalities - David Hockney's Diary | base_T=0.832064059755911 cur_T=0.9142592763980442 base_dev=0.005997006971160401 cur_dev=0.17267414574460205
  - pos 15: James Brown - Santa Claus, Santa Claus -> Ramones - Glad to See You Go (40th Anniversary Mix) | base_T=0.5153161003897999 cur_T=0.829455849593771 base_dev=0.0044046067747152695 cur_dev=0.3090328786583877
  - pos 16: Marvin Gaye - I Wanna Be Where You Are -> David Bowie - Suffragette City | base_T=0.7758543087250149 cur_T=0.7586708660369917 base_dev=0.1458928281625957 cur_dev=None
  - pos 17: Video Age - Blushing -> The Fall - Perverted by Languagelive at Electric Ballroom, London 8 December 1983 | base_T=0.44735900435803805 cur_T=0.8926473957940299 base_dev=0.063808524900418 cur_dev=0.2692867594119475
  - pos 18: Deerhoof - Fraction Anthem -> Art Feynman - Early Signs of Rhythm | base_T=0.4748340800811629 cur_T=0.8374564949764935 base_dev=0.0695872440346178 cur_dev=0.48190324301956805
  - pos 19: Porches - Glow -> Talking Heads - Cities | base_T=0.7223826143454176 cur_T=0.8609959869184659 base_dev=0.18416985195786595 cur_dev=0.3584849210937879
  - pos 20: David Bowie - Changes -> The Beach Boys - Our Car Club (Mono) | base_T=0.8127477255057459 cur_T=0.8627276767248913 base_dev=0.19226278364176386 cur_dev=0.21503277677762772

- hash `9b90a97cf9b35c67040ad2a9c16bc52bf45afb6ed5da7a7f9cd1325da3255cca` label `arc_w2.0_t0.05_abs_m0.1_autoFalse_tb0.02_beam50` overlap=None
  - pos 1: David Bowie - Suffragette City -> David Bowie - Life On Mars? | base_T=None cur_T=None base_dev=None cur_dev=None
  - pos 2: Fleetwood Mac - Go Your Own Way -> teen suicide - My Little World | base_T=0.7261979944443382 cur_T=0.8061904896633129 base_dev=0.39555771819091223 cur_dev=0.3433774234597585
  - pos 3: Art Feynman - Early Signs of Rhythm -> Tom Waits - It's Over (Remastered) | base_T=0.8753313515770027 cur_T=0.7695236674405489 base_dev=0.38497219966163226 cur_dev=0.27705930792705674
  - pos 4: Sonic Youth w. Lydia Lunch - Death Valley '69 -> St. Vincent - The Bed | base_T=0.8197566531001381 cur_T=0.7780357102243484 base_dev=0.23366589741170762 cur_dev=0.2192099496484453
  - pos 5: The Jimi Hendrix Experience - Voodoo Child (Slight Return) -> King Krule - The Cadet Leaps | base_T=0.8745940561886648 cur_T=0.8130570399659223 base_dev=0.1625239416204261 cur_dev=0.027197924918060112
  - pos 6: The Stooges - T.V. Eye -> Sigur Rós - Andrá | base_T=0.8266032228365947 cur_T=0.8160552428817576 base_dev=0.07121066038760437 cur_dev=0.07191639938006988
  - pos 7: David Bowie - Cracked Actor -> Foxes in Fiction - Insomnia Keys | base_T=0.8824453880191951 cur_T=0.7853552801920686 base_dev=0.25962938178114026 cur_dev=0.18482013225565375
  - pos 8: Prince - Baby I'm a Star ("Take Me with U" 7" B-Side Edit) (Take Me with U 7" B-Side Edit; 2017 Remaster) -> M83 - Another Wave from You | base_T=0.8793224970039951 cur_T=0.6604160308918139 base_dev=0.3108689861654296 cur_dev=0.31354826082438414
  - pos 9: David Bowie - Let's Dance -> David Bowie - Warszawa | base_T=0.7845443668679988 cur_T=0.8865096925932843 base_dev=None cur_dev=None
  - pos 10: Fleetwood Mac - Blue Letter -> Cornelius - Surfing on Mind Wave, Pt. 2 | base_T=0.5734668558132143 cur_T=0.8034951146243114 base_dev=0.18860675779001007 cur_dev=0.2072080595593752
  - pos 11: Unknown Mortal Orchestra - Swim and Sleep (Like a Shark) -> Microphones - Here With Summer | base_T=0.7524002636145654 cur_T=0.8398301230820251 base_dev=0.25883326316655203 cur_dev=0.0800313041791113
  - pos 12: Prince - Take Me with U (2015 Paisley Park Remaster) -> PaPerCuts - Mattress on the Floor | base_T=0.5663842533065095 cur_T=0.8967572984165522 base_dev=0.0025446620136969217 cur_dev=0.05568463355117209
  - pos 13: The Supremes - Ask Any Girl -> Dirty Beaches - Sweet 17 | base_T=0.5122538172333997 cur_T=0.8756926284369257 base_dev=0.002159490343563275 cur_dev=0.02039670325270082
  - pos 14: Stevie Wonder - Never Had A Dream Come True -> Television Personalities - David Hockney's Diary | base_T=0.832064059755911 cur_T=0.9142592763980442 base_dev=0.005997006971160401 cur_dev=0.17267414574460205
  - pos 15: James Brown - Santa Claus, Santa Claus -> Ramones - Glad to See You Go (40th Anniversary Mix) | base_T=0.5153161003897999 cur_T=0.829455849593771 base_dev=0.0044046067747152695 cur_dev=0.3090328786583877
  - pos 16: Marvin Gaye - I Wanna Be Where You Are -> David Bowie - Suffragette City | base_T=0.7758543087250149 cur_T=0.7586708660369917 base_dev=0.1458928281625957 cur_dev=None
  - pos 17: Video Age - Blushing -> The Fall - Perverted by Languagelive at Electric Ballroom, London 8 December 1983 | base_T=0.44735900435803805 cur_T=0.8926473957940299 base_dev=0.063808524900418 cur_dev=0.2692867594119475
  - pos 18: Deerhoof - Fraction Anthem -> Art Feynman - Early Signs of Rhythm | base_T=0.4748340800811629 cur_T=0.8374564949764935 base_dev=0.0695872440346178 cur_dev=0.48190324301956805
  - pos 19: Porches - Glow -> Talking Heads - Cities | base_T=0.7223826143454176 cur_T=0.8609959869184659 base_dev=0.18416985195786595 cur_dev=0.3584849210937879
  - pos 20: David Bowie - Changes -> The Beach Boys - Our Car Club (Mono) | base_T=0.8127477255057459 cur_T=0.8627276767248913 base_dev=0.19226278364176386 cur_dev=0.21503277677762772

- hash `d80e2ff8571061b9e567399d9467cb359c9fb8c0469a3ab03cecaeafb67bf387` label `arc_w2.0_t0.05_abs_m0.1_autoTrue_tbNone_beamNone` overlap=None
  - pos 2: Fleetwood Mac - Go Your Own Way -> No Age - Flutter Freer | base_T=0.7261979944443382 cur_T=0.8874990957333945 base_dev=0.39555771819091223 cur_dev=0.26323797500881996
  - pos 3: Art Feynman - Early Signs of Rhythm -> Talking Heads - Life During Wartime | base_T=0.8753313515770027 cur_T=0.7718437194466157 base_dev=0.38497219966163226 cur_dev=0.21589957646407731
  - pos 4: Sonic Youth w. Lydia Lunch - Death Valley '69 -> Fleetwood Mac - Go Your Own Way | base_T=0.8197566531001381 cur_T=0.6328555843932391 base_dev=0.23366589741170762 cur_dev=0.12495966811781378
  - pos 5: The Jimi Hendrix Experience - Voodoo Child (Slight Return) -> Art Feynman - Early Signs of Rhythm | base_T=0.8745940561886648 cur_T=0.8753313515770027 base_dev=0.1625239416204261 cur_dev=0.03141880906835853
  - pos 6: The Stooges - T.V. Eye -> David Bowie - Cracked Actor | base_T=0.8266032228365947 cur_T=0.7742549371410699 base_dev=0.07121066038760437 cur_dev=0.09741770737041133
  - pos 7: David Bowie - Cracked Actor -> Prince - Baby I'm a Star ("Take Me with U" 7" B-Side Edit) (Take Me with U 7" B-Side Edit; 2017 Remaster) | base_T=0.8824453880191951 cur_T=0.8793224970039951 base_dev=0.25962938178114026 cur_dev=0.20248261050305993
  - pos 8: Prince - Baby I'm a Star ("Take Me with U" 7" B-Side Edit) (Take Me with U 7" B-Side Edit; 2017 Remaster) -> The Jimi Hendrix Experience - Voodoo Child (Slight Return) | base_T=0.8793224970039951 cur_T=0.5314857512094759 base_dev=0.3108689861654296 cur_dev=0.2994158246352173
  - pos 10: Fleetwood Mac - Blue Letter -> Talking Heads - Mind | base_T=0.5734668558132143 cur_T=0.8634542784698765 base_dev=0.18860675779001007 cur_dev=0.1089753127366695
  - pos 11: Unknown Mortal Orchestra - Swim and Sleep (Like a Shark) -> The Beatles - Yellow Submarine (2022 Mix) | base_T=0.7524002636145654 cur_T=0.7964319722259314 base_dev=0.25883326316655203 cur_dev=0.179700202007851
  - pos 13: The Supremes - Ask Any Girl -> Fleetwood Mac - Blue Letter | base_T=0.5122538172333997 cur_T=0.6820691965383506 base_dev=0.002159490343563275 cur_dev=0.012887702628465059
  - pos 14: Stevie Wonder - Never Had A Dream Come True -> Black Moth Super Rainbow - Last House In the Enchanted Forest | base_T=0.832064059755911 cur_T=0.8747167831653818 base_dev=0.005997006971160401 cur_dev=0.0063320178530172355
  - pos 15: James Brown - Santa Claus, Santa Claus -> Hyldon - Cor De Maçã | base_T=0.5153161003897999 cur_T=0.599202548108818 base_dev=0.0044046067747152695 cur_dev=0.07287166623764857
  - pos 17: Video Age - Blushing -> Norah Jones - Don't Know Why | base_T=0.44735900435803805 cur_T=0.6177673307821179 base_dev=0.063808524900418 cur_dev=0.16778030878295835
  - pos 18: Deerhoof - Fraction Anthem -> The Beach Boys - I Know There's An Answer | base_T=0.4748340800811629 cur_T=0.6245758584405523 base_dev=0.0695872440346178 cur_dev=0.11558086651025046
  - pos 19: Porches - Glow -> Deerhoof - Fraction Anthem | base_T=0.7223826143454176 cur_T=0.3382076170986094 base_dev=0.18416985195786595 cur_dev=0.16439027540520545
  - pos 20: David Bowie - Changes -> Porches - Glow | base_T=0.8127477255057459 cur_T=0.7223826143454176 base_dev=0.19226278364176386 cur_dev=0.2633406922625141
  - pos 24: Godspeed You! Black Emperor - Peasantry or ‘Light! Inside of Light!’ -> Guided By Voices - Order For The New Slave Trade | base_T=0.7247934731539941 cur_T=0.4929041288274822 base_dev=0.434280496159301 cur_dev=0.230187028140788
  - pos 25: Mount Eerie - Night Palace -> Moses Sumney - Gagarin | base_T=0.6996319683550355 cur_T=0.7759193130677012 base_dev=0.3812978609756771 cur_dev=0.1973077453269741
  - pos 26: King Krule - The Cadet Leaps -> Stevie Wonder - I Beleive (When I Fall In Love It Will Be Forever) | base_T=0.535524171086011 cur_T=0.6535986334562918 base_dev=0.13845839189621728 cur_dev=0.08750722898462288
  - pos 27: Sigur Rós - Andrá -> King Krule - The Cadet Leaps | base_T=0.8160552428817576 cur_T=0.535700751887979 base_dev=0.008164849824317755 cur_dev=0.08406254206009711

- hash `d80e2ff8571061b9e567399d9467cb359c9fb8c0469a3ab03cecaeafb67bf387` label `arc_w2.0_t0.08_abs_m0.1_autoFalse_tbNone_beamNone` overlap=None
  - pos 2: Fleetwood Mac - Go Your Own Way -> No Age - Flutter Freer | base_T=0.7261979944443382 cur_T=0.8874990957333945 base_dev=0.39555771819091223 cur_dev=0.26323797500881996
  - pos 3: Art Feynman - Early Signs of Rhythm -> Talking Heads - Life During Wartime | base_T=0.8753313515770027 cur_T=0.7718437194466157 base_dev=0.38497219966163226 cur_dev=0.21589957646407731
  - pos 4: Sonic Youth w. Lydia Lunch - Death Valley '69 -> Fleetwood Mac - Go Your Own Way | base_T=0.8197566531001381 cur_T=0.6328555843932391 base_dev=0.23366589741170762 cur_dev=0.12495966811781378
  - pos 5: The Jimi Hendrix Experience - Voodoo Child (Slight Return) -> Art Feynman - Early Signs of Rhythm | base_T=0.8745940561886648 cur_T=0.8753313515770027 base_dev=0.1625239416204261 cur_dev=0.03141880906835853
  - pos 6: The Stooges - T.V. Eye -> David Bowie - Cracked Actor | base_T=0.8266032228365947 cur_T=0.7742549371410699 base_dev=0.07121066038760437 cur_dev=0.09741770737041133
  - pos 7: David Bowie - Cracked Actor -> Prince - Baby I'm a Star ("Take Me with U" 7" B-Side Edit) (Take Me with U 7" B-Side Edit; 2017 Remaster) | base_T=0.8824453880191951 cur_T=0.8793224970039951 base_dev=0.25962938178114026 cur_dev=0.20248261050305993
  - pos 8: Prince - Baby I'm a Star ("Take Me with U" 7" B-Side Edit) (Take Me with U 7" B-Side Edit; 2017 Remaster) -> The Jimi Hendrix Experience - Voodoo Child (Slight Return) | base_T=0.8793224970039951 cur_T=0.5314857512094759 base_dev=0.3108689861654296 cur_dev=0.2994158246352173
  - pos 10: Fleetwood Mac - Blue Letter -> Talking Heads - Mind | base_T=0.5734668558132143 cur_T=0.8634542784698765 base_dev=0.18860675779001007 cur_dev=0.1089753127366695
  - pos 11: Unknown Mortal Orchestra - Swim and Sleep (Like a Shark) -> The Beatles - Yellow Submarine (2022 Mix) | base_T=0.7524002636145654 cur_T=0.7964319722259314 base_dev=0.25883326316655203 cur_dev=0.179700202007851
  - pos 13: The Supremes - Ask Any Girl -> Fleetwood Mac - Blue Letter | base_T=0.5122538172333997 cur_T=0.6820691965383506 base_dev=0.002159490343563275 cur_dev=0.012887702628465059
  - pos 14: Stevie Wonder - Never Had A Dream Come True -> Black Moth Super Rainbow - Last House In the Enchanted Forest | base_T=0.832064059755911 cur_T=0.8747167831653818 base_dev=0.005997006971160401 cur_dev=0.0063320178530172355
  - pos 15: James Brown - Santa Claus, Santa Claus -> Hyldon - Cor De Maçã | base_T=0.5153161003897999 cur_T=0.599202548108818 base_dev=0.0044046067747152695 cur_dev=0.07287166623764857
  - pos 17: Video Age - Blushing -> Norah Jones - Don't Know Why | base_T=0.44735900435803805 cur_T=0.6177673307821179 base_dev=0.063808524900418 cur_dev=0.16778030878295835
  - pos 18: Deerhoof - Fraction Anthem -> The Beach Boys - I Know There's An Answer | base_T=0.4748340800811629 cur_T=0.6245758584405523 base_dev=0.0695872440346178 cur_dev=0.11558086651025046
  - pos 19: Porches - Glow -> Deerhoof - Fraction Anthem | base_T=0.7223826143454176 cur_T=0.3382076170986094 base_dev=0.18416985195786595 cur_dev=0.16439027540520545
  - pos 20: David Bowie - Changes -> Porches - Glow | base_T=0.8127477255057459 cur_T=0.7223826143454176 base_dev=0.19226278364176386 cur_dev=0.2633406922625141
  - pos 24: Godspeed You! Black Emperor - Peasantry or ‘Light! Inside of Light!’ -> Guided By Voices - Order For The New Slave Trade | base_T=0.7247934731539941 cur_T=0.4929041288274822 base_dev=0.434280496159301 cur_dev=0.230187028140788
  - pos 25: Mount Eerie - Night Palace -> Moses Sumney - Gagarin | base_T=0.6996319683550355 cur_T=0.7759193130677012 base_dev=0.3812978609756771 cur_dev=0.1973077453269741
  - pos 26: King Krule - The Cadet Leaps -> Stevie Wonder - I Beleive (When I Fall In Love It Will Be Forever) | base_T=0.535524171086011 cur_T=0.6535986334562918 base_dev=0.13845839189621728 cur_dev=0.08750722898462288
  - pos 27: Sigur Rós - Andrá -> King Krule - The Cadet Leaps | base_T=0.8160552428817576 cur_T=0.535700751887979 base_dev=0.008164849824317755 cur_dev=0.08406254206009711

- hash `cacb480844940559819e44612b8e5de4f0b21951e58dfeb80856bc62db40a5f7` label `arc_w2.0_t0.08_abs_m0.4_autoFalse_tbNone_beam50` overlap=None
  - pos 1: David Bowie - Suffragette City -> David Bowie - Life On Mars? | base_T=None cur_T=None base_dev=None cur_dev=None
  - pos 2: Fleetwood Mac - Go Your Own Way -> teen suicide - My Little World | base_T=0.7261979944443382 cur_T=0.8061904896633129 base_dev=0.39555771819091223 cur_dev=0.3433774234597585
  - pos 3: Art Feynman - Early Signs of Rhythm -> Moses Sumney - Gagarin | base_T=0.8753313515770027 cur_T=0.6792681330380438 base_dev=0.38497219966163226 cur_dev=0.23911623499088108
  - pos 4: Sonic Youth w. Lydia Lunch - Death Valley '69 -> St. Vincent - The Bed | base_T=0.8197566531001381 cur_T=0.826443004919119 base_dev=0.23366589741170762 cur_dev=0.2192099496484453
  - pos 5: The Jimi Hendrix Experience - Voodoo Child (Slight Return) -> King Krule - The Cadet Leaps | base_T=0.8745940561886648 cur_T=0.8130570399659223 base_dev=0.1625239416204261 cur_dev=0.027197924918060112
  - pos 6: The Stooges - T.V. Eye -> Sigur Rós - Andrá | base_T=0.8266032228365947 cur_T=0.8160552428817576 base_dev=0.07121066038760437 cur_dev=0.07191639938006988
  - pos 7: David Bowie - Cracked Actor -> Foxes in Fiction - Insomnia Keys | base_T=0.8824453880191951 cur_T=0.7853552801920686 base_dev=0.25962938178114026 cur_dev=0.18482013225565375
  - pos 8: Prince - Baby I'm a Star ("Take Me with U" 7" B-Side Edit) (Take Me with U 7" B-Side Edit; 2017 Remaster) -> M83 - Another Wave from You | base_T=0.8793224970039951 cur_T=0.6604160308918139 base_dev=0.3108689861654296 cur_dev=0.31354826082438414
  - pos 9: David Bowie - Let's Dance -> David Bowie - Warszawa | base_T=0.7845443668679988 cur_T=0.8865096925932843 base_dev=None cur_dev=None
  - pos 10: Fleetwood Mac - Blue Letter -> Cornelius - Surfing on Mind Wave, Pt. 2 | base_T=0.5734668558132143 cur_T=0.8034951146243114 base_dev=0.18860675779001007 cur_dev=0.2072080595593752
  - pos 11: Unknown Mortal Orchestra - Swim and Sleep (Like a Shark) -> Microphones - Here With Summer | base_T=0.7524002636145654 cur_T=0.8398301230820251 base_dev=0.25883326316655203 cur_dev=0.0800313041791113
  - pos 12: Prince - Take Me with U (2015 Paisley Park Remaster) -> PaPerCuts - Mattress on the Floor | base_T=0.5663842533065095 cur_T=0.8967572984165522 base_dev=0.0025446620136969217 cur_dev=0.05568463355117209
  - pos 13: The Supremes - Ask Any Girl -> Dirty Beaches - Sweet 17 | base_T=0.5122538172333997 cur_T=0.8756926284369257 base_dev=0.002159490343563275 cur_dev=0.02039670325270082
  - pos 14: Stevie Wonder - Never Had A Dream Come True -> Television Personalities - David Hockney's Diary | base_T=0.832064059755911 cur_T=0.9142592763980442 base_dev=0.005997006971160401 cur_dev=0.17267414574460205
  - pos 15: James Brown - Santa Claus, Santa Claus -> Ramones - Glad to See You Go (40th Anniversary Mix) | base_T=0.5153161003897999 cur_T=0.829455849593771 base_dev=0.0044046067747152695 cur_dev=0.3090328786583877
  - pos 16: Marvin Gaye - I Wanna Be Where You Are -> David Bowie - Suffragette City | base_T=0.7758543087250149 cur_T=0.7586708660369917 base_dev=0.1458928281625957 cur_dev=None
  - pos 17: Video Age - Blushing -> The Fall - Perverted by Languagelive at Electric Ballroom, London 8 December 1983 | base_T=0.44735900435803805 cur_T=0.8926473957940299 base_dev=0.063808524900418 cur_dev=0.2692867594119475
  - pos 18: Deerhoof - Fraction Anthem -> Art Feynman - Early Signs of Rhythm | base_T=0.4748340800811629 cur_T=0.8374564949764935 base_dev=0.0695872440346178 cur_dev=0.48190324301956805
  - pos 19: Porches - Glow -> Talking Heads - Cities | base_T=0.7223826143454176 cur_T=0.8609959869184659 base_dev=0.18416985195786595 cur_dev=0.3584849210937879
  - pos 20: David Bowie - Changes -> The Beach Boys - Our Car Club (Mono) | base_T=0.8127477255057459 cur_T=0.8627276767248913 base_dev=0.19226278364176386 cur_dev=0.21503277677762772

- hash `f92bae6baa7b44b88b534c4f3b6ff28117aa897666988fb0054f482cd722513d` label `arc_w2.0_t0.0_abs_m0.15_autoTrue_tbNone_beamNone` overlap=None
  - pos 2: Fleetwood Mac - Go Your Own Way -> No Age - Flutter Freer | base_T=0.7261979944443382 cur_T=0.8874990957333945 base_dev=0.39555771819091223 cur_dev=0.26323797500881996
  - pos 3: Art Feynman - Early Signs of Rhythm -> Talking Heads - Life During Wartime | base_T=0.8753313515770027 cur_T=0.7718437194466157 base_dev=0.38497219966163226 cur_dev=0.21589957646407731
  - pos 4: Sonic Youth w. Lydia Lunch - Death Valley '69 -> Fleetwood Mac - Go Your Own Way | base_T=0.8197566531001381 cur_T=0.6328555843932391 base_dev=0.23366589741170762 cur_dev=0.12495966811781378
  - pos 5: The Jimi Hendrix Experience - Voodoo Child (Slight Return) -> Art Feynman - Early Signs of Rhythm | base_T=0.8745940561886648 cur_T=0.8753313515770027 base_dev=0.1625239416204261 cur_dev=0.03141880906835853
  - pos 6: The Stooges - T.V. Eye -> Prince - Baby I'm a Star ("Take Me with U" 7" B-Side Edit) (Take Me with U 7" B-Side Edit; 2017 Remaster) | base_T=0.8266032228365947 cur_T=0.8888348523258907 base_dev=0.07121066038760437 cur_dev=0.04027093609233101
  - pos 7: David Bowie - Cracked Actor -> The Jimi Hendrix Experience - Voodoo Child (Slight Return) | base_T=0.8824453880191951 cur_T=0.5314857512094759 base_dev=0.25962938178114026 cur_dev=0.19102944897284768
  - pos 8: Prince - Baby I'm a Star ("Take Me with U" 7" B-Side Edit) (Take Me with U 7" B-Side Edit; 2017 Remaster) -> The Stooges - T.V. Eye | base_T=0.8793224970039951 cur_T=0.8266032228365947 base_dev=0.3108689861654296 cur_dev=0.34180871046070294
  - pos 10: Fleetwood Mac - Blue Letter -> Talking Heads - Mind | base_T=0.5734668558132143 cur_T=0.8634542784698765 base_dev=0.18860675779001007 cur_dev=0.1089753127366695
  - pos 11: Unknown Mortal Orchestra - Swim and Sleep (Like a Shark) -> The Beatles - Yellow Submarine (2022 Mix) | base_T=0.7524002636145654 cur_T=0.7964319722259314 base_dev=0.25883326316655203 cur_dev=0.179700202007851
  - pos 13: The Supremes - Ask Any Girl -> Fleetwood Mac - Blue Letter | base_T=0.5122538172333997 cur_T=0.6820691965383506 base_dev=0.002159490343563275 cur_dev=0.012887702628465059
  - pos 14: Stevie Wonder - Never Had A Dream Come True -> Black Moth Super Rainbow - Last House In the Enchanted Forest | base_T=0.832064059755911 cur_T=0.8747167831653818 base_dev=0.005997006971160401 cur_dev=0.0063320178530172355
  - pos 15: James Brown - Santa Claus, Santa Claus -> Hyldon - Cor De Maçã | base_T=0.5153161003897999 cur_T=0.599202548108818 base_dev=0.0044046067747152695 cur_dev=0.07287166623764857
  - pos 17: Video Age - Blushing -> Norah Jones - Don't Know Why | base_T=0.44735900435803805 cur_T=0.6177673307821179 base_dev=0.063808524900418 cur_dev=0.16778030878295835
  - pos 18: Deerhoof - Fraction Anthem -> The Beach Boys - I Know There's An Answer | base_T=0.4748340800811629 cur_T=0.6245758584405523 base_dev=0.0695872440346178 cur_dev=0.11558086651025046
  - pos 19: Porches - Glow -> Deerhoof - Fraction Anthem | base_T=0.7223826143454176 cur_T=0.3382076170986094 base_dev=0.18416985195786595 cur_dev=0.16439027540520545
  - pos 20: David Bowie - Changes -> Porches - Glow | base_T=0.8127477255057459 cur_T=0.7223826143454176 base_dev=0.19226278364176386 cur_dev=0.2633406922625141
  - pos 24: Godspeed You! Black Emperor - Peasantry or ‘Light! Inside of Light!’ -> Moses Sumney - Gagarin | base_T=0.7247934731539941 cur_T=0.6210902344886864 base_dev=0.434280496159301 cur_dev=0.33604727834881687
  - pos 25: Mount Eerie - Night Palace -> Stevie Wonder - I Beleive (When I Fall In Love It Will Be Forever) | base_T=0.6996319683550355 cur_T=0.6535986334562918 base_dev=0.3812978609756771 cur_dev=0.28799166293583245

- hash `d49ac2b9e99e464f0cd940e69262016b77a81e1341a7bfbf16ffda8f3ab62d8f` label `arc_w2.0_t0.0_abs_m0.3_autoTrue_tbNone_beamNone` overlap=None
  - pos 2: Fleetwood Mac - Go Your Own Way -> No Age - Flutter Freer | base_T=0.7261979944443382 cur_T=0.8874990957333945 base_dev=0.39555771819091223 cur_dev=0.26323797500881996
  - pos 3: Art Feynman - Early Signs of Rhythm -> New Order - Sooner Than You Think | base_T=0.8753313515770027 cur_T=0.7637640115534492 base_dev=0.38497219966163226 cur_dev=0.18310379875102856
  - pos 4: Sonic Youth w. Lydia Lunch - Death Valley '69 -> Talking Heads - Life During Wartime | base_T=0.8197566531001381 cur_T=0.8286986811809618 base_dev=0.23366589741170762 cur_dev=0.053687902053348446
  - pos 5: The Jimi Hendrix Experience - Voodoo Child (Slight Return) -> Art Feynman - Early Signs of Rhythm | base_T=0.8745940561886648 cur_T=0.7287241576301154 base_dev=0.1625239416204261 cur_dev=0.03141880906835853
  - pos 6: The Stooges - T.V. Eye -> Prince - Baby I'm a Star ("Take Me with U" 7" B-Side Edit) (Take Me with U 7" B-Side Edit; 2017 Remaster) | base_T=0.8266032228365947 cur_T=0.8888348523258907 base_dev=0.07121066038760437 cur_dev=0.04027093609233101
  - pos 7: David Bowie - Cracked Actor -> The Jimi Hendrix Experience - Voodoo Child (Slight Return) | base_T=0.8824453880191951 cur_T=0.5314857512094759 base_dev=0.25962938178114026 cur_dev=0.19102944897284768
  - pos 8: Prince - Baby I'm a Star ("Take Me with U" 7" B-Side Edit) (Take Me with U 7" B-Side Edit; 2017 Remaster) -> The Stooges - T.V. Eye | base_T=0.8793224970039951 cur_T=0.8266032228365947 base_dev=0.3108689861654296 cur_dev=0.34180871046070294
  - pos 10: Fleetwood Mac - Blue Letter -> Talking Heads - Mind | base_T=0.5734668558132143 cur_T=0.8634542784698765 base_dev=0.18860675779001007 cur_dev=0.1089753127366695
  - pos 11: Unknown Mortal Orchestra - Swim and Sleep (Like a Shark) -> Pixies - Here Comes Your Man | base_T=0.7524002636145654 cur_T=0.5376986426370051 base_dev=0.25883326316655203 cur_dev=0.027841799077980192
  - pos 12: Prince - Take Me with U (2015 Paisley Park Remaster) -> The Beatles - Yellow Submarine (2022 Mix) | base_T=0.5663842533065095 cur_T=0.5524427188893772 base_dev=0.0025446620136969217 cur_dev=0.12013150929065633
  - pos 13: The Supremes - Ask Any Girl -> Fleetwood Mac - Blue Letter | base_T=0.5122538172333997 cur_T=0.5054779736540025 base_dev=0.002159490343563275 cur_dev=0.012887702628465059
  - pos 14: Stevie Wonder - Never Had A Dream Come True -> Tom Waits - Chick a Boom | base_T=0.832064059755911 cur_T=0.821306503377541 base_dev=0.005997006971160401 cur_dev=0.03500376743053363
  - pos 15: James Brown - Santa Claus, Santa Claus -> Hyldon - Cor De Maçã | base_T=0.5153161003897999 cur_T=0.6223922831079162 base_dev=0.0044046067747152695 cur_dev=0.07287166623764857
  - pos 17: Video Age - Blushing -> Norah Jones - Don't Know Why | base_T=0.44735900435803805 cur_T=0.6177673307821179 base_dev=0.063808524900418 cur_dev=0.16778030878295835
  - pos 18: Deerhoof - Fraction Anthem -> The Beach Boys - I Know There's An Answer | base_T=0.4748340800811629 cur_T=0.6245758584405523 base_dev=0.0695872440346178 cur_dev=0.11558086651025046
  - pos 19: Porches - Glow -> Deerhoof - Fraction Anthem | base_T=0.7223826143454176 cur_T=0.3382076170986094 base_dev=0.18416985195786595 cur_dev=0.16439027540520545
  - pos 20: David Bowie - Changes -> Porches - Glow | base_T=0.8127477255057459 cur_T=0.7223826143454176 base_dev=0.19226278364176386 cur_dev=0.2633406922625141
  - pos 24: Godspeed You! Black Emperor - Peasantry or ‘Light! Inside of Light!’ -> Moses Sumney - Gagarin | base_T=0.7247934731539941 cur_T=0.6210902344886864 base_dev=0.434280496159301 cur_dev=0.33604727834881687
  - pos 25: Mount Eerie - Night Palace -> Stevie Wonder - I Beleive (When I Fall In Love It Will Be Forever) | base_T=0.6996319683550355 cur_T=0.6535986334562918 base_dev=0.3812978609756771 cur_dev=0.28799166293583245

- hash `b6c1bb87090f85faac66c222acb67d1ec03598c6848b0cd3ee93cfbc7608cba5` label `arc_w2.0_t0.0_abs_m0.4_autoTrue_tb0.02_beam50` overlap=None
  - pos 1: David Bowie - Suffragette City -> David Bowie - Life On Mars? | base_T=None cur_T=None base_dev=None cur_dev=None
  - pos 2: Fleetwood Mac - Go Your Own Way -> teen suicide - My Little World | base_T=0.7261979944443382 cur_T=0.8061904896633129 base_dev=0.39555771819091223 cur_dev=0.3433774234597585
  - pos 3: Art Feynman - Early Signs of Rhythm -> Moses Sumney - Gagarin | base_T=0.8753313515770027 cur_T=0.6792681330380438 base_dev=0.38497219966163226 cur_dev=0.23911623499088108
  - pos 4: Sonic Youth w. Lydia Lunch - Death Valley '69 -> St. Vincent - The Bed | base_T=0.8197566531001381 cur_T=0.826443004919119 base_dev=0.23366589741170762 cur_dev=0.2192099496484453
  - pos 5: The Jimi Hendrix Experience - Voodoo Child (Slight Return) -> King Krule - The Cadet Leaps | base_T=0.8745940561886648 cur_T=0.8130570399659223 base_dev=0.1625239416204261 cur_dev=0.027197924918060112
  - pos 6: The Stooges - T.V. Eye -> Sigur Rós - Andrá | base_T=0.8266032228365947 cur_T=0.8160552428817576 base_dev=0.07121066038760437 cur_dev=0.07191639938006988
  - pos 7: David Bowie - Cracked Actor -> Foxes in Fiction - Insomnia Keys | base_T=0.8824453880191951 cur_T=0.7853552801920686 base_dev=0.25962938178114026 cur_dev=0.18482013225565375
  - pos 8: Prince - Baby I'm a Star ("Take Me with U" 7" B-Side Edit) (Take Me with U 7" B-Side Edit; 2017 Remaster) -> M83 - Another Wave from You | base_T=0.8793224970039951 cur_T=0.6604160308918139 base_dev=0.3108689861654296 cur_dev=0.31354826082438414
  - pos 9: David Bowie - Let's Dance -> David Bowie - Warszawa | base_T=0.7845443668679988 cur_T=0.8865096925932843 base_dev=None cur_dev=None
  - pos 10: Fleetwood Mac - Blue Letter -> Cornelius - Surfing on Mind Wave, Pt. 2 | base_T=0.5734668558132143 cur_T=0.8034951146243114 base_dev=0.18860675779001007 cur_dev=0.2072080595593752
  - pos 11: Unknown Mortal Orchestra - Swim and Sleep (Like a Shark) -> Microphones - Here With Summer | base_T=0.7524002636145654 cur_T=0.8398301230820251 base_dev=0.25883326316655203 cur_dev=0.0800313041791113
  - pos 12: Prince - Take Me with U (2015 Paisley Park Remaster) -> PaPerCuts - Mattress on the Floor | base_T=0.5663842533065095 cur_T=0.8967572984165522 base_dev=0.0025446620136969217 cur_dev=0.05568463355117209
  - pos 13: The Supremes - Ask Any Girl -> Dirty Beaches - Sweet 17 | base_T=0.5122538172333997 cur_T=0.8756926284369257 base_dev=0.002159490343563275 cur_dev=0.02039670325270082
  - pos 14: Stevie Wonder - Never Had A Dream Come True -> Television Personalities - David Hockney's Diary | base_T=0.832064059755911 cur_T=0.9142592763980442 base_dev=0.005997006971160401 cur_dev=0.17267414574460205
  - pos 15: James Brown - Santa Claus, Santa Claus -> Ramones - Glad to See You Go (40th Anniversary Mix) | base_T=0.5153161003897999 cur_T=0.829455849593771 base_dev=0.0044046067747152695 cur_dev=0.3090328786583877
  - pos 16: Marvin Gaye - I Wanna Be Where You Are -> David Bowie - Suffragette City | base_T=0.7758543087250149 cur_T=0.7586708660369917 base_dev=0.1458928281625957 cur_dev=None
  - pos 17: Video Age - Blushing -> The Fall - Perverted by Languagelive at Electric Ballroom, London 8 December 1983 | base_T=0.44735900435803805 cur_T=0.8926473957940299 base_dev=0.063808524900418 cur_dev=0.2692867594119475
  - pos 18: Deerhoof - Fraction Anthem -> Art Feynman - Early Signs of Rhythm | base_T=0.4748340800811629 cur_T=0.8374564949764935 base_dev=0.0695872440346178 cur_dev=0.48190324301956805
  - pos 19: Porches - Glow -> Talking Heads - Cities | base_T=0.7223826143454176 cur_T=0.8609959869184659 base_dev=0.18416985195786595 cur_dev=0.3584849210937879
  - pos 20: David Bowie - Changes -> The Beach Boys - Our Car Club (Mono) | base_T=0.8127477255057459 cur_T=0.8627276767248913 base_dev=0.19226278364176386 cur_dev=0.21503277677762772

- hash `bd8ea5d4e1c5763c79aeb4bb145b2ba939c7d585923d498703ca9a358e5c6378` label `arc_w2.0_t0.2_abs_m0.1_autoTrue_tbNone_beam25` overlap=None
  - pos 1: David Bowie - Suffragette City -> David Bowie - Life On Mars? | base_T=None cur_T=None base_dev=None cur_dev=None
  - pos 2: Fleetwood Mac - Go Your Own Way -> teen suicide - My Little World | base_T=0.7261979944443382 cur_T=0.8061904896633129 base_dev=0.39555771819091223 cur_dev=0.3433774234597585
  - pos 3: Art Feynman - Early Signs of Rhythm -> Tom Waits - It's Over (Remastered) | base_T=0.8753313515770027 cur_T=0.7695236674405489 base_dev=0.38497219966163226 cur_dev=0.27705930792705674
  - pos 4: Sonic Youth w. Lydia Lunch - Death Valley '69 -> St. Vincent - The Bed | base_T=0.8197566531001381 cur_T=0.7780357102243484 base_dev=0.23366589741170762 cur_dev=0.2192099496484453
  - pos 5: The Jimi Hendrix Experience - Voodoo Child (Slight Return) -> King Krule - The Cadet Leaps | base_T=0.8745940561886648 cur_T=0.8130570399659223 base_dev=0.1625239416204261 cur_dev=0.027197924918060112
  - pos 6: The Stooges - T.V. Eye -> Sigur Rós - Andrá | base_T=0.8266032228365947 cur_T=0.8160552428817576 base_dev=0.07121066038760437 cur_dev=0.07191639938006988
  - pos 7: David Bowie - Cracked Actor -> Foxes in Fiction - Insomnia Keys | base_T=0.8824453880191951 cur_T=0.7853552801920686 base_dev=0.25962938178114026 cur_dev=0.18482013225565375
  - pos 8: Prince - Baby I'm a Star ("Take Me with U" 7" B-Side Edit) (Take Me with U 7" B-Side Edit; 2017 Remaster) -> M83 - Another Wave from You | base_T=0.8793224970039951 cur_T=0.6604160308918139 base_dev=0.3108689861654296 cur_dev=0.31354826082438414
  - pos 9: David Bowie - Let's Dance -> David Bowie - Warszawa | base_T=0.7845443668679988 cur_T=0.8865096925932843 base_dev=None cur_dev=None
  - pos 10: Fleetwood Mac - Blue Letter -> Cornelius - Surfing on Mind Wave, Pt. 2 | base_T=0.5734668558132143 cur_T=0.8034951146243114 base_dev=0.18860675779001007 cur_dev=0.2072080595593752
  - pos 11: Unknown Mortal Orchestra - Swim and Sleep (Like a Shark) -> Microphones - Here With Summer | base_T=0.7524002636145654 cur_T=0.8398301230820251 base_dev=0.25883326316655203 cur_dev=0.0800313041791113
  - pos 12: Prince - Take Me with U (2015 Paisley Park Remaster) -> PaPerCuts - Mattress on the Floor | base_T=0.5663842533065095 cur_T=0.8967572984165522 base_dev=0.0025446620136969217 cur_dev=0.05568463355117209
  - pos 13: The Supremes - Ask Any Girl -> Dirty Beaches - Sweet 17 | base_T=0.5122538172333997 cur_T=0.8756926284369257 base_dev=0.002159490343563275 cur_dev=0.02039670325270082
  - pos 14: Stevie Wonder - Never Had A Dream Come True -> Television Personalities - David Hockney's Diary | base_T=0.832064059755911 cur_T=0.9142592763980442 base_dev=0.005997006971160401 cur_dev=0.17267414574460205
  - pos 15: James Brown - Santa Claus, Santa Claus -> Ramones - Glad to See You Go (40th Anniversary Mix) | base_T=0.5153161003897999 cur_T=0.829455849593771 base_dev=0.0044046067747152695 cur_dev=0.3090328786583877
  - pos 16: Marvin Gaye - I Wanna Be Where You Are -> David Bowie - Suffragette City | base_T=0.7758543087250149 cur_T=0.7586708660369917 base_dev=0.1458928281625957 cur_dev=None
  - pos 17: Video Age - Blushing -> Tyvek - Kid Tut | base_T=0.44735900435803805 cur_T=0.879525228136799 base_dev=0.063808524900418 cur_dev=0.27140003943548924
  - pos 18: Deerhoof - Fraction Anthem -> Fleetwood Mac - Go Your Own Way | base_T=0.4748340800811629 cur_T=0.8310964276920598 base_dev=0.0695872440346178 cur_dev=0.38410238588647844
  - pos 19: Porches - Glow -> Art Feynman - Early Signs of Rhythm | base_T=0.7223826143454176 cur_T=0.8753313515770027 base_dev=0.18416985195786595 cur_dev=0.4223345503023734
  - pos 20: David Bowie - Changes -> The Beach Boys - Our Car Club (Mono) | base_T=0.8127477255057459 cur_T=0.8092975068410233 base_dev=0.19226278364176386 cur_dev=0.21503277677762772

- hash `ea5bbde926138bc14c86fc4a106a2ebcc1210622972c457d4b7351bb34565486` label `arc_w2.0_t0.2_abs_m0.3_autoFalse_tbNone_beam50` overlap=None
  - pos 1: David Bowie - Suffragette City -> David Bowie - Life On Mars? | base_T=None cur_T=None base_dev=None cur_dev=None
  - pos 2: Fleetwood Mac - Go Your Own Way -> teen suicide - My Little World | base_T=0.7261979944443382 cur_T=0.8061904896633129 base_dev=0.39555771819091223 cur_dev=0.3433774234597585
  - pos 3: Art Feynman - Early Signs of Rhythm -> Moses Sumney - Gagarin | base_T=0.8753313515770027 cur_T=0.6792681330380438 base_dev=0.38497219966163226 cur_dev=0.23911623499088108
  - pos 4: Sonic Youth w. Lydia Lunch - Death Valley '69 -> St. Vincent - The Bed | base_T=0.8197566531001381 cur_T=0.826443004919119 base_dev=0.23366589741170762 cur_dev=0.2192099496484453
  - pos 5: The Jimi Hendrix Experience - Voodoo Child (Slight Return) -> King Krule - The Cadet Leaps | base_T=0.8745940561886648 cur_T=0.8130570399659223 base_dev=0.1625239416204261 cur_dev=0.027197924918060112
  - pos 6: The Stooges - T.V. Eye -> Sigur Rós - Andrá | base_T=0.8266032228365947 cur_T=0.8160552428817576 base_dev=0.07121066038760437 cur_dev=0.07191639938006988
  - pos 7: David Bowie - Cracked Actor -> Foxes in Fiction - Insomnia Keys | base_T=0.8824453880191951 cur_T=0.7853552801920686 base_dev=0.25962938178114026 cur_dev=0.18482013225565375
  - pos 8: Prince - Baby I'm a Star ("Take Me with U" 7" B-Side Edit) (Take Me with U 7" B-Side Edit; 2017 Remaster) -> M83 - Another Wave from You | base_T=0.8793224970039951 cur_T=0.6604160308918139 base_dev=0.3108689861654296 cur_dev=0.31354826082438414
  - pos 9: David Bowie - Let's Dance -> David Bowie - Warszawa | base_T=0.7845443668679988 cur_T=0.8865096925932843 base_dev=None cur_dev=None
  - pos 10: Fleetwood Mac - Blue Letter -> Cornelius - Surfing on Mind Wave, Pt. 2 | base_T=0.5734668558132143 cur_T=0.8034951146243114 base_dev=0.18860675779001007 cur_dev=0.2072080595593752
  - pos 11: Unknown Mortal Orchestra - Swim and Sleep (Like a Shark) -> Microphones - Here With Summer | base_T=0.7524002636145654 cur_T=0.8398301230820251 base_dev=0.25883326316655203 cur_dev=0.0800313041791113
  - pos 12: Prince - Take Me with U (2015 Paisley Park Remaster) -> PaPerCuts - Mattress on the Floor | base_T=0.5663842533065095 cur_T=0.8967572984165522 base_dev=0.0025446620136969217 cur_dev=0.05568463355117209
  - pos 13: The Supremes - Ask Any Girl -> Dirty Beaches - Sweet 17 | base_T=0.5122538172333997 cur_T=0.8756926284369257 base_dev=0.002159490343563275 cur_dev=0.02039670325270082
  - pos 14: Stevie Wonder - Never Had A Dream Come True -> Television Personalities - David Hockney's Diary | base_T=0.832064059755911 cur_T=0.9142592763980442 base_dev=0.005997006971160401 cur_dev=0.17267414574460205
  - pos 15: James Brown - Santa Claus, Santa Claus -> Ramones - Glad to See You Go (40th Anniversary Mix) | base_T=0.5153161003897999 cur_T=0.829455849593771 base_dev=0.0044046067747152695 cur_dev=0.3090328786583877
  - pos 16: Marvin Gaye - I Wanna Be Where You Are -> David Bowie - Suffragette City | base_T=0.7758543087250149 cur_T=0.7586708660369917 base_dev=0.1458928281625957 cur_dev=None
  - pos 17: Video Age - Blushing -> Tyvek - Kid Tut | base_T=0.44735900435803805 cur_T=0.879525228136799 base_dev=0.063808524900418 cur_dev=0.27140003943548924
  - pos 18: Deerhoof - Fraction Anthem -> Fleetwood Mac - Go Your Own Way | base_T=0.4748340800811629 cur_T=0.8310964276920598 base_dev=0.0695872440346178 cur_dev=0.38410238588647844
  - pos 19: Porches - Glow -> Art Feynman - Early Signs of Rhythm | base_T=0.7223826143454176 cur_T=0.8753313515770027 base_dev=0.18416985195786595 cur_dev=0.4223345503023734
  - pos 20: David Bowie - Changes -> The Beach Boys - Our Car Club (Mono) | base_T=0.8127477255057459 cur_T=0.8092975068410233 base_dev=0.19226278364176386 cur_dev=0.21503277677762772

### seed 09daedfeff5751c11bd5a39b97475425
- baseline hash: `50a1079806aacc471777be193c72d9d555a460c3e0edc48542d5a334537a860c`

- hash `e6e052df46509e66b7a159d7e158e985172e2dd4a41518b7e7486e9d4798b2f2` label `arc_w0.75_t0.0_abs_m0.1_autoFalse_tb0.02_beam50` overlap=None
  - pos 1: David Bowie - Let's Dance -> David Bowie - Life On Mars? | base_T=None cur_T=None base_dev=None cur_dev=None
  - pos 2: Prince - Baby I'm a Star ("Take Me with U" 7" B-Side Edit) (Take Me with U 7" B-Side Edit; 2017 Remaster) -> Wednesday - Carolina Murder Suicide | base_T=0.7877364949183092 cur_T=0.853585374207696 base_dev=0.31086898616542963 cur_dev=0.4059412324158763
  - pos 3: Dead Kennedys - Viva Las Vegas (2022 Mix) -> Ekko Astral - burning alive on k st. | base_T=0.9038338369336413 cur_T=0.7931098502510812 base_dev=0.3234365407861713 cur_dev=0.41032519900269543
  - pos 4: Big Black - Strange Things -> St. Vincent - The Bed | base_T=0.868413413419985 cur_T=0.8778668052631965 base_dev=0.13430912676290752 cur_dev=0.2192099496484453
  - pos 5: Nirvana - Swap Meet (Remastered) -> King Krule - The Cadet Leaps | base_T=0.8330160032210623 cur_T=0.8130570399659223 base_dev=0.05792099456554084 cur_dev=0.027197924918060112
  - pos 6: Minutemen - Corona -> Sigur Rós - Andrá | base_T=0.8609920669629056 cur_T=0.8160552428817576 base_dev=0.11121147782532603 cur_dev=0.07191639938006988
  - pos 7: Reatards - Your Old News Baby -> Codeine - Second Chance | base_T=0.8868485626989104 cur_T=0.6343020234150984 base_dev=0.20592478598467834 cur_dev=0.17720610789946512
  - pos 8: Talking Heads - Life During Wartime -> M83 - Another Wave from You | base_T=0.7747388414523939 cur_T=0.812172068147903 base_dev=0.32428595212644684 cur_dev=0.31354826082438414
  - pos 9: David Bowie - Suffragette City -> David Bowie - Warszawa | base_T=0.6329123608906382 cur_T=0.8865096925932843 base_dev=None cur_dev=None
  - pos 10: The Beatles - Long Tall Sally -> Microphones - Here With Summer | base_T=0.8438698874380759 cur_T=0.6552326095259688 base_dev=0.1837260675379918 cur_dev=0.21877083720095408
  - pos 11: The Fall - Just Step Sideways -> PaPerCuts - Mattress on the Floor | base_T=0.9020908500716743 cur_T=0.8967572984165522 base_dev=0.28779182893992095 cur_dev=0.25616906750238166
  - pos 12: The Beets - Friends Of Friends -> Guided By Voices - Tractor Rape Chain | base_T=0.7690583079574286 cur_T=0.8780779985451096 base_dev=0.07103774067145574 cur_dev=0.04132543425308627
  - pos 13: Seapony - Dreaming -> Ariel Pink's Dark Side - Queen of the Virgins | base_T=0.5812891962062137 cur_T=0.8572865593008591 base_dev=0.15653896152558605 cur_dev=0.09525394820549138
  - pos 14: Ariel Pink - Off The Dome -> Bikini Kill - Rebel Girl | base_T=0.8816809087878533 cur_T=0.8221788176610882 base_dev=0.04484929428316553 cur_dev=0.22193373855142984
  - pos 15: 48 Chairs - You Were Never There -> Tyvek - Going Through My Things | base_T=0.6213349949035449 cur_T=0.9268529584952317 base_dev=0.05632035573627997 cur_dev=0.41020552771838203
  - pos 16: Sonic Youth - Dude Ranch Nurse -> David Bowie - Suffragette City | base_T=0.6897304399639835 cur_T=0.8524873868886713 base_dev=0.020540089255099758 cur_dev=None
  - pos 17: Helvetia - 3 Boys -> No Age - Flutter Freer | base_T=0.8019982511029264 cur_T=0.8874990957333945 base_dev=0.08813442003398564 cur_dev=0.25178264270438616
  - pos 18: Wilco - Please Be Patient with Me -> Reatards - Your Old News Baby | base_T=0.5282689610509975 cur_T=0.8442933812141848 base_dev=0.01117756888181809 cur_dev=0.16411629632077135
  - pos 19: Sufjan Stevens - That Was the Worst Christmas Ever! -> Nirvana - Swap Meet (Remastered) | base_T=0.7544948044875062 cur_T=0.8565112579200121 base_dev=0.08593498488629436 cur_dev=0.05333947241261655
  - pos 20: Sam Wilkes - Own -> Art Feynman - Early Signs of Rhythm | base_T=0.5726484074044664 cur_T=0.8878898426648157 base_dev=0.2222045506634247 cur_dev=0.07984165790979869

- hash `84f9c7b15a675edbf2486e03016eb53808feb3fd3c9746bd39e7943299a0a1f4` label `arc_w1.5_t0.05_abs_m0.2_autoFalse_tb0.02_beamNone` overlap=None
  - pos 6: Minutemen - Corona -> Reatards - Your Old News Baby | base_T=0.8609920669629056 cur_T=0.8339293121838041 base_dev=0.11121147782532603 cur_dev=0.043713111573949415
  - pos 7: Reatards - Your Old News Baby -> Tyvek - Mary Ellen Claims | base_T=0.8868485626989104 cur_T=0.8857081316422318 base_dev=0.20592478598467834 cur_dev=0.14419335755346485
  - pos 8: Talking Heads - Life During Wartime -> No Age - Flutter Freer | base_T=0.7747388414523939 cur_T=0.7996202067251209 base_dev=0.32428595212644684 cur_dev=0.26323797500881996
  - pos 10: The Beatles - Long Tall Sally -> Here We Go Magic - Tunnelvision | base_T=0.8438698874380759 cur_T=0.8576887312447541 base_dev=0.1837260675379918 cur_dev=0.22503054049837937
  - pos 11: The Fall - Just Step Sideways -> Guided By Voices - Mincer Ray | base_T=0.9020908500716743 cur_T=0.7963139839760061 base_dev=0.28779182893992095 cur_dev=0.2529436298268758
  - pos 12: The Beets - Friends Of Friends -> The Beatles - Long Tall Sally | base_T=0.7690583079574286 cur_T=0.8235979338072454 base_dev=0.07103774067145574 cur_dev=0.0871778526810949
  - pos 18: Wilco - Please Be Patient with Me -> Early Day Miners - The Way We Live Now | base_T=0.5282689610509975 cur_T=0.37890212046114924 base_dev=0.01117756888181809 cur_dev=0.0026778613017034436
  - pos 19: Sufjan Stevens - That Was the Worst Christmas Ever! -> Wilco - Please Be Patient with Me | base_T=0.7544948044875062 cur_T=0.7108870752574645 base_dev=0.08593498488629436 cur_dev=0.08362546248876956
  - pos 20: Sam Wilkes - Own -> Sufjan Stevens - That Was the Worst Christmas Ever! | base_T=0.5726484074044664 cur_T=0.7544948044875062 base_dev=0.2222045506634247 cur_dev=0.16510582519094252
  - pos 24: Godspeed You! Black Emperor - Peasantry or ‘Light! Inside of Light!’ -> Helvetia - Inverted | base_T=0.7247934731539941 cur_T=0.8006093138472479 base_dev=0.434280496159301 cur_dev=0.22134405671042223
  - pos 25: King Krule - The Cadet Leaps -> SPIRIT OF THE BEEHIVE - 1/500 | base_T=0.6955376759430785 cur_T=0.6241630687142188 base_dev=0.33894282584742685 cur_dev=0.22209661668943015
  - pos 26: Sigur Rós - Andrá -> King Krule - The Cadet Leaps | base_T=0.8160552428817576 cur_T=0.5275679496756788 base_dev=0.23068578378063215 cur_dev=0.13845839189621728
  - pos 28: Codeine - Second Chance -> Sigur Rós - Andrá | base_T=0.824862215843409 cur_T=0.6915031512808127 base_dev=0.13539761823555807 cur_dev=0.19231958412689176
  - pos 29: Spacemen 3 - Things'll Never Be the Same -> Codeine - Second Chance | base_T=0.8162504740633206 cur_T=0.6343020234150984 base_dev=0.3082726195809624 cur_dev=0.2741371512574009

- hash `7b5cce4454dff67c983760f119bf0a41c9e822d53207d3080499586666e322c0` label `arc_w1.5_t0.08_abs_m0.15_autoFalse_tb0.02_beam10` overlap=None
  - pos 1: David Bowie - Let's Dance -> David Bowie - Life On Mars? | base_T=None cur_T=None base_dev=None cur_dev=None
  - pos 2: Prince - Baby I'm a Star ("Take Me with U" 7" B-Side Edit) (Take Me with U 7" B-Side Edit; 2017 Remaster) -> teen suicide - My Little World | base_T=0.7877364949183092 cur_T=0.8061904896633129 base_dev=0.31086898616542963 cur_dev=0.3433774234597585
  - pos 3: Dead Kennedys - Viva Las Vegas (2022 Mix) -> Tom Waits - It's Over (Remastered) | base_T=0.9038338369336413 cur_T=0.7695236674405489 base_dev=0.3234365407861713 cur_dev=0.27705930792705674
  - pos 4: Big Black - Strange Things -> Wednesday - Carolina Murder Suicide | base_T=0.868413413419985 cur_T=0.8166042806934403 base_dev=0.13430912676290752 cur_dev=0.13534318234277787
  - pos 5: Nirvana - Swap Meet (Remastered) -> King Krule - The Cadet Leaps | base_T=0.8330160032210623 cur_T=0.7665084890929141 base_dev=0.05792099456554084 cur_dev=0.027197924918060112
  - pos 6: Minutemen - Corona -> Sigur Rós - Andrá | base_T=0.8609920669629056 cur_T=0.8160552428817576 base_dev=0.11121147782532603 cur_dev=0.07191639938006988
  - pos 7: Reatards - Your Old News Baby -> Codeine - Second Chance | base_T=0.8868485626989104 cur_T=0.6343020234150984 base_dev=0.20592478598467834 cur_dev=0.17720610789946512
  - pos 8: Talking Heads - Life During Wartime -> M83 - Another Wave from You | base_T=0.7747388414523939 cur_T=0.812172068147903 base_dev=0.32428595212644684 cur_dev=0.31354826082438414
  - pos 9: David Bowie - Suffragette City -> David Bowie - Warszawa | base_T=0.6329123608906382 cur_T=0.8865096925932843 base_dev=None cur_dev=None
  - pos 10: The Beatles - Long Tall Sally -> Cornelius - Surfing on Mind Wave, Pt. 2 | base_T=0.8438698874380759 cur_T=0.8034951146243114 base_dev=0.1837260675379918 cur_dev=0.2072080595593752
  - pos 11: The Fall - Just Step Sideways -> Microphones - Here With Summer | base_T=0.9020908500716743 cur_T=0.8398301230820251 base_dev=0.28779182893992095 cur_dev=0.0800313041791113
  - pos 12: The Beets - Friends Of Friends -> PaPerCuts - Mattress on the Floor | base_T=0.7690583079574286 cur_T=0.8967572984165522 base_dev=0.07103774067145574 cur_dev=0.05568463355117209
  - pos 13: Seapony - Dreaming -> Ariel Pink's Dark Side - Queen of the Virgins | base_T=0.5812891962062137 cur_T=0.8549094391475107 base_dev=0.15653896152558605 cur_dev=0.09525394820549138
  - pos 14: Ariel Pink - Off The Dome -> Bikini Kill - Rebel Girl | base_T=0.8816809087878533 cur_T=0.8221788176610882 base_dev=0.04484929428316553 cur_dev=0.22193373855142984
  - pos 15: 48 Chairs - You Were Never There -> Tyvek - Going Through My Things | base_T=0.6213349949035449 cur_T=0.9268529584952317 base_dev=0.05632035573627997 cur_dev=0.41020552771838203
  - pos 16: Sonic Youth - Dude Ranch Nurse -> David Bowie - Suffragette City | base_T=0.6897304399639835 cur_T=0.8524873868886713 base_dev=0.020540089255099758 cur_dev=None
  - pos 17: Helvetia - 3 Boys -> No Age - Flutter Freer | base_T=0.8019982511029264 cur_T=0.8874990957333945 base_dev=0.08813442003398564 cur_dev=0.25178264270438616
  - pos 18: Wilco - Please Be Patient with Me -> Nirvana - Swap Meet (Remastered) | base_T=0.5282689610509975 cur_T=0.8811234108617932 base_dev=0.01117756888181809 cur_dev=0.2538239063638261
  - pos 19: Sufjan Stevens - That Was the Worst Christmas Ever! -> Art Feynman - Early Signs of Rhythm | base_T=0.7544948044875062 cur_T=0.8878898426648157 base_dev=0.08593498488629436 cur_dev=0.1426792760465157
  - pos 20: Sam Wilkes - Own -> Dead Kennedys - Viva Las Vegas (2022 Mix) | base_T=0.5726484074044664 cur_T=0.9134879188754146 base_dev=0.2222045506634247 cur_dev=0.08114361717105467

- hash `7b5cce4454dff67c983760f119bf0a41c9e822d53207d3080499586666e322c0` label `arc_w1.5_t0.08_abs_m0.15_autoTrue_tb0.02_beam10` overlap=None
  - pos 1: David Bowie - Let's Dance -> David Bowie - Life On Mars? | base_T=None cur_T=None base_dev=None cur_dev=None
  - pos 2: Prince - Baby I'm a Star ("Take Me with U" 7" B-Side Edit) (Take Me with U 7" B-Side Edit; 2017 Remaster) -> teen suicide - My Little World | base_T=0.7877364949183092 cur_T=0.8061904896633129 base_dev=0.31086898616542963 cur_dev=0.3433774234597585
  - pos 3: Dead Kennedys - Viva Las Vegas (2022 Mix) -> Tom Waits - It's Over (Remastered) | base_T=0.9038338369336413 cur_T=0.7695236674405489 base_dev=0.3234365407861713 cur_dev=0.27705930792705674
  - pos 4: Big Black - Strange Things -> Wednesday - Carolina Murder Suicide | base_T=0.868413413419985 cur_T=0.8166042806934403 base_dev=0.13430912676290752 cur_dev=0.13534318234277787
  - pos 5: Nirvana - Swap Meet (Remastered) -> King Krule - The Cadet Leaps | base_T=0.8330160032210623 cur_T=0.7665084890929141 base_dev=0.05792099456554084 cur_dev=0.027197924918060112
  - pos 6: Minutemen - Corona -> Sigur Rós - Andrá | base_T=0.8609920669629056 cur_T=0.8160552428817576 base_dev=0.11121147782532603 cur_dev=0.07191639938006988
  - pos 7: Reatards - Your Old News Baby -> Codeine - Second Chance | base_T=0.8868485626989104 cur_T=0.6343020234150984 base_dev=0.20592478598467834 cur_dev=0.17720610789946512
  - pos 8: Talking Heads - Life During Wartime -> M83 - Another Wave from You | base_T=0.7747388414523939 cur_T=0.812172068147903 base_dev=0.32428595212644684 cur_dev=0.31354826082438414
  - pos 9: David Bowie - Suffragette City -> David Bowie - Warszawa | base_T=0.6329123608906382 cur_T=0.8865096925932843 base_dev=None cur_dev=None
  - pos 10: The Beatles - Long Tall Sally -> Cornelius - Surfing on Mind Wave, Pt. 2 | base_T=0.8438698874380759 cur_T=0.8034951146243114 base_dev=0.1837260675379918 cur_dev=0.2072080595593752
  - pos 11: The Fall - Just Step Sideways -> Microphones - Here With Summer | base_T=0.9020908500716743 cur_T=0.8398301230820251 base_dev=0.28779182893992095 cur_dev=0.0800313041791113
  - pos 12: The Beets - Friends Of Friends -> PaPerCuts - Mattress on the Floor | base_T=0.7690583079574286 cur_T=0.8967572984165522 base_dev=0.07103774067145574 cur_dev=0.05568463355117209
  - pos 13: Seapony - Dreaming -> Ariel Pink's Dark Side - Queen of the Virgins | base_T=0.5812891962062137 cur_T=0.8549094391475107 base_dev=0.15653896152558605 cur_dev=0.09525394820549138
  - pos 14: Ariel Pink - Off The Dome -> Bikini Kill - Rebel Girl | base_T=0.8816809087878533 cur_T=0.8221788176610882 base_dev=0.04484929428316553 cur_dev=0.22193373855142984
  - pos 15: 48 Chairs - You Were Never There -> Tyvek - Going Through My Things | base_T=0.6213349949035449 cur_T=0.9268529584952317 base_dev=0.05632035573627997 cur_dev=0.41020552771838203
  - pos 16: Sonic Youth - Dude Ranch Nurse -> David Bowie - Suffragette City | base_T=0.6897304399639835 cur_T=0.8524873868886713 base_dev=0.020540089255099758 cur_dev=None
  - pos 17: Helvetia - 3 Boys -> No Age - Flutter Freer | base_T=0.8019982511029264 cur_T=0.8874990957333945 base_dev=0.08813442003398564 cur_dev=0.25178264270438616
  - pos 18: Wilco - Please Be Patient with Me -> Nirvana - Swap Meet (Remastered) | base_T=0.5282689610509975 cur_T=0.8811234108617932 base_dev=0.01117756888181809 cur_dev=0.2538239063638261
  - pos 19: Sufjan Stevens - That Was the Worst Christmas Ever! -> Art Feynman - Early Signs of Rhythm | base_T=0.7544948044875062 cur_T=0.8878898426648157 base_dev=0.08593498488629436 cur_dev=0.1426792760465157
  - pos 20: Sam Wilkes - Own -> Dead Kennedys - Viva Las Vegas (2022 Mix) | base_T=0.5726484074044664 cur_T=0.9134879188754146 base_dev=0.2222045506634247 cur_dev=0.08114361717105467

- hash `1c532eeb42d9057faa12ff235d1be77b638ee49052ec762696d41ef3b74d5c65` label `arc_w1.5_t0.08_abs_m0.1_autoTrue_tbNone_beamNone` overlap=None
  - pos 6: Minutemen - Corona -> Reatards - Your Old News Baby | base_T=0.8609920669629056 cur_T=0.8339293121838041 base_dev=0.11121147782532603 cur_dev=0.043713111573949415
  - pos 7: Reatards - Your Old News Baby -> Tyvek - Mary Ellen Claims | base_T=0.8868485626989104 cur_T=0.8857081316422318 base_dev=0.20592478598467834 cur_dev=0.14419335755346485
  - pos 8: Talking Heads - Life During Wartime -> No Age - Flutter Freer | base_T=0.7747388414523939 cur_T=0.7996202067251209 base_dev=0.32428595212644684 cur_dev=0.26323797500881996
  - pos 10: The Beatles - Long Tall Sally -> Here We Go Magic - Tunnelvision | base_T=0.8438698874380759 cur_T=0.8576887312447541 base_dev=0.1837260675379918 cur_dev=0.22503054049837937
  - pos 11: The Fall - Just Step Sideways -> Guided By Voices - Mincer Ray | base_T=0.9020908500716743 cur_T=0.7963139839760061 base_dev=0.28779182893992095 cur_dev=0.2529436298268758
  - pos 12: The Beets - Friends Of Friends -> The Beatles - Long Tall Sally | base_T=0.7690583079574286 cur_T=0.8235979338072454 base_dev=0.07103774067145574 cur_dev=0.0871778526810949
  - pos 13: Seapony - Dreaming -> Ariel Pink - Off The Dome | base_T=0.5812891962062137 cur_T=0.5891595409843458 base_dev=0.15653896152558605 cur_dev=0.13965232565375324
  - pos 14: Ariel Pink - Off The Dome -> Seapony - Dreaming | base_T=0.8816809087878533 cur_T=0.9183315956054764 base_dev=0.04484929428316553 cur_dev=0.06173593015499834
  - pos 18: Wilco - Please Be Patient with Me -> Early Day Miners - The Way We Live Now | base_T=0.5282689610509975 cur_T=0.37890212046114924 base_dev=0.01117756888181809 cur_dev=0.0026778613017034436
  - pos 19: Sufjan Stevens - That Was the Worst Christmas Ever! -> Wilco - Please Be Patient with Me | base_T=0.7544948044875062 cur_T=0.7108870752574645 base_dev=0.08593498488629436 cur_dev=0.08362546248876956
  - pos 20: Sam Wilkes - Own -> Pavement - Old to Begin | base_T=0.5726484074044664 cur_T=0.7031226473189253 base_dev=0.2222045506634247 cur_dev=0.1909593421617155
  - pos 24: Godspeed You! Black Emperor - Peasantry or ‘Light! Inside of Light!’ -> Helvetia - Inverted | base_T=0.7247934731539941 cur_T=0.8006093138472479 base_dev=0.434280496159301 cur_dev=0.22134405671042223
  - pos 25: King Krule - The Cadet Leaps -> SPIRIT OF THE BEEHIVE - 1/500 | base_T=0.6955376759430785 cur_T=0.6241630687142188 base_dev=0.33894282584742685 cur_dev=0.22209661668943015
  - pos 26: Sigur Rós - Andrá -> Pavement - Transport Is Arranged | base_T=0.8160552428817576 cur_T=0.7170408931921906 base_dev=0.23068578378063215 cur_dev=0.10744419490987855
  - pos 27: Mount Eerie - Night Palace -> King Krule - The Cadet Leaps | base_T=0.5418085451639895 cur_T=0.5635191366436167 base_dev=0.04170750693184688 cur_dev=0.08406254206009711
  - pos 28: Codeine - Second Chance -> Sigur Rós - Andrá | base_T=0.824862215843409 cur_T=0.8160552428817576 base_dev=0.13539761823555807 cur_dev=0.19231958412689176
  - pos 29: Spacemen 3 - Things'll Never Be the Same -> Codeine - Second Chance | base_T=0.8162504740633206 cur_T=0.6343020234150984 base_dev=0.3082726195809624 cur_dev=0.2741371512574009

- hash `f295daa16cf91cb1fe3b799bf3a650b4537a7d8b1b56a54a90275b9b59d52e4c` label `arc_w1.5_t0.0_abs_m0.2_autoFalse_tb0.02_beamNone` overlap=None
  - pos 6: Minutemen - Corona -> Reatards - Your Old News Baby | base_T=0.8609920669629056 cur_T=0.8339293121838041 base_dev=0.11121147782532603 cur_dev=0.043713111573949415
  - pos 7: Reatards - Your Old News Baby -> Tyvek - Mary Ellen Claims | base_T=0.8868485626989104 cur_T=0.8857081316422318 base_dev=0.20592478598467834 cur_dev=0.14419335755346485
  - pos 8: Talking Heads - Life During Wartime -> No Age - Flutter Freer | base_T=0.7747388414523939 cur_T=0.7996202067251209 base_dev=0.32428595212644684 cur_dev=0.26323797500881996
  - pos 10: The Beatles - Long Tall Sally -> Here We Go Magic - Tunnelvision | base_T=0.8438698874380759 cur_T=0.8576887312447541 base_dev=0.1837260675379918 cur_dev=0.22503054049837937
  - pos 11: The Fall - Just Step Sideways -> Guided By Voices - Mincer Ray | base_T=0.9020908500716743 cur_T=0.7963139839760061 base_dev=0.28779182893992095 cur_dev=0.2529436298268758
  - pos 12: The Beets - Friends Of Friends -> The Beatles - Long Tall Sally | base_T=0.7690583079574286 cur_T=0.8235979338072454 base_dev=0.07103774067145574 cur_dev=0.0871778526810949
  - pos 18: Wilco - Please Be Patient with Me -> Early Day Miners - The Way We Live Now | base_T=0.5282689610509975 cur_T=0.37890212046114924 base_dev=0.01117756888181809 cur_dev=0.0026778613017034436
  - pos 19: Sufjan Stevens - That Was the Worst Christmas Ever! -> Wilco - Please Be Patient with Me | base_T=0.7544948044875062 cur_T=0.7108870752574645 base_dev=0.08593498488629436 cur_dev=0.08362546248876956
  - pos 20: Sam Wilkes - Own -> Sufjan Stevens - That Was the Worst Christmas Ever! | base_T=0.5726484074044664 cur_T=0.7544948044875062 base_dev=0.2222045506634247 cur_dev=0.16510582519094252
  - pos 24: Godspeed You! Black Emperor - Peasantry or ‘Light! Inside of Light!’ -> Helvetia - Inverted | base_T=0.7247934731539941 cur_T=0.8006093138472479 base_dev=0.434280496159301 cur_dev=0.22134405671042223
  - pos 25: King Krule - The Cadet Leaps -> SPIRIT OF THE BEEHIVE - 1/500 | base_T=0.6955376759430785 cur_T=0.6241630687142188 base_dev=0.33894282584742685 cur_dev=0.22209661668943015
  - pos 26: Sigur Rós - Andrá -> King Krule - The Cadet Leaps | base_T=0.8160552428817576 cur_T=0.5275679496756788 base_dev=0.23068578378063215 cur_dev=0.13845839189621728
  - pos 27: Mount Eerie - Night Palace -> Slowdive - No Longer Making Time | base_T=0.5418085451639895 cur_T=0.43563470693952566 base_dev=0.04170750693184688 cur_dev=0.02401060564296753
  - pos 28: Codeine - Second Chance -> Sigur Rós - Andrá | base_T=0.824862215843409 cur_T=0.39324738679093163 base_dev=0.13539761823555807 cur_dev=0.19231958412689176
  - pos 29: Spacemen 3 - Things'll Never Be the Same -> Codeine - Second Chance | base_T=0.8162504740633206 cur_T=0.6343020234150984 base_dev=0.3082726195809624 cur_dev=0.2741371512574009

- hash `e13ee6f2fba63164875f3735603b01b69e42b1dd07ac6257234cf44f1b4ba299` label `arc_w1.5_t0.12_abs_m0.15_autoFalse_tb0.02_beam10` overlap=None
  - pos 1: David Bowie - Let's Dance -> David Bowie - Life On Mars? | base_T=None cur_T=None base_dev=None cur_dev=None
  - pos 2: Prince - Baby I'm a Star ("Take Me with U" 7" B-Side Edit) (Take Me with U 7" B-Side Edit; 2017 Remaster) -> teen suicide - My Little World | base_T=0.7877364949183092 cur_T=0.8061904896633129 base_dev=0.31086898616542963 cur_dev=0.3433774234597585
  - pos 3: Dead Kennedys - Viva Las Vegas (2022 Mix) -> Tom Waits - It's Over (Remastered) | base_T=0.9038338369336413 cur_T=0.7695236674405489 base_dev=0.3234365407861713 cur_dev=0.27705930792705674
  - pos 4: Big Black - Strange Things -> Wednesday - Carolina Murder Suicide | base_T=0.868413413419985 cur_T=0.8166042806934403 base_dev=0.13430912676290752 cur_dev=0.13534318234277787
  - pos 5: Nirvana - Swap Meet (Remastered) -> King Krule - The Cadet Leaps | base_T=0.8330160032210623 cur_T=0.7665084890929141 base_dev=0.05792099456554084 cur_dev=0.027197924918060112
  - pos 6: Minutemen - Corona -> Sigur Rós - Andrá | base_T=0.8609920669629056 cur_T=0.8160552428817576 base_dev=0.11121147782532603 cur_dev=0.07191639938006988
  - pos 7: Reatards - Your Old News Baby -> Codeine - Second Chance | base_T=0.8868485626989104 cur_T=0.6343020234150984 base_dev=0.20592478598467834 cur_dev=0.17720610789946512
  - pos 8: Talking Heads - Life During Wartime -> M83 - Another Wave from You | base_T=0.7747388414523939 cur_T=0.812172068147903 base_dev=0.32428595212644684 cur_dev=0.31354826082438414
  - pos 9: David Bowie - Suffragette City -> David Bowie - Warszawa | base_T=0.6329123608906382 cur_T=0.8865096925932843 base_dev=None cur_dev=None
  - pos 10: The Beatles - Long Tall Sally -> Microphones - Here With Summer | base_T=0.8438698874380759 cur_T=0.6552326095259688 base_dev=0.1837260675379918 cur_dev=0.21877083720095408
  - pos 11: The Fall - Just Step Sideways -> PaPerCuts - Mattress on the Floor | base_T=0.9020908500716743 cur_T=0.8967572984165522 base_dev=0.28779182893992095 cur_dev=0.25616906750238166
  - pos 12: The Beets - Friends Of Friends -> Guided By Voices - Tractor Rape Chain | base_T=0.7690583079574286 cur_T=0.8780779985451096 base_dev=0.07103774067145574 cur_dev=0.04132543425308627
  - pos 13: Seapony - Dreaming -> Tyvek - Going Through My Things | base_T=0.5812891962062137 cur_T=0.8103966985680967 base_dev=0.15653896152558605 cur_dev=0.07098156074532969
  - pos 14: Ariel Pink - Off The Dome -> Television Personalities - David Hockney's Diary | base_T=0.8816809087878533 cur_T=0.8313127464409515 base_dev=0.04484929428316553 cur_dev=0.17267414574460205
  - pos 15: 48 Chairs - You Were Never There -> Bikini Kill - Rebel Girl | base_T=0.6213349949035449 cur_T=0.8553334675301828 base_dev=0.05632035573627997 cur_dev=0.36067327157327267
  - pos 16: Sonic Youth - Dude Ranch Nurse -> David Bowie - Suffragette City | base_T=0.6897304399639835 cur_T=0.8313625834223988 base_dev=0.020540089255099758 cur_dev=None
  - pos 17: Helvetia - 3 Boys -> No Age - Flutter Freer | base_T=0.8019982511029264 cur_T=0.8874990957333945 base_dev=0.08813442003398564 cur_dev=0.25178264270438616
  - pos 18: Wilco - Please Be Patient with Me -> Nirvana - Swap Meet (Remastered) | base_T=0.5282689610509975 cur_T=0.8811234108617932 base_dev=0.01117756888181809 cur_dev=0.2538239063638261
  - pos 19: Sufjan Stevens - That Was the Worst Christmas Ever! -> Art Feynman - Early Signs of Rhythm | base_T=0.7544948044875062 cur_T=0.8878898426648157 base_dev=0.08593498488629436 cur_dev=0.1426792760465157
  - pos 20: Sam Wilkes - Own -> Dead Kennedys - Viva Las Vegas (2022 Mix) | base_T=0.5726484074044664 cur_T=0.9134879188754146 base_dev=0.2222045506634247 cur_dev=0.08114361717105467

- hash `5c13d3b0cc747b007f33d51b41fc476289136f7a855f67464338648df93dfb8d` label `arc_w1.5_t0.2_abs_m0.1_autoTrue_tbNone_beam25` overlap=None
  - pos 1: David Bowie - Let's Dance -> David Bowie - Life On Mars? | base_T=None cur_T=None base_dev=None cur_dev=None
  - pos 2: Prince - Baby I'm a Star ("Take Me with U" 7" B-Side Edit) (Take Me with U 7" B-Side Edit; 2017 Remaster) -> teen suicide - My Little World | base_T=0.7877364949183092 cur_T=0.8061904896633129 base_dev=0.31086898616542963 cur_dev=0.3433774234597585
  - pos 3: Dead Kennedys - Viva Las Vegas (2022 Mix) -> Tom Waits - It's Over (Remastered) | base_T=0.9038338369336413 cur_T=0.7695236674405489 base_dev=0.3234365407861713 cur_dev=0.27705930792705674
  - pos 4: Big Black - Strange Things -> Wednesday - Carolina Murder Suicide | base_T=0.868413413419985 cur_T=0.8166042806934403 base_dev=0.13430912676290752 cur_dev=0.13534318234277787
  - pos 5: Nirvana - Swap Meet (Remastered) -> King Krule - The Cadet Leaps | base_T=0.8330160032210623 cur_T=0.7665084890929141 base_dev=0.05792099456554084 cur_dev=0.027197924918060112
  - pos 6: Minutemen - Corona -> Sigur Rós - Andrá | base_T=0.8609920669629056 cur_T=0.8160552428817576 base_dev=0.11121147782532603 cur_dev=0.07191639938006988
  - pos 7: Reatards - Your Old News Baby -> Codeine - Second Chance | base_T=0.8868485626989104 cur_T=0.6343020234150984 base_dev=0.20592478598467834 cur_dev=0.17720610789946512
  - pos 8: Talking Heads - Life During Wartime -> M83 - Another Wave from You | base_T=0.7747388414523939 cur_T=0.812172068147903 base_dev=0.32428595212644684 cur_dev=0.31354826082438414
  - pos 9: David Bowie - Suffragette City -> David Bowie - Warszawa | base_T=0.6329123608906382 cur_T=0.8865096925932843 base_dev=None cur_dev=None
  - pos 10: The Beatles - Long Tall Sally -> Microphones - Here With Summer | base_T=0.8438698874380759 cur_T=0.6552326095259688 base_dev=0.1837260675379918 cur_dev=0.21877083720095408
  - pos 11: The Fall - Just Step Sideways -> PaPerCuts - Mattress on the Floor | base_T=0.9020908500716743 cur_T=0.8967572984165522 base_dev=0.28779182893992095 cur_dev=0.25616906750238166
  - pos 12: The Beets - Friends Of Friends -> Guided By Voices - Tractor Rape Chain | base_T=0.7690583079574286 cur_T=0.8780779985451096 base_dev=0.07103774067145574 cur_dev=0.04132543425308627
  - pos 13: Seapony - Dreaming -> Ariel Pink's Dark Side - Queen of the Virgins | base_T=0.5812891962062137 cur_T=0.8572865593008591 base_dev=0.15653896152558605 cur_dev=0.09525394820549138
  - pos 14: Ariel Pink - Off The Dome -> Bikini Kill - Rebel Girl | base_T=0.8816809087878533 cur_T=0.8221788176610882 base_dev=0.04484929428316553 cur_dev=0.22193373855142984
  - pos 15: 48 Chairs - You Were Never There -> Tyvek - Going Through My Things | base_T=0.6213349949035449 cur_T=0.9268529584952317 base_dev=0.05632035573627997 cur_dev=0.41020552771838203
  - pos 16: Sonic Youth - Dude Ranch Nurse -> David Bowie - Suffragette City | base_T=0.6897304399639835 cur_T=0.8524873868886713 base_dev=0.020540089255099758 cur_dev=None
  - pos 17: Helvetia - 3 Boys -> No Age - Flutter Freer | base_T=0.8019982511029264 cur_T=0.8874990957333945 base_dev=0.08813442003398564 cur_dev=0.25178264270438616
  - pos 18: Wilco - Please Be Patient with Me -> Nirvana - Swap Meet (Remastered) | base_T=0.5282689610509975 cur_T=0.8811234108617932 base_dev=0.01117756888181809 cur_dev=0.2538239063638261
  - pos 19: Sufjan Stevens - That Was the Worst Christmas Ever! -> Dead Kennedys - Viva Las Vegas (2022 Mix) | base_T=0.7544948044875062 cur_T=0.8529024165312802 base_dev=0.08593498488629436 cur_dev=0.14137731678525972
  - pos 20: Sam Wilkes - Own -> Big Black - Strange Things | base_T=0.5726484074044664 cur_T=0.868413413419985 base_dev=0.2222045506634247 cur_dev=0.054227877558519655

- hash `dfe0fba7fe7d584b18c57bb58afb24f81cf014cf5fb5d50420106e25e9ffb973` label `arc_w2.0_t0.05_abs_m0.1_autoFalse_tb0.02_beam50` overlap=None
  - pos 1: David Bowie - Let's Dance -> David Bowie - Life On Mars? | base_T=None cur_T=None base_dev=None cur_dev=None
  - pos 2: Prince - Baby I'm a Star ("Take Me with U" 7" B-Side Edit) (Take Me with U 7" B-Side Edit; 2017 Remaster) -> teen suicide - My Little World | base_T=0.7877364949183092 cur_T=0.8061904896633129 base_dev=0.31086898616542963 cur_dev=0.3433774234597585
  - pos 3: Dead Kennedys - Viva Las Vegas (2022 Mix) -> Tom Waits - It's Over (Remastered) | base_T=0.9038338369336413 cur_T=0.7695236674405489 base_dev=0.3234365407861713 cur_dev=0.27705930792705674
  - pos 4: Big Black - Strange Things -> Wednesday - Carolina Murder Suicide | base_T=0.868413413419985 cur_T=0.8166042806934403 base_dev=0.13430912676290752 cur_dev=0.13534318234277787
  - pos 5: Nirvana - Swap Meet (Remastered) -> King Krule - The Cadet Leaps | base_T=0.8330160032210623 cur_T=0.7665084890929141 base_dev=0.05792099456554084 cur_dev=0.027197924918060112
  - pos 6: Minutemen - Corona -> Sigur Rós - Andrá | base_T=0.8609920669629056 cur_T=0.8160552428817576 base_dev=0.11121147782532603 cur_dev=0.07191639938006988
  - pos 7: Reatards - Your Old News Baby -> Codeine - Second Chance | base_T=0.8868485626989104 cur_T=0.6343020234150984 base_dev=0.20592478598467834 cur_dev=0.17720610789946512
  - pos 8: Talking Heads - Life During Wartime -> M83 - Another Wave from You | base_T=0.7747388414523939 cur_T=0.812172068147903 base_dev=0.32428595212644684 cur_dev=0.31354826082438414
  - pos 9: David Bowie - Suffragette City -> David Bowie - Warszawa | base_T=0.6329123608906382 cur_T=0.8865096925932843 base_dev=None cur_dev=None
  - pos 10: The Beatles - Long Tall Sally -> Cornelius - Surfing on Mind Wave, Pt. 2 | base_T=0.8438698874380759 cur_T=0.8034951146243114 base_dev=0.1837260675379918 cur_dev=0.2072080595593752
  - pos 11: The Fall - Just Step Sideways -> Microphones - Here With Summer | base_T=0.9020908500716743 cur_T=0.8398301230820251 base_dev=0.28779182893992095 cur_dev=0.0800313041791113
  - pos 12: The Beets - Friends Of Friends -> PaPerCuts - Mattress on the Floor | base_T=0.7690583079574286 cur_T=0.8967572984165522 base_dev=0.07103774067145574 cur_dev=0.05568463355117209
  - pos 13: Seapony - Dreaming -> Ariel Pink's Dark Side - Queen of the Virgins | base_T=0.5812891962062137 cur_T=0.8549094391475107 base_dev=0.15653896152558605 cur_dev=0.09525394820549138
  - pos 14: Ariel Pink - Off The Dome -> Bikini Kill - Rebel Girl | base_T=0.8816809087878533 cur_T=0.8221788176610882 base_dev=0.04484929428316553 cur_dev=0.22193373855142984
  - pos 15: 48 Chairs - You Were Never There -> Tyvek - Going Through My Things | base_T=0.6213349949035449 cur_T=0.9268529584952317 base_dev=0.05632035573627997 cur_dev=0.41020552771838203
  - pos 16: Sonic Youth - Dude Ranch Nurse -> David Bowie - Suffragette City | base_T=0.6897304399639835 cur_T=0.8524873868886713 base_dev=0.020540089255099758 cur_dev=None
  - pos 17: Helvetia - 3 Boys -> No Age - Flutter Freer | base_T=0.8019982511029264 cur_T=0.8874990957333945 base_dev=0.08813442003398564 cur_dev=0.25178264270438616
  - pos 18: Wilco - Please Be Patient with Me -> Reatards - Your Old News Baby | base_T=0.5282689610509975 cur_T=0.8442933812141848 base_dev=0.01117756888181809 cur_dev=0.16411629632077135
  - pos 19: Sufjan Stevens - That Was the Worst Christmas Ever! -> Nirvana - Swap Meet (Remastered) | base_T=0.7544948044875062 cur_T=0.8565112579200121 base_dev=0.08593498488629436 cur_dev=0.05333947241261655
  - pos 20: Sam Wilkes - Own -> Art Feynman - Early Signs of Rhythm | base_T=0.5726484074044664 cur_T=0.8878898426648157 base_dev=0.2222045506634247 cur_dev=0.07984165790979869

- hash `f9967a4cf7d1a8428cb494920839cda996a3c21b36dec56c3f6f5daef3c926cd` label `arc_w2.0_t0.05_abs_m0.1_autoTrue_tbNone_beamNone` overlap=None
  - pos 6: Minutemen - Corona -> Reatards - Your Old News Baby | base_T=0.8609920669629056 cur_T=0.8339293121838041 base_dev=0.11121147782532603 cur_dev=0.043713111573949415
  - pos 7: Reatards - Your Old News Baby -> Tyvek - Mary Ellen Claims | base_T=0.8868485626989104 cur_T=0.8857081316422318 base_dev=0.20592478598467834 cur_dev=0.14419335755346485
  - pos 8: Talking Heads - Life During Wartime -> No Age - Flutter Freer | base_T=0.7747388414523939 cur_T=0.7996202067251209 base_dev=0.32428595212644684 cur_dev=0.26323797500881996
  - pos 10: The Beatles - Long Tall Sally -> Here We Go Magic - Tunnelvision | base_T=0.8438698874380759 cur_T=0.8576887312447541 base_dev=0.1837260675379918 cur_dev=0.22503054049837937
  - pos 11: The Fall - Just Step Sideways -> Guided By Voices - Mincer Ray | base_T=0.9020908500716743 cur_T=0.7963139839760061 base_dev=0.28779182893992095 cur_dev=0.2529436298268758
  - pos 12: The Beets - Friends Of Friends -> The Beatles - Long Tall Sally | base_T=0.7690583079574286 cur_T=0.8235979338072454 base_dev=0.07103774067145574 cur_dev=0.0871778526810949
  - pos 13: Seapony - Dreaming -> Ariel Pink - Off The Dome | base_T=0.5812891962062137 cur_T=0.5891595409843458 base_dev=0.15653896152558605 cur_dev=0.13965232565375324
  - pos 14: Ariel Pink - Off The Dome -> Seapony - Dreaming | base_T=0.8816809087878533 cur_T=0.9183315956054764 base_dev=0.04484929428316553 cur_dev=0.06173593015499834
  - pos 17: Helvetia - 3 Boys -> Ativin - Thirteen Ovens | base_T=0.8019982511029264 cur_T=0.7405545459159097 base_dev=0.08813442003398564 cur_dev=0.04736172950407436
  - pos 18: Wilco - Please Be Patient with Me -> Helvetia - 3 Boys | base_T=0.5282689610509975 cur_T=0.48804649745684164 base_dev=0.01117756888181809 cur_dev=0.1938158226146075
  - pos 19: Sufjan Stevens - That Was the Worst Christmas Ever! -> Wilco - Please Be Patient with Me | base_T=0.7544948044875062 cur_T=0.5282689610509975 base_dev=0.08593498488629436 cur_dev=0.08362546248876956
  - pos 20: Sam Wilkes - Own -> Pavement - Old to Begin | base_T=0.5726484074044664 cur_T=0.7031226473189253 base_dev=0.2222045506634247 cur_dev=0.1909593421617155
  - pos 24: Godspeed You! Black Emperor - Peasantry or ‘Light! Inside of Light!’ -> Helvetia - Inverted | base_T=0.7247934731539941 cur_T=0.8006093138472479 base_dev=0.434280496159301 cur_dev=0.22134405671042223
  - pos 25: King Krule - The Cadet Leaps -> SPIRIT OF THE BEEHIVE - 1/500 | base_T=0.6955376759430785 cur_T=0.6241630687142188 base_dev=0.33894282584742685 cur_dev=0.22209661668943015
  - pos 26: Sigur Rós - Andrá -> Godspeed You! Black Emperor - Peasantry or ‘Light! Inside of Light!’ | base_T=0.8160552428817576 cur_T=0.4922310530022715 base_dev=0.23068578378063215 cur_dev=0.09505652918624863
  - pos 27: Mount Eerie - Night Palace -> King Krule - The Cadet Leaps | base_T=0.5418085451639895 cur_T=0.6955376759430785 base_dev=0.04170750693184688 cur_dev=0.08406254206009711
  - pos 28: Codeine - Second Chance -> Sigur Rós - Andrá | base_T=0.824862215843409 cur_T=0.8160552428817576 base_dev=0.13539761823555807 cur_dev=0.19231958412689176
  - pos 29: Spacemen 3 - Things'll Never Be the Same -> Codeine - Second Chance | base_T=0.8162504740633206 cur_T=0.6343020234150984 base_dev=0.3082726195809624 cur_dev=0.2741371512574009

- hash `f9967a4cf7d1a8428cb494920839cda996a3c21b36dec56c3f6f5daef3c926cd` label `arc_w2.0_t0.08_abs_m0.1_autoFalse_tbNone_beamNone` overlap=None
  - pos 6: Minutemen - Corona -> Reatards - Your Old News Baby | base_T=0.8609920669629056 cur_T=0.8339293121838041 base_dev=0.11121147782532603 cur_dev=0.043713111573949415
  - pos 7: Reatards - Your Old News Baby -> Tyvek - Mary Ellen Claims | base_T=0.8868485626989104 cur_T=0.8857081316422318 base_dev=0.20592478598467834 cur_dev=0.14419335755346485
  - pos 8: Talking Heads - Life During Wartime -> No Age - Flutter Freer | base_T=0.7747388414523939 cur_T=0.7996202067251209 base_dev=0.32428595212644684 cur_dev=0.26323797500881996
  - pos 10: The Beatles - Long Tall Sally -> Here We Go Magic - Tunnelvision | base_T=0.8438698874380759 cur_T=0.8576887312447541 base_dev=0.1837260675379918 cur_dev=0.22503054049837937
  - pos 11: The Fall - Just Step Sideways -> Guided By Voices - Mincer Ray | base_T=0.9020908500716743 cur_T=0.7963139839760061 base_dev=0.28779182893992095 cur_dev=0.2529436298268758
  - pos 12: The Beets - Friends Of Friends -> The Beatles - Long Tall Sally | base_T=0.7690583079574286 cur_T=0.8235979338072454 base_dev=0.07103774067145574 cur_dev=0.0871778526810949
  - pos 13: Seapony - Dreaming -> Ariel Pink - Off The Dome | base_T=0.5812891962062137 cur_T=0.5891595409843458 base_dev=0.15653896152558605 cur_dev=0.13965232565375324
  - pos 14: Ariel Pink - Off The Dome -> Seapony - Dreaming | base_T=0.8816809087878533 cur_T=0.9183315956054764 base_dev=0.04484929428316553 cur_dev=0.06173593015499834
  - pos 17: Helvetia - 3 Boys -> Ativin - Thirteen Ovens | base_T=0.8019982511029264 cur_T=0.7405545459159097 base_dev=0.08813442003398564 cur_dev=0.04736172950407436
  - pos 18: Wilco - Please Be Patient with Me -> Helvetia - 3 Boys | base_T=0.5282689610509975 cur_T=0.48804649745684164 base_dev=0.01117756888181809 cur_dev=0.1938158226146075
  - pos 19: Sufjan Stevens - That Was the Worst Christmas Ever! -> Wilco - Please Be Patient with Me | base_T=0.7544948044875062 cur_T=0.5282689610509975 base_dev=0.08593498488629436 cur_dev=0.08362546248876956
  - pos 20: Sam Wilkes - Own -> Pavement - Old to Begin | base_T=0.5726484074044664 cur_T=0.7031226473189253 base_dev=0.2222045506634247 cur_dev=0.1909593421617155
  - pos 24: Godspeed You! Black Emperor - Peasantry or ‘Light! Inside of Light!’ -> Helvetia - Inverted | base_T=0.7247934731539941 cur_T=0.8006093138472479 base_dev=0.434280496159301 cur_dev=0.22134405671042223
  - pos 25: King Krule - The Cadet Leaps -> SPIRIT OF THE BEEHIVE - 1/500 | base_T=0.6955376759430785 cur_T=0.6241630687142188 base_dev=0.33894282584742685 cur_dev=0.22209661668943015
  - pos 26: Sigur Rós - Andrá -> Godspeed You! Black Emperor - Peasantry or ‘Light! Inside of Light!’ | base_T=0.8160552428817576 cur_T=0.4922310530022715 base_dev=0.23068578378063215 cur_dev=0.09505652918624863
  - pos 27: Mount Eerie - Night Palace -> King Krule - The Cadet Leaps | base_T=0.5418085451639895 cur_T=0.6955376759430785 base_dev=0.04170750693184688 cur_dev=0.08406254206009711
  - pos 28: Codeine - Second Chance -> Sigur Rós - Andrá | base_T=0.824862215843409 cur_T=0.8160552428817576 base_dev=0.13539761823555807 cur_dev=0.19231958412689176
  - pos 29: Spacemen 3 - Things'll Never Be the Same -> Codeine - Second Chance | base_T=0.8162504740633206 cur_T=0.6343020234150984 base_dev=0.3082726195809624 cur_dev=0.2741371512574009

- hash `0536ca9d531004055fa13454ad8590fd2d67b5e5dd961c4dcd8b50c7545eaa03` label `arc_w2.0_t0.08_abs_m0.4_autoFalse_tbNone_beam50` overlap=None
  - pos 1: David Bowie - Let's Dance -> David Bowie - Life On Mars? | base_T=None cur_T=None base_dev=None cur_dev=None
  - pos 2: Prince - Baby I'm a Star ("Take Me with U" 7" B-Side Edit) (Take Me with U 7" B-Side Edit; 2017 Remaster) -> Wednesday - Carolina Murder Suicide | base_T=0.7877364949183092 cur_T=0.853585374207696 base_dev=0.31086898616542963 cur_dev=0.4059412324158763
  - pos 3: Dead Kennedys - Viva Las Vegas (2022 Mix) -> Youth Lagoon - Ghost To Me (Bonus Track) | base_T=0.9038338369336413 cur_T=0.8247360215272048 base_dev=0.3234365407861713 cur_dev=0.2928009828382454
  - pos 4: Big Black - Strange Things -> Dressed Up Animals - Mondtanz | base_T=0.868413413419985 cur_T=0.7822777019747553 base_dev=0.13430912676290752 cur_dev=0.09885559488118478
  - pos 5: Nirvana - Swap Meet (Remastered) -> King Krule - The Cadet Leaps | base_T=0.8330160032210623 cur_T=0.8067052528326448 base_dev=0.05792099456554084 cur_dev=0.027197924918060112
  - pos 6: Minutemen - Corona -> Sigur Rós - Andrá | base_T=0.8609920669629056 cur_T=0.8160552428817576 base_dev=0.11121147782532603 cur_dev=0.07191639938006988
  - pos 7: Reatards - Your Old News Baby -> Codeine - Second Chance | base_T=0.8868485626989104 cur_T=0.6343020234150984 base_dev=0.20592478598467834 cur_dev=0.17720610789946512
  - pos 8: Talking Heads - Life During Wartime -> M83 - Another Wave from You | base_T=0.7747388414523939 cur_T=0.812172068147903 base_dev=0.32428595212644684 cur_dev=0.31354826082438414
  - pos 9: David Bowie - Suffragette City -> David Bowie - Warszawa | base_T=0.6329123608906382 cur_T=0.8865096925932843 base_dev=None cur_dev=None
  - pos 10: The Beatles - Long Tall Sally -> Cornelius - Surfing on Mind Wave, Pt. 2 | base_T=0.8438698874380759 cur_T=0.8034951146243114 base_dev=0.1837260675379918 cur_dev=0.2072080595593752
  - pos 11: The Fall - Just Step Sideways -> Microphones - Here With Summer | base_T=0.9020908500716743 cur_T=0.8398301230820251 base_dev=0.28779182893992095 cur_dev=0.0800313041791113
  - pos 12: The Beets - Friends Of Friends -> PaPerCuts - Mattress on the Floor | base_T=0.7690583079574286 cur_T=0.8967572984165522 base_dev=0.07103774067145574 cur_dev=0.05568463355117209
  - pos 13: Seapony - Dreaming -> Television Personalities - David Hockney's Diary | base_T=0.5812891962062137 cur_T=0.8199087788846247 base_dev=0.15653896152558605 cur_dev=0.02781028820660747
  - pos 14: Ariel Pink - Off The Dome -> Bikini Kill - Rebel Girl | base_T=0.8816809087878533 cur_T=0.8553334675301828 base_dev=0.04484929428316553 cur_dev=0.22193373855142984
  - pos 15: 48 Chairs - You Were Never There -> Tyvek - Going Through My Things | base_T=0.6213349949035449 cur_T=0.9268529584952317 base_dev=0.05632035573627997 cur_dev=0.41020552771838203
  - pos 16: Sonic Youth - Dude Ranch Nurse -> David Bowie - Suffragette City | base_T=0.6897304399639835 cur_T=0.8524873868886713 base_dev=0.020540089255099758 cur_dev=None
  - pos 17: Helvetia - 3 Boys -> No Age - Flutter Freer | base_T=0.8019982511029264 cur_T=0.8874990957333945 base_dev=0.08813442003398564 cur_dev=0.25178264270438616
  - pos 18: Wilco - Please Be Patient with Me -> Reatards - Your Old News Baby | base_T=0.5282689610509975 cur_T=0.8442933812141848 base_dev=0.01117756888181809 cur_dev=0.16411629632077135
  - pos 19: Sufjan Stevens - That Was the Worst Christmas Ever! -> Nirvana - Swap Meet (Remastered) | base_T=0.7544948044875062 cur_T=0.8565112579200121 base_dev=0.08593498488629436 cur_dev=0.05333947241261655
  - pos 20: Sam Wilkes - Own -> Art Feynman - Early Signs of Rhythm | base_T=0.5726484074044664 cur_T=0.8878898426648157 base_dev=0.2222045506634247 cur_dev=0.07984165790979869

- hash `f890784b8f217153d5be96ba113e89b1fa07f01f12512b5e9e1919dd588a5f2f` label `arc_w2.0_t0.0_abs_m0.15_autoTrue_tbNone_beamNone` overlap=None
  - pos 6: Minutemen - Corona -> Reatards - Your Old News Baby | base_T=0.8609920669629056 cur_T=0.8339293121838041 base_dev=0.11121147782532603 cur_dev=0.043713111573949415
  - pos 7: Reatards - Your Old News Baby -> Tyvek - Mary Ellen Claims | base_T=0.8868485626989104 cur_T=0.8857081316422318 base_dev=0.20592478598467834 cur_dev=0.14419335755346485
  - pos 8: Talking Heads - Life During Wartime -> No Age - Flutter Freer | base_T=0.7747388414523939 cur_T=0.7996202067251209 base_dev=0.32428595212644684 cur_dev=0.26323797500881996
  - pos 10: The Beatles - Long Tall Sally -> Here We Go Magic - Tunnelvision | base_T=0.8438698874380759 cur_T=0.8576887312447541 base_dev=0.1837260675379918 cur_dev=0.22503054049837937
  - pos 11: The Fall - Just Step Sideways -> Guided By Voices - Mincer Ray | base_T=0.9020908500716743 cur_T=0.7963139839760061 base_dev=0.28779182893992095 cur_dev=0.2529436298268758
  - pos 12: The Beets - Friends Of Friends -> The Beatles - Long Tall Sally | base_T=0.7690583079574286 cur_T=0.8235979338072454 base_dev=0.07103774067145574 cur_dev=0.0871778526810949
  - pos 13: Seapony - Dreaming -> Dirty Beaches - Night Walk | base_T=0.5812891962062137 cur_T=0.7381574811163805 base_dev=0.15653896152558605 cur_dev=0.08554442259566503
  - pos 14: Ariel Pink - Off The Dome -> Seapony - Dreaming | base_T=0.8816809087878533 cur_T=0.7846687546599537 base_dev=0.04484929428316553 cur_dev=0.06173593015499834
  - pos 18: Wilco - Please Be Patient with Me -> Ativin - Thirteen Ovens | base_T=0.5282689610509975 cur_T=0.7416631148580815 base_dev=0.01117756888181809 cur_dev=0.05831967307654751
  - pos 19: Sufjan Stevens - That Was the Worst Christmas Ever! -> Wilco - Please Be Patient with Me | base_T=0.7544948044875062 cur_T=0.7378405154336215 base_dev=0.08593498488629436 cur_dev=0.08362546248876956
  - pos 20: Sam Wilkes - Own -> Pavement - Old to Begin | base_T=0.5726484074044664 cur_T=0.7031226473189253 base_dev=0.2222045506634247 cur_dev=0.1909593421617155
  - pos 24: Godspeed You! Black Emperor - Peasantry or ‘Light! Inside of Light!’ -> Helvetia - Inverted | base_T=0.7247934731539941 cur_T=0.8006093138472479 base_dev=0.434280496159301 cur_dev=0.22134405671042223
  - pos 25: King Krule - The Cadet Leaps -> SPIRIT OF THE BEEHIVE - 1/500 | base_T=0.6955376759430785 cur_T=0.6241630687142188 base_dev=0.33894282584742685 cur_dev=0.22209661668943015
  - pos 26: Sigur Rós - Andrá -> King Krule - The Cadet Leaps | base_T=0.8160552428817576 cur_T=0.5275679496756788 base_dev=0.23068578378063215 cur_dev=0.13845839189621728
  - pos 27: Mount Eerie - Night Palace -> Sigur Rós - Andrá | base_T=0.5418085451639895 cur_T=0.8160552428817576 base_dev=0.04170750693184688 cur_dev=0.008164849824317755
  - pos 28: Codeine - Second Chance -> Foxes in Fiction - Insomnia Keys | base_T=0.824862215843409 cur_T=0.7853552801920686 base_dev=0.13539761823555807 cur_dev=0.1430116425917467
  - pos 29: Spacemen 3 - Things'll Never Be the Same -> Codeine - Second Chance | base_T=0.8162504740633206 cur_T=0.592862442194348 base_dev=0.3082726195809624 cur_dev=0.2741371512574009

- hash `9d4b48480d26318798eb5efa7523318e04b2136b386edaedab9d49c432561d40` label `arc_w2.0_t0.0_abs_m0.3_autoTrue_tbNone_beamNone` overlap=None
  - pos 2: Prince - Baby I'm a Star ("Take Me with U" 7" B-Side Edit) (Take Me with U 7" B-Side Edit; 2017 Remaster) -> The Jimi Hendrix Experience - Voodoo Child (Slight Return) | base_T=0.7877364949183092 cur_T=0.5193118098399735 base_dev=0.31086898616542963 cur_dev=0.2994158246352174
  - pos 3: Dead Kennedys - Viva Las Vegas (2022 Mix) -> Prince - Baby I'm a Star ("Take Me with U" 7" B-Side Edit) (Take Me with U 7" B-Side Edit; 2017 Remaster) | base_T=0.9038338369336413 cur_T=0.8451825380099117 base_dev=0.3234365407861713 cur_dev=0.20248261050306005
  - pos 5: Nirvana - Swap Meet (Remastered) -> Dead Kennedys - Viva Las Vegas (2022 Mix) | base_T=0.8330160032210623 cur_T=0.7868589209451524 base_dev=0.05792099456554084 cur_dev=0.03011684980710244
  - pos 6: Minutemen - Corona -> Reatards - Your Old News Baby | base_T=0.8609920669629056 cur_T=0.8460453500498276 base_dev=0.11121147782532603 cur_dev=0.043713111573949415
  - pos 7: Reatards - Your Old News Baby -> Tyvek - Mary Ellen Claims | base_T=0.8868485626989104 cur_T=0.8857081316422318 base_dev=0.20592478598467834 cur_dev=0.14419335755346485
  - pos 8: Talking Heads - Life During Wartime -> No Age - Flutter Freer | base_T=0.7747388414523939 cur_T=0.7996202067251209 base_dev=0.32428595212644684 cur_dev=0.26323797500881996
  - pos 10: The Beatles - Long Tall Sally -> Here We Go Magic - Tunnelvision | base_T=0.8438698874380759 cur_T=0.8576887312447541 base_dev=0.1837260675379918 cur_dev=0.22503054049837937
  - pos 11: The Fall - Just Step Sideways -> Guided By Voices - Mincer Ray | base_T=0.9020908500716743 cur_T=0.7963139839760061 base_dev=0.28779182893992095 cur_dev=0.2529436298268758
  - pos 12: The Beets - Friends Of Friends -> The Beatles - Long Tall Sally | base_T=0.7690583079574286 cur_T=0.8235979338072454 base_dev=0.07103774067145574 cur_dev=0.0871778526810949
  - pos 13: Seapony - Dreaming -> Dirty Beaches - Night Walk | base_T=0.5812891962062137 cur_T=0.7381574811163805 base_dev=0.15653896152558605 cur_dev=0.08554442259566503
  - pos 14: Ariel Pink - Off The Dome -> Seapony - Dreaming | base_T=0.8816809087878533 cur_T=0.7846687546599537 base_dev=0.04484929428316553 cur_dev=0.06173593015499834
  - pos 18: Wilco - Please Be Patient with Me -> Ativin - Thirteen Ovens | base_T=0.5282689610509975 cur_T=0.7416631148580815 base_dev=0.01117756888181809 cur_dev=0.05831967307654751
  - pos 19: Sufjan Stevens - That Was the Worst Christmas Ever! -> Wilco - Please Be Patient with Me | base_T=0.7544948044875062 cur_T=0.7378405154336215 base_dev=0.08593498488629436 cur_dev=0.08362546248876956
  - pos 20: Sam Wilkes - Own -> Pavement - Old to Begin | base_T=0.5726484074044664 cur_T=0.7031226473189253 base_dev=0.2222045506634247 cur_dev=0.1909593421617155
  - pos 24: Godspeed You! Black Emperor - Peasantry or ‘Light! Inside of Light!’ -> Helvetia - Inverted | base_T=0.7247934731539941 cur_T=0.8006093138472479 base_dev=0.434280496159301 cur_dev=0.22134405671042223
  - pos 25: King Krule - The Cadet Leaps -> SPIRIT OF THE BEEHIVE - 1/500 | base_T=0.6955376759430785 cur_T=0.6241630687142188 base_dev=0.33894282584742685 cur_dev=0.22209661668943015
  - pos 26: Sigur Rós - Andrá -> King Krule - The Cadet Leaps | base_T=0.8160552428817576 cur_T=0.5275679496756788 base_dev=0.23068578378063215 cur_dev=0.13845839189621728
  - pos 27: Mount Eerie - Night Palace -> Sigur Rós - Andrá | base_T=0.5418085451639895 cur_T=0.8160552428817576 base_dev=0.04170750693184688 cur_dev=0.008164849824317755
  - pos 28: Codeine - Second Chance -> Foxes in Fiction - Insomnia Keys | base_T=0.824862215843409 cur_T=0.7853552801920686 base_dev=0.13539761823555807 cur_dev=0.1430116425917467
  - pos 29: Spacemen 3 - Things'll Never Be the Same -> Codeine - Second Chance | base_T=0.8162504740633206 cur_T=0.592862442194348 base_dev=0.3082726195809624 cur_dev=0.2741371512574009

- hash `812d1b12113dfa226c13570b2fd15c44c129ed166d773f33d875b63d8993e0de` label `arc_w2.0_t0.0_abs_m0.4_autoTrue_tb0.02_beam50` overlap=None
  - pos 1: David Bowie - Let's Dance -> David Bowie - Life On Mars? | base_T=None cur_T=None base_dev=None cur_dev=None
  - pos 2: Prince - Baby I'm a Star ("Take Me with U" 7" B-Side Edit) (Take Me with U 7" B-Side Edit; 2017 Remaster) -> Wednesday - Carolina Murder Suicide | base_T=0.7877364949183092 cur_T=0.853585374207696 base_dev=0.31086898616542963 cur_dev=0.4059412324158763
  - pos 3: Dead Kennedys - Viva Las Vegas (2022 Mix) -> Youth Lagoon - Ghost To Me (Bonus Track) | base_T=0.9038338369336413 cur_T=0.8247360215272048 base_dev=0.3234365407861713 cur_dev=0.2928009828382454
  - pos 4: Big Black - Strange Things -> Dressed Up Animals - Mondtanz | base_T=0.868413413419985 cur_T=0.7822777019747553 base_dev=0.13430912676290752 cur_dev=0.09885559488118478
  - pos 5: Nirvana - Swap Meet (Remastered) -> King Krule - The Cadet Leaps | base_T=0.8330160032210623 cur_T=0.8067052528326448 base_dev=0.05792099456554084 cur_dev=0.027197924918060112
  - pos 6: Minutemen - Corona -> Sigur Rós - Andrá | base_T=0.8609920669629056 cur_T=0.8160552428817576 base_dev=0.11121147782532603 cur_dev=0.07191639938006988
  - pos 7: Reatards - Your Old News Baby -> Codeine - Second Chance | base_T=0.8868485626989104 cur_T=0.6343020234150984 base_dev=0.20592478598467834 cur_dev=0.17720610789946512
  - pos 8: Talking Heads - Life During Wartime -> M83 - Another Wave from You | base_T=0.7747388414523939 cur_T=0.812172068147903 base_dev=0.32428595212644684 cur_dev=0.31354826082438414
  - pos 9: David Bowie - Suffragette City -> David Bowie - Warszawa | base_T=0.6329123608906382 cur_T=0.8865096925932843 base_dev=None cur_dev=None
  - pos 10: The Beatles - Long Tall Sally -> Cornelius - Surfing on Mind Wave, Pt. 2 | base_T=0.8438698874380759 cur_T=0.8034951146243114 base_dev=0.1837260675379918 cur_dev=0.2072080595593752
  - pos 11: The Fall - Just Step Sideways -> Microphones - Here With Summer | base_T=0.9020908500716743 cur_T=0.8398301230820251 base_dev=0.28779182893992095 cur_dev=0.0800313041791113
  - pos 12: The Beets - Friends Of Friends -> PaPerCuts - Mattress on the Floor | base_T=0.7690583079574286 cur_T=0.8967572984165522 base_dev=0.07103774067145574 cur_dev=0.05568463355117209
  - pos 13: Seapony - Dreaming -> Television Personalities - David Hockney's Diary | base_T=0.5812891962062137 cur_T=0.8199087788846247 base_dev=0.15653896152558605 cur_dev=0.02781028820660747
  - pos 14: Ariel Pink - Off The Dome -> Bikini Kill - Rebel Girl | base_T=0.8816809087878533 cur_T=0.8553334675301828 base_dev=0.04484929428316553 cur_dev=0.22193373855142984
  - pos 15: 48 Chairs - You Were Never There -> Tyvek - Going Through My Things | base_T=0.6213349949035449 cur_T=0.9268529584952317 base_dev=0.05632035573627997 cur_dev=0.41020552771838203
  - pos 16: Sonic Youth - Dude Ranch Nurse -> David Bowie - Suffragette City | base_T=0.6897304399639835 cur_T=0.8524873868886713 base_dev=0.020540089255099758 cur_dev=None
  - pos 17: Helvetia - 3 Boys -> No Age - Flutter Freer | base_T=0.8019982511029264 cur_T=0.8874990957333945 base_dev=0.08813442003398564 cur_dev=0.25178264270438616
  - pos 18: Wilco - Please Be Patient with Me -> Reatards - Your Old News Baby | base_T=0.5282689610509975 cur_T=0.8442933812141848 base_dev=0.01117756888181809 cur_dev=0.16411629632077135
  - pos 19: Sufjan Stevens - That Was the Worst Christmas Ever! -> Minutemen - Corona | base_T=0.7544948044875062 cur_T=0.906137690823718 base_dev=0.08593498488629436 cur_dev=0.03113022862093845
  - pos 20: Sam Wilkes - Own -> Art Feynman - Early Signs of Rhythm | base_T=0.5726484074044664 cur_T=0.8805088793040731 base_dev=0.2222045506634247 cur_dev=0.07984165790979869

- hash `9761e586a90c875399d22fac3385e8c4996abfcbc6e67dbec07c7d687b2ed59e` label `arc_w2.0_t0.2_abs_m0.1_autoTrue_tbNone_beam25` overlap=None
  - pos 1: David Bowie - Let's Dance -> David Bowie - Life On Mars? | base_T=None cur_T=None base_dev=None cur_dev=None
  - pos 2: Prince - Baby I'm a Star ("Take Me with U" 7" B-Side Edit) (Take Me with U 7" B-Side Edit; 2017 Remaster) -> teen suicide - My Little World | base_T=0.7877364949183092 cur_T=0.8061904896633129 base_dev=0.31086898616542963 cur_dev=0.3433774234597585
  - pos 3: Dead Kennedys - Viva Las Vegas (2022 Mix) -> Tom Waits - It's Over (Remastered) | base_T=0.9038338369336413 cur_T=0.7695236674405489 base_dev=0.3234365407861713 cur_dev=0.27705930792705674
  - pos 4: Big Black - Strange Things -> Wednesday - Carolina Murder Suicide | base_T=0.868413413419985 cur_T=0.8166042806934403 base_dev=0.13430912676290752 cur_dev=0.13534318234277787
  - pos 5: Nirvana - Swap Meet (Remastered) -> King Krule - The Cadet Leaps | base_T=0.8330160032210623 cur_T=0.7665084890929141 base_dev=0.05792099456554084 cur_dev=0.027197924918060112
  - pos 6: Minutemen - Corona -> Sigur Rós - Andrá | base_T=0.8609920669629056 cur_T=0.8160552428817576 base_dev=0.11121147782532603 cur_dev=0.07191639938006988
  - pos 7: Reatards - Your Old News Baby -> Codeine - Second Chance | base_T=0.8868485626989104 cur_T=0.6343020234150984 base_dev=0.20592478598467834 cur_dev=0.17720610789946512
  - pos 8: Talking Heads - Life During Wartime -> M83 - Another Wave from You | base_T=0.7747388414523939 cur_T=0.812172068147903 base_dev=0.32428595212644684 cur_dev=0.31354826082438414
  - pos 9: David Bowie - Suffragette City -> David Bowie - Warszawa | base_T=0.6329123608906382 cur_T=0.8865096925932843 base_dev=None cur_dev=None
  - pos 10: The Beatles - Long Tall Sally -> Microphones - Here With Summer | base_T=0.8438698874380759 cur_T=0.6552326095259688 base_dev=0.1837260675379918 cur_dev=0.21877083720095408
  - pos 11: The Fall - Just Step Sideways -> PaPerCuts - Mattress on the Floor | base_T=0.9020908500716743 cur_T=0.8967572984165522 base_dev=0.28779182893992095 cur_dev=0.25616906750238166
  - pos 12: The Beets - Friends Of Friends -> Guided By Voices - Tractor Rape Chain | base_T=0.7690583079574286 cur_T=0.8780779985451096 base_dev=0.07103774067145574 cur_dev=0.04132543425308627
  - pos 13: Seapony - Dreaming -> Ariel Pink's Dark Side - Queen of the Virgins | base_T=0.5812891962062137 cur_T=0.8572865593008591 base_dev=0.15653896152558605 cur_dev=0.09525394820549138
  - pos 14: Ariel Pink - Off The Dome -> Bikini Kill - Rebel Girl | base_T=0.8816809087878533 cur_T=0.8221788176610882 base_dev=0.04484929428316553 cur_dev=0.22193373855142984
  - pos 15: 48 Chairs - You Were Never There -> Tyvek - Going Through My Things | base_T=0.6213349949035449 cur_T=0.9268529584952317 base_dev=0.05632035573627997 cur_dev=0.41020552771838203
  - pos 16: Sonic Youth - Dude Ranch Nurse -> David Bowie - Suffragette City | base_T=0.6897304399639835 cur_T=0.8524873868886713 base_dev=0.020540089255099758 cur_dev=None
  - pos 17: Helvetia - 3 Boys -> No Age - Flutter Freer | base_T=0.8019982511029264 cur_T=0.8874990957333945 base_dev=0.08813442003398564 cur_dev=0.25178264270438616
  - pos 18: Wilco - Please Be Patient with Me -> Nirvana - Swap Meet (Remastered) | base_T=0.5282689610509975 cur_T=0.8811234108617932 base_dev=0.01117756888181809 cur_dev=0.2538239063638261
  - pos 19: Sufjan Stevens - That Was the Worst Christmas Ever! -> Art Feynman - Early Signs of Rhythm | base_T=0.7544948044875062 cur_T=0.8878898426648157 base_dev=0.08593498488629436 cur_dev=0.1426792760465157
  - pos 20: Sam Wilkes - Own -> Dead Kennedys - Viva Las Vegas (2022 Mix) | base_T=0.5726484074044664 cur_T=0.9134879188754146 base_dev=0.2222045506634247 cur_dev=0.08114361717105467

- hash `f15cb5589859ccf3bd2880d0909ea5a13725bc4f5ff81bfbe3810af963159685` label `arc_w2.0_t0.2_abs_m0.3_autoFalse_tbNone_beam50` overlap=None
  - pos 1: David Bowie - Let's Dance -> David Bowie - Life On Mars? | base_T=None cur_T=None base_dev=None cur_dev=None
  - pos 2: Prince - Baby I'm a Star ("Take Me with U" 7" B-Side Edit) (Take Me with U 7" B-Side Edit; 2017 Remaster) -> teen suicide - My Little World | base_T=0.7877364949183092 cur_T=0.8061904896633129 base_dev=0.31086898616542963 cur_dev=0.3433774234597585
  - pos 3: Dead Kennedys - Viva Las Vegas (2022 Mix) -> Tom Waits - It's Over (Remastered) | base_T=0.9038338369336413 cur_T=0.7695236674405489 base_dev=0.3234365407861713 cur_dev=0.27705930792705674
  - pos 4: Big Black - Strange Things -> Wednesday - Carolina Murder Suicide | base_T=0.868413413419985 cur_T=0.8166042806934403 base_dev=0.13430912676290752 cur_dev=0.13534318234277787
  - pos 5: Nirvana - Swap Meet (Remastered) -> King Krule - The Cadet Leaps | base_T=0.8330160032210623 cur_T=0.7665084890929141 base_dev=0.05792099456554084 cur_dev=0.027197924918060112
  - pos 6: Minutemen - Corona -> Sigur Rós - Andrá | base_T=0.8609920669629056 cur_T=0.8160552428817576 base_dev=0.11121147782532603 cur_dev=0.07191639938006988
  - pos 7: Reatards - Your Old News Baby -> Codeine - Second Chance | base_T=0.8868485626989104 cur_T=0.6343020234150984 base_dev=0.20592478598467834 cur_dev=0.17720610789946512
  - pos 8: Talking Heads - Life During Wartime -> M83 - Another Wave from You | base_T=0.7747388414523939 cur_T=0.812172068147903 base_dev=0.32428595212644684 cur_dev=0.31354826082438414
  - pos 9: David Bowie - Suffragette City -> David Bowie - Warszawa | base_T=0.6329123608906382 cur_T=0.8865096925932843 base_dev=None cur_dev=None
  - pos 10: The Beatles - Long Tall Sally -> Microphones - Here With Summer | base_T=0.8438698874380759 cur_T=0.6552326095259688 base_dev=0.1837260675379918 cur_dev=0.21877083720095408
  - pos 11: The Fall - Just Step Sideways -> PaPerCuts - Mattress on the Floor | base_T=0.9020908500716743 cur_T=0.8967572984165522 base_dev=0.28779182893992095 cur_dev=0.25616906750238166
  - pos 12: The Beets - Friends Of Friends -> Guided By Voices - Tractor Rape Chain | base_T=0.7690583079574286 cur_T=0.8780779985451096 base_dev=0.07103774067145574 cur_dev=0.04132543425308627
  - pos 13: Seapony - Dreaming -> Television Personalities - David Hockney's Diary | base_T=0.5812891962062137 cur_T=0.8583693022843132 base_dev=0.15653896152558605 cur_dev=0.02781028820660747
  - pos 14: Ariel Pink - Off The Dome -> Bikini Kill - Rebel Girl | base_T=0.8816809087878533 cur_T=0.8553334675301828 base_dev=0.04484929428316553 cur_dev=0.22193373855142984
  - pos 15: 48 Chairs - You Were Never There -> Tyvek - Going Through My Things | base_T=0.6213349949035449 cur_T=0.9268529584952317 base_dev=0.05632035573627997 cur_dev=0.41020552771838203
  - pos 16: Sonic Youth - Dude Ranch Nurse -> David Bowie - Suffragette City | base_T=0.6897304399639835 cur_T=0.8524873868886713 base_dev=0.020540089255099758 cur_dev=None
  - pos 17: Helvetia - 3 Boys -> No Age - Flutter Freer | base_T=0.8019982511029264 cur_T=0.8874990957333945 base_dev=0.08813442003398564 cur_dev=0.25178264270438616
  - pos 18: Wilco - Please Be Patient with Me -> Nirvana - Swap Meet (Remastered) | base_T=0.5282689610509975 cur_T=0.8811234108617932 base_dev=0.01117756888181809 cur_dev=0.2538239063638261
  - pos 19: Sufjan Stevens - That Was the Worst Christmas Ever! -> Dead Kennedys - Viva Las Vegas (2022 Mix) | base_T=0.7544948044875062 cur_T=0.8529024165312802 base_dev=0.08593498488629436 cur_dev=0.14137731678525972
  - pos 20: Sam Wilkes - Own -> Big Black - Strange Things | base_T=0.5726484074044664 cur_T=0.868413413419985 base_dev=0.2222045506634247 cur_dev=0.054227877558519655

## UI Dial Proposal

The following presets are derived from sweep outcomes. If pacing metrics are missing, use smoothness + overlap until pacing metrics are available.

| preset | label | weight | tolerance | loss | max_step | autoscale | tie_break_band |
| --- | --- | ---: | ---: | --- | ---: | --- | ---: |
| Loose | arc_w0.75_t0.0_abs_m0.1_autoFalse_tb0.02_beam50 | 0.75 | 0.0 | abs | 0.1 | False | 0.02 |
| Balanced | arc_w1.5_t0.08_abs_m0.15_autoTrue_tb0.02_beam10 | 1.5 | 0.08 | abs | 0.15 | True | 0.02 |
| Guided | arc_w2.0_t0.05_abs_m0.1_autoTrue_tbNone_beamNone | 2.0 | 0.05 | abs | 0.1 | True | None |
| Rail | arc_w2.0_t0.05_abs_m0.1_autoTrue_tbNone_beamNone | 2.0 | 0.05 | abs | 0.1 | True | None |
