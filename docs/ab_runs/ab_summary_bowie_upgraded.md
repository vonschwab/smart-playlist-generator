# Bowie A/B Summary (Upgraded)

## Creation Details

- A run_id: `ds_dynamic_20260106T155214Z_ad159afb`
- B run_id: `ds_dynamic_20260106T155230Z_ca61bb41`
- ds_mode: `dynamic`
- playlist_length: `30`

### Shared Tuning (A and B)

| setting | value |
| --- | --- |
| transition_floor | 0.2 |
| bridge_floor | 0.03 |
| weight_bridge | 0.6 |
| weight_transition | 0.4 |
| genre_tiebreak_weight | 0.05 |
| genre_penalty_threshold | 0.2 |
| genre_penalty_strength | 0.1 |
| genre_tie_break_band | None |
| segment_pool_strategy | segment_scored |
| segment_pool_max | 400 |
| progress.enabled | True |
| progress.monotonic_epsilon | 0.05 |
| progress.penalty_weight | 0.15 |

### Progress Arc Delta

| setting | A (baseline) | B (progress arc) |
| --- | --- | --- |
| progress_arc.enabled | False | True |
| progress_arc.weight | 0.25 | 0.25 |
| progress_arc.shape | linear | arc |
| progress_arc.tolerance | 0.0 | 0.0 |
| progress_arc.loss | abs | abs |
| progress_arc.huber_delta | 0.1 | 0.1 |
| progress_arc.max_step | None | None |
| progress_arc.max_step_mode | penalty | penalty |
| progress_arc.max_step_penalty | 0.25 | 0.25 |
| progress_arc.autoscale.enabled | False | False |
| progress_arc.autoscale.min_distance | 0.05 | 0.05 |
| progress_arc.autoscale.distance_scale | 0.5 | 0.5 |
| progress_arc.autoscale.per_step_scale | False | False |

## Summary Metrics

| metric | A (baseline) | B (progress arc) |
| --- | ---: | ---: |
| min_transition | 0.6618672715841734 | 0.6721428426990063 |
| mean_transition | 0.8177523403860731 | 0.8262608903413018 |
| below_floor_count | 0 | 0 |
| soft_genre_penalty_hits | 436 | 480 |
| soft_genre_penalty_edges_scored | 61863 | 83014 |

## Tracklist Overlap

- shared_tracks: `20`
- unique_to_A: `10`
- unique_to_B: `10`

## Data Science Assessment

- B improves mean transition similarity while keeping floors unchanged, indicating slightly tighter edge quality.
- B raises the minimum transition score, suggesting fewer weak links in the bridge path.
- Progress arc changes ordering without altering segment pool gating or transition floors; differences should be attributable to the arc scoring term.

## Tracklists

### A (Baseline)

| pos | artist | title | album | duration | track_id |
| ---: | --- | --- | --- | ---: | --- |
| 1 | David Bowie | Life On Mars? | Hunky Dory | 3:56 | `9dbf29a650b43f5ab7e0e836428000e1` |
| 2 | Helado Negro | Tartamudo | Private Energy | 4:11 | `b9f2f810cf9a35595709917651892d55` |
| 3 | St. Vincent | The Bed | Actor | 3:43 | `6de186f6b60d39d60608dfe9dd29fed3` |
| 4 | King Krule | The Cadet Leaps | The Ooz | 4:21 | `b0df2faa5f29325a90eb813611dd8d95` |
| 5 | Tim Hecker & Daniel Lopatin | Uptown Psychedelia | Instrumental Tourist | 5:58 | `5a26ffde7a71d272970b48f3cd3260a9` |
| 6 | Cornelius  | Thatness And Thereness (Cornelius Remodel) | Ethereal Essence | 3:40 | `94ca1ba5b36296ef818217619ac35c3e` |
| 7 | The Smashing Pumpkins | To Forgive (Remastered 2012) | Mellon Collie And The Infinite Sadness (30th Anniversary Edition) | 4:17 | `c4ccc75dc1eec7c7469aa39d50cb720f` |
| 8 | Sigur Rós | Andrá | ÁTTA | 4:07 | `6449f4bcc14f3aefb948532ba9d16110` |
| 9 | Foxes in Fiction | Insomnia Keys | Swung from The Branches | 3:28 | `4140456ae7d4877db12c0b739d585179` |
| 10 | M83 | Another Wave from You | Hurry Up, We're Dreaming  | 1:54 | `410dbf3fbade1aa7e68f0ae31803230f` |
| 11 | David Bowie | Warszawa | Low | 6:27 | `55979af56e425c605d2b8df5c6e1b5d7` |
| 12 | Cass McCombs | Interior Live Oak | Interior Live Oak | 5:58 | `14ff3a1a7c543efb2dd6c0b1e5ebc5ef` |
| 13 | PaPerCuts | Mattress on the Floor | Parallel Universe Blues | 1:58 | `75f32dbfe2c384ef0c3f57fc4b532a2a` |
| 14 | No Age | A Ceiling Dreams of a Floor | An Object | 2:36 | `ff2c3b740bba1a21db00357e0e00ae2c` |
| 15 | Parquet Courts | Psycho Structures | Content Nausea | 2:53 | `8cee27849333788854c8083df5c08368` |
| 16 | semiwestern | Zero for Conduct | semiwestern | 3:02 | `18e957230b984e54f971c38e86b31cd2` |
| 17 | Dirty Beaches | Sweet 17 | Badlands | 3:25 | `bdda5322fdb2ba33e1dbee89c248c82a` |
| 18 | Television Personalities | David Hockney's Diary | They Could Have Been Bigger Than The Beatles | 2:50 | `1b17ad82b6624ad6009c1b5207b6bbe7` |
| 19 | Ramones | What's Your Name (Demo) | Ramones: 40th Anniversary Deluxe Edition (Remastered) | 2:57 | `51f67e975d41324f8431c7098116b825` |
| 20 | Peach Kelli Pop | Big Man | Peach Kelli Pop III | 1:56 | `ea2261383751d36121242f9f70fd52ea` |
| 21 | David Bowie | Suffragette City | The Rise and Fall of Ziggy Stardust and the Spiders from Mars (1984 US CD release) | 3:24 | `328656b5878f1f293a4e981a0b3d961c` |
| 22 | No Age | Flutter Freer | People Helping People | 1:50 | `9c38290d44a212977e949e0dac9225cd` |
| 23 | Minutemen | Corona | Double Nickels on the Dime | 2:25 | `a9a2c69a0b00642bb7561ab014f2a7a4` |
| 24 | Pavement | Baptist Blacktick | Cautionary Tales: Jukebox Classiques | 2:03 | `de16c4dcbcc5426f7cedb62ba5c6b989` |
| 25 | Television Personalities | A Day in Heaven | Mummy Your Not Watching Me | 4:42 | `7d402918f60f8e80be22b7c5d673a983` |
| 26 | Sonic Youth w. Lydia Lunch | Death Valley '69 | Bad Moon Rising | 5:11 | `8e71005037f9cdf0a593c79dd0df1007` |
| 27 | The Jimi Hendrix Experience | Voodoo Child (Slight Return) | Electric Ladyland 50th | 5:13 | `0201222528b10d307e41061a2af38229` |
| 28 | Prince | Baby I'm a Star (2015 Paisley Park Remaster) | Purple Rain | 4:23 | `db8ff8f780183de419f945bad846df77` |
| 29 | The Stooges | T.V. Eye | Fun House | 4:18 | `45041aaba26141ecd87e8ed3dbef8084` |
| 30 | David Bowie | Let's Dance | Let's Dance | 7:38 | `00e9f570e6a9839e2844dd8410242fce` |

### B (Progress Arc)

| pos | artist | title | album | duration | track_id |
| ---: | --- | --- | --- | ---: | --- |
| 1 | David Bowie | Life On Mars? | Hunky Dory | 3:56 | `9dbf29a650b43f5ab7e0e836428000e1` |
| 2 | Helado Negro | Tartamudo | Private Energy | 4:11 | `b9f2f810cf9a35595709917651892d55` |
| 3 | teen suicide | My Little World | It's the Big Joyous Celebration, Let's Stir the Honeypot | 2:24 | `eee822e0bdce833182efd2454ad40f21` |
| 4 | Tom Waits | It's Over (Remastered) | Orphans: Brawlers, Bawlers & Bastards (Remastered) | 4:40 | `9545366b1dba1b951df147f0b1daa765` |
| 5 | King Krule | The Cadet Leaps | The Ooz | 4:21 | `b0df2faa5f29325a90eb813611dd8d95` |
| 6 | Tim Hecker & Daniel Lopatin | Uptown Psychedelia | Instrumental Tourist | 5:58 | `5a26ffde7a71d272970b48f3cd3260a9` |
| 7 | Sigur Rós | Andrá | ÁTTA | 4:07 | `6449f4bcc14f3aefb948532ba9d16110` |
| 8 | Foxes in Fiction | Insomnia Keys | Swung from The Branches | 3:28 | `4140456ae7d4877db12c0b739d585179` |
| 9 | Spacemen 3 | Things'll Never Be the Same | The Perfect Prescription | 6:04 | `b5bef3d7c5677b70d184ad25c92571e9` |
| 10 | M83 | Another Wave from You | Hurry Up, We're Dreaming  | 1:54 | `410dbf3fbade1aa7e68f0ae31803230f` |
| 11 | David Bowie | Warszawa | Low | 6:27 | `55979af56e425c605d2b8df5c6e1b5d7` |
| 12 | Cornelius | Surfing on Mind Wave, Pt. 2 | Mellow Waves | 5:34 | `3777383da08e29a2704112a370bfe3ab` |
| 13 | Microphones | Here With Summer | Don't Wake Me Up | 3:59 | `f6cd1c351475ff2e4e2e59b46f927a22` |
| 14 | Spacemen 3 | Feel So Good | The Perfect Prescription | 5:32 | `ca55b5cfce0e31e143277eada00a8f1d` |
| 15 | PaPerCuts | Mattress on the Floor | Parallel Universe Blues | 1:58 | `75f32dbfe2c384ef0c3f57fc4b532a2a` |
| 16 | No Age | A Ceiling Dreams of a Floor | An Object | 2:36 | `ff2c3b740bba1a21db00357e0e00ae2c` |
| 17 | Dirty Beaches | Sweet 17 | Badlands | 3:25 | `bdda5322fdb2ba33e1dbee89c248c82a` |
| 18 | Television Personalities | David Hockney's Diary | They Could Have Been Bigger Than The Beatles | 2:50 | `1b17ad82b6624ad6009c1b5207b6bbe7` |
| 19 | Ramones | What's Your Name (Demo) | Ramones: 40th Anniversary Deluxe Edition (Remastered) | 2:57 | `51f67e975d41324f8431c7098116b825` |
| 20 | empath | Born 100 Times | Visitor | 2:08 | `ef60a2304d7ca1d8bd87fb759655e4d2` |
| 21 | David Bowie | Suffragette City | The Rise and Fall of Ziggy Stardust and the Spiders from Mars (1984 US CD release) | 3:24 | `328656b5878f1f293a4e981a0b3d961c` |
| 22 | Art Feynman | In CD | Be Good The Crazy Boys | 4:16 | `d592e33af61879996ebc703301d6cc8a` |
| 23 | No Age | Flutter Freer | People Helping People | 1:50 | `9c38290d44a212977e949e0dac9225cd` |
| 24 | Here We Go Magic | Tunnelvision | Here We Go Magic | 4:22 | `5e686d3477b86d39a6db4dc1e8586f3a` |
| 25 | Ariel Pink’s Haunted Graffiti | Hot Body Rub | Before Today | 2:26 | `b908fd771414701ee666d7411b55e27f` |
| 26 | Pavement | Baptist Blacktick | Cautionary Tales: Jukebox Classiques | 2:03 | `de16c4dcbcc5426f7cedb62ba5c6b989` |
| 27 | Sonic Youth w. Lydia Lunch | Death Valley '69 | Bad Moon Rising | 5:11 | `8e71005037f9cdf0a593c79dd0df1007` |
| 28 | The Jimi Hendrix Experience | Voodoo Child (Slight Return) | Electric Ladyland 50th | 5:13 | `0201222528b10d307e41061a2af38229` |
| 29 | Prince | Baby I'm a Star (2015 Paisley Park Remaster) | Purple Rain | 4:23 | `db8ff8f780183de419f945bad846df77` |
| 30 | David Bowie | Let's Dance | Let's Dance | 7:38 | `00e9f570e6a9839e2844dd8410242fce` |
