# Mode Audit: David Bowie (Fixed Piers)

## Setup
- Artist: David Bowie
- Fixed pier seeds (rating_key order):
  1) 9dbf29a650b43f5ab7e0e836428000e1 (Life On Mars?)
  2) 55979af56e425c605d2b8df5c6e1b5d7 (Warszawa)
  3) 00e9f570e6a9839e2844dd8410242fce (Let's Dance)
  4) 328656b5878f1f293a4e981a0b3d961c (Suffragette City)
- DS mode: dynamic (constant)
- Dry-run: true
- Audit reports: docs/run_audits

## Runs (key metrics)
| Run | genre_mode | sonic_mode | min_genre_sim | min_sonic_sim | below_genre | below_sonic | eligible | mean_transition | audit file |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | strict | strict | 0.50 | 0.20 | 355 | 4629 | 6664 | 0.8176 | ds_dynamic_20260105T175041Z_bb5fb2a2.md |
| 2 | discover | discover | 0.20 | 0.00 | 0 | 372 | 11332 | 0.8249 | ds_dynamic_20260105T175100Z_33c18d94.md |
| 3 | strict | off | 0.50 | None | 635 | 0 | 11692 | 0.8144 | ds_dynamic_20260105T175116Z_16874bc7.md |
| 4 | off | strict | None | 0.20 | 0 | 0 | 18785 | 0.8351 | ds_dynamic_20260105T175129Z_12b9b3d4.md |
| 5 | dynamic | dynamic | 0.30 | 0.06 | 0 | 1057 | 10647 | 0.8249 | ds_dynamic_20260105T175147Z_1ec7a235.md |

## Sample tracklist differences (non-seed picks)
- strict/strict: St. Vincent - The Bed; King Krule - The Cadet Leaps; Sigur Ros - Andra; Weyes Blood - Used to Be; Chromatics - Through The Looking Glass.
- discover/discover: Helado Negro - Tartamudo; Tim Hecker & Daniel Lopatin - Uptown Psychedelia; Cornelius - Thatness And Thereness; Smashing Pumpkins - To Forgive.
- strict/off: similar to strict/strict early on, but genre-only weighting (sonic disabled).
- off/strict: Wednesday - Carolina Murder Suicide; Cole Pulice - Arc of Shadows; Shabason, Krgovich, Sage - Joe; M83 - Another Wave from You.
- dynamic/dynamic: close to discover early on (Helado Negro, Tim Hecker, Cornelius).

## Pier-bridge validation
- Fixed pier seeds were used in all runs (log shows "Passing 4 anchor_seed_ids" and "Pier seeds" with the same four IDs).
- Pier order was identical across runs: Life On Mars? -> Warszawa -> Suffragette City -> Let's Dance.
- All runs produced 30 tracks with 3 successful segments.

## Findings
1) Genre mode gates are working as expected.
   - strict/strict and strict/off enforce the hard gate (below_genre=355 and 635 respectively).
   - discover/dynamic/off show no hard genre exclusions.

2) Sonic mode now affects both weighting and min_sonic_similarity.
   - strict/strict uses min_sonic_similarity=0.20 and shows 4629 sonic rejections.
   - discover/dynamic drop the sonic floor (0.00/0.06), with fewer sonic rejections.
   - strict/off disables the sonic gate (min_sonic_similarity=None) and uses genre-only weighting.

3) Genre-only vs sonic-only modes are behaving distinctly.
   - strict/off drives w_genre=1.0, w_sonic=0.0 with hard genre gating.
   - off/strict drives w_sonic=1.0, w_genre=0.0 with genre gating disabled and a larger eligible pool.

## Recommended next checks
- If the goal is more dramatic mode separation, increase the spacing between min_sonic_similarity values (strict/narrow/dynamic/discover) and consider mode-specific similarity_floor values as well.
