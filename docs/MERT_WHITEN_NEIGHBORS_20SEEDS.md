# MERT `whiten_l2` — cross-catalog nearest neighbors for 20 diverse seeds

**Transform:** `whiten_l2` (mean-center → per-dim std → L2) over mean-pooled-13-layer MERT-v1-95M embeddings.
**Pool:** all 8,658 scanned tracks (full library so far, no per-artist cap).
**Seed-artist EXCLUDED** via `normalize_artist_key` parity with the generator — every track sharing the seed's artist key is removed from that seed's pool, so these are genuine *cross-catalog* matches (what the playlist generator actually does: principle 10/11). This supersedes the earlier self-inclusive dump, which was ~3.4× inflated by intra-catalog self-matches.
**Score:** cosine in whitened MERT space. Genres = top effective genres (eyeball only; not used in retrieval).
**Flag:** `✓` on-target · `~` adjacent/defensible · `⚠` genre miss · `◆` alias/collab leak (see note).

> Source: `scripts/calibrate_mert_transform.py --seeds data/mert_calibration_seeds_diverse.txt --subset-size 1000000 --max-per-artist 1000000` (seed-artist exclusion built in).
>
> **Two identity leaks remain** that the generator's *full* identity resolver would catch but the `normalize_artist_key` approximation does not: **AFX ↔ Aphex Twin** (an alias, not a normalization) and **"Beyoncé feat. BEAM" ↔ Beyoncé** (a collaboration; the generator splits participants, we don't). Flagged `◆` below. They affect 2 of 20 seeds and don't move the verdict.

---

### Black Flag — Forever Time  *(hardcore punk / grunge)* — **the cassette-fi test**
| cos | neighbor | genres | |
|----:|---|---|:--:|
| +0.457 | Dinosaur Jr. — I Don't Wanna Go There | noise rock, grunge, hard rock | ✓ |
| +0.438 | Dinosaur Jr. — Start Choppin | grunge, hard rock, alternative rock | ✓ |
| +0.411 | Dinosaur Jr. — Said the People | noise rock, grunge, hard rock | ✓ |
| +0.409 | Dinosaur Jr. — Pieces | noise rock, grunge, hard rock | ✓ |
| +0.395 | Dinosaur Jr. — Pieces | grunge, hard rock, alternative rock | ✓ |
| +0.389 | Dinosaur Jr. — Love Is... | noise rock, grunge, alternative rock | ✓ |

> **Verdict on your hypothesis:** it did *not* return random lo-fi cassette tracks — it returned Dinosaur Jr., a genre-adjacent loud-distorted-guitar band (a studio act, *not* cassette-fi). So MERT is keying on the punk/noise-rock idiom, not tape hiss. The caveat: it's monogamously Dinosaur Jr. — one tight production/timbre cluster swept the top 6, where Dead Kennedys / Sonic Youth / Black Sabbath (all in-library) didn't crack it. That's the timbre-coarseness signature again, milder than the towers. At playlist level the per-artist cap means only 1 of these is ever picked, so it's a non-issue in practice.

### Charli XCX — Baby  *(hyperpop / electropop)*
| cos | neighbor | genres | |
|----:|---|---|:--:|
| +0.607 | Carly Rae Jepsen — Stay Away | electropop, synth-pop, contemporary r&b | ✓ |
| +0.566 | Dua Lipa — Whatcha Doing | alternative pop, boogie, electropop | ✓ |
| +0.560 | Dua Lipa — Whatcha Doing (Extended) | dance-pop, electropop, nu-disco | ✓ |
| +0.558 | Dua Lipa — Cool | europop, synth-pop, nu-disco | ✓ |
| +0.554 | Carly Rae Jepsen — Talking to Yourself | synth-pop, indie pop, dance-pop | ✓ |
| +0.546 | Alan Palomo — Nudista Mundial '89 | chillwave, city pop, disco | ✓ |

### Donald Byrd — Just My Imagination  *(jazz fusion / soul jazz)*
| cos | neighbor | genres | |
|----:|---|---|:--:|
| +0.471 | Cecil Wary — Olympic's Starter | afro-cuban jazz, easy listening, jazz fusion | ✓ |
| +0.449 | Cecil Wary — European's Confrontation | afro-cuban jazz, easy listening, jazz fusion | ✓ |
| +0.431 | Cocteau Twins — Squeeze-Wax | indie, pop, rock | ⚠ |
| +0.413 | Antônio Carlos Jobim — Tema Jazz | jazz, bossa nova, brazilian | ✓ |
| +0.413 | 48 Chairs — Snap It Around | art punk, minimal wave, power pop | ⚠ |
| +0.409 | Belle and Sebastian — Women's Realm | folk rock, j-rock, psychedelic pop | ⚠ |

> Half-right without the catalog crutch: real jazz (Cecil Wary, Jobim) at the top, then drift into dream-pop / art-punk. MERT's jazz-fusion representation is decent, not airtight.

### Aphex Twin — midi pipe2c edit  *(IDM / techno)*
| cos | neighbor | genres | |
|----:|---|---|:--:|
| +0.588 | AFX — Bodmin 2 | techno, idm, acid techno | ◆ alias |
| +0.544 | Vegyn — If looks could talk | downtempo, idm, instrumental hip hop | ✓ |
| +0.531 | AFX — Carnival Acid | drill and bass, glitch, idm | ◆ alias |
| +0.530 | Vegyn — LA musica | downtempo, idm, instrumental hip hop | ✓ |
| +0.527 | Autechre — bqbqbq | techno, idm, ambient techno | ✓ |
| +0.521 | AFX — T08 dx1+5 | idm, electro, drill and bass | ◆ alias |

> AFX is Aphex Twin's alias — the generator's full identity resolver wouldn't necessarily catch it either, but it's not a true cross-artist match. The genuine ones (Vegyn, Autechre) are spot-on IDM.

### Cate le Bon — Moderation  *(art pop / indie rock)*
| cos | neighbor | genres | |
|----:|---|---|:--:|
| +0.471 | Art Feynman — Emancipate Your Love Life | lounge, indie pop | ~ |
| +0.459 | DIIV — Valentine | dream pop, jangle pop, krautrock | ✓ |
| +0.455 | Cocteau Twins — Wolf in the Breast | ambient techno, indie rock, dream pop | ✓ |
| +0.453 | Alvvays — Party Police | alternative rock, dream pop, twee pop | ✓ |
| +0.449 | Cocteau Twins — Tishbite | dream pop, shoegaze, ethereal wave | ✓ |
| +0.416 | Destroyer — Chinatown | soft rock, chamber pop, easy listening | ~ |

### SF Symphony / MTT — Ibéria  *(classical / orchestral)*
| cos | neighbor | genres | |
|----:|---|---|:--:|
| +0.610 | Myung-Whun Chung — Hymne Au Saint-Sacrement | classical | ✓ |
| +0.574 | Akira Kosemura — Imagery | contemporary classical, modern classical, minimalist | ✓ |
| +0.536 | Danny L Harle — For So Long | bubblegum bass, pc music, hyperpop | ⚠ |
| +0.533 | Dirty Projectors — Now I Know | indie rock, art pop, freak folk | ⚠ |
| +0.531 | Tom Waits — Whistle Down The Wind | americana, avant-garde jazz, experimental | ~ |
| +0.524 | Johann Pachelbel — Three Variations | alternative, avant-garde, experimental | ✓ |

> 3/6 real classical; the rest is MERT reaching because the 8.6k-track library is classical-sparse. Expect this to firm up as the full ~40k library fills in more orchestral material.

### Durand Jones & The Indications — Sea Gets Hotter  *(neo-soul / r&b)*
| cos | neighbor | genres | |
|----:|---|---|:--:|
| +0.409 | Duke Ellington / Coltrane — My Little Brown Book | bebop, cool jazz, hard bop | ~ |
| +0.403 | Brendan Eder Ensemble — Cape Cod Cottage | easy listening, contemporary jazz, classical | ~ |
| +0.395 | Gianni Brezzo — Virginio | nu jazz | ~ |
| +0.388 | Antônio Carlos Jobim — Rockanalia | jazz, bossa nova, brazilian | ~ |
| +0.381 | Stan Getz & João Gilberto — Corcovado | jazz | ~ |
| +0.366 | Alex G — Immunity | bedroom pop, alternative country, americana | ⚠ |

> Leans jazz/lounge rather than soul — the smooth warm production reads as "mellow jazz." Weakest absolute cosines of the 20.

### David Bowie — D.J.  *(art rock / new wave)*
| cos | neighbor | genres | |
|----:|---|---|:--:|
| +0.550 | Echo & The Bunnymen — Rescue | new wave, alternative rock, post-punk | ✓ |
| +0.495 | Echo & the Bunnymen — Rescue | new wave | ✓ |
| +0.462 | Dead Kennedys — Ill In The Head | hardcore punk, classic punk, punk rock | ~ |
| +0.457 | Dead Kennedys — Fleshdunce | hardcore punk, classic punk | ~ |
| +0.456 | Blitzen Trapper — The All Girl Team | alternative country, indie folk, indie rock | ~ |
| +0.453 | Dead Kennedys — Kill The Poor | hardcore punk, classic punk, punk rock | ~ |

### 90 Day Men — Sweater Queen  *(post-rock / math rock)*
| cos | neighbor | genres | |
|----:|---|---|:--:|
| +0.454 | blue smiley — hurt | noise pop, shoegaze, slacker rock | ~ |
| +0.438 | Dead Kennedys — This Could Be Anywhere | hardcore punk, post-punk, classic punk | ~ |
| +0.432 | Bill Callahan — Bowevil | indie folk, alternative country, americana | ⚠ |
| +0.418 | blue smiley — bad | noise pop, shoegaze, slacker rock | ~ |
| +0.407 | Damak — Side B | post-punk | ✓ |
| +0.393 | Built To Spill — Another Day | indie pop, singer-songwriter, alternative rock | ~ |

### Bowery Electric — Black Light  *(ambient dub / dream pop / downtempo)*
| cos | neighbor | genres | |
|----:|---|---|:--:|
| +0.546 | Deerhunter — White Ink | neo-psychedelia, post-punk, noise rock | ~ |
| +0.491 | Ann Annie — home | contemporary classical, ambient americana, ambient electronic | ~ |
| +0.484 | Built to Spill — Velvet Waltz | indie rock, alternative rock, indie pop | ⚠ |
| +0.464 | Codeine — Old Things | shoegaze, slowcore, alternative rock | ~ |
| +0.454 | Dinosaur Jr. — I Ain't | grunge, hard rock, alternative rock | ⚠ |
| +0.453 | Black Sabbath — After Forever | stoner rock | ⚠ |

> Most scattered of the 20 (also true in the self-inclusive run). Hazy downtempo texture pulls a slowcore/shoegaze/grunge grab-bag. Genuine weak spot.

### Al Green — La-La for You  *(southern soul / r&b)*
| cos | neighbor | genres | |
|----:|---|---|:--:|
| +0.512 | Ann Peebles — A Love Vibration | rhythm and blues, r&b/soul, southern soul | ✓ |
| +0.448 | Dirty Projectors — What Is The Time (Live) | indie rock, art pop, freak folk | ⚠ |
| +0.420 | The Supremes — He Means The World To Me | rhythm and blues, disco, pop soul | ✓ |
| +0.410 | Ariel Pink's Haunted Graffiti — Hot Body Rub | psychedelic pop, new wave, freak folk | ~ |
| +0.407 | Eddie Chacon — Outside | electropop, indie pop, funk | ~ |
| +0.384 | Doris Duke — He's Gone | soul | ✓ |

> Cross-catalog soul works: Ann Peebles, The Supremes, Doris Duke are genuine soul/R&B from other artists.

### Antônio Carlos Jobim — Tema Jazz  *(bossa nova / jazz)*
| cos | neighbor | genres | |
|----:|---|---|:--:|
| +0.616 | Chihiro Yamanaka — Rhapsody in Blue | j-pop, smooth jazz, contemporary jazz | ✓ |
| +0.615 | Chihiro Yamanaka — Piano Sonata No. 4 | j-pop, smooth jazz, contemporary jazz | ✓ |
| +0.591 | Donald Byrd — Essence | disco, funk, jazz fusion | ✓ |
| +0.587 | Chihiro Yamanaka — Utopia | j-pop, smooth jazz, contemporary jazz | ✓ |
| +0.585 | Duke Ellington / Coltrane — Big Nick | bebop, cool jazz, hard bop | ✓ |
| +0.575 | Chihiro Yamanaka — Le Cygne | j-pop, smooth jazz, contemporary jazz | ✓ |

> Skews to instrumental piano-jazz (Chihiro Yamanaka cluster) rather than bossa specifically — sonically coherent, culturally a touch off the bossa mark.

### Ciccone Youth — Burnin' Up  *(experimental rock / no wave — a lo-fi SY Madonna cover)*
| cos | neighbor | genres | |
|----:|---|---|:--:|
| +0.473 | The Beatles — Another Girl | folk rock, psychedelic rock, folk pop | ⚠ |
| +0.459 | David Bowie — Repetition | alternative rock, art pop, new wave | ~ |
| +0.447 | David Bowie — Hang On to Yourself | alternative rock, art pop, new wave | ~ |
| +0.447 | David Bowie — Hang On to Yourself | art rock, glam rock, art pop | ~ |
| +0.419 | ARTHUR — Wow Fuck | emo, hypnagogic pop, psychedelic folk | ~ |
| +0.415 | Ariel Pink — I Wanna Be A Girl | indie rock, synth-pop, shoegaze | ~ |

### Destroyer — Dark Purposes  *(art pop / chamber)*
| cos | neighbor | genres | |
|----:|---|---|:--:|
| +0.591 | Boards of Canada — Sundown | folktronica, electronica, trip-hop | ~ |
| +0.581 | Cindy Lee — Gallows Smile | experimental rock, noise pop, dream pop | ✓ |
| +0.579 | Black Moth Super Rainbow — Changing You All | downtempo, hypnagogic pop, neo-psychedelia | ~ |
| +0.577 | Daniel Johnston — Going Down | freak folk, bedroom pop, avant-garde pop | ~ |
| +0.577 | Daniel Johnston — Worried Shoes | freak folk, indie pop, indie rock | ~ |
| +0.572 | Being Dead — Storybook Bay | surf rock, indie rock, indie pop | ~ |

### Danny L Harle — Ocean's Theme  *(PC music / ambient)*
| cos | neighbor | genres | |
|----:|---|---|:--:|
| +0.539 | Dravier — Sacred Passage | house, techno | ✓ |
| +0.531 | Alex Somers — Avalanche | neoclassical | ✓ |
| +0.516 | Alex Somers — Atlas | ambient, experimental | ✓ |
| +0.502 | Autechre — esc desc | ambient techno, avant-garde, drill and bass | ✓ |
| +0.497 | Dravier — Discovery | house, techno | ✓ |
| +0.485 | Félicia Atkinson & J. — Indefatigable Purple | (none) | ✓ |

### Beach House — 10:37  *(this cut is ambient/instrumental, not their pop)*
| cos | neighbor | genres | |
|----:|---|---|:--:|
| +0.432 | Brendan Eder Ensemble — QX 2021 | chamber music, neoclassical, contemporary classical | ~ |
| +0.417 | Landon Caldwell — Four Ways To Peel An Orange | ambient jazz, tape music, film score | ~ |
| +0.398 | Julianna Barwick — Pyrrhic | ambient pop, dream pop, experimental electronic | ✓ |
| +0.390 | Black Moth Super Rainbow — Snail Garden | hypnagogic pop, indietronica, psychedelic rock | ~ |
| +0.386 | Black Moth Super Rainbow — Snail Garden | hypnagogic pop, indietronica, neo-psychedelia | ~ |
| +0.380 | Bitchin Bajas — Space Is the Place | experimental electronic, new age, progressive electronic | ✓ |

### Ana Benegas — Las Olas  *(spanish electronic / pop)*
| cos | neighbor | genres | |
|----:|---|---|:--:|
| +0.470 | Carly Rae Jepsen — Bends | synth-pop, indie pop, dance-pop | ✓ |
| +0.439 | Aphex Twin — Cock/ver10 | acid breaks, breakcore, idm | ⚠ |
| +0.429 | Ava Luna — Stages | alternative r&b, art pop, experimental pop | ~ |
| +0.429 | Blood Orange — Good For You | alternative r&b, neo-soul, r&b/soul | ~ |
| +0.404 | David Woodruff — Drop A Line | indie rock | ~ |
| +0.404 | Blood Orange — I Wanna C U | alternative r&b, neo-soul, r&b/soul | ~ |

### Beyoncé — ENERGY  *(pop / r&b)*
| cos | neighbor | genres | |
|----:|---|---|:--:|
| +0.998 | Beyoncé feat. BEAM — ENERGY | pop | ◆ collab dupe |
| +0.625 | De La Soul — The Bizness (feat. Common) | alternative hip hop, hip hop, jazz rap | ✓ |
| +0.572 | B. Cool-Aid — Brandy, Aaliyah | alternative hip hop, hip hop, jazz rap | ✓ |
| +0.542 | A Tribe Called Quest — Give Me | boom bap, jazz rap, hip hop | ✓ |
| +0.540 | Caribou — Odessa (Junior Boys Mix) | electro, dream pop, indietronica | ~ |
| +0.537 | A Tribe Called Quest — Steppin' It Up | boom bap, jazz rap, hip hop | ✓ |

> The +0.998 is the *same track* credited "Beyoncé feat. BEAM" — a collaboration the generator's resolver would split-and-exclude; our key-only approximation doesn't. Ignore it; the rest is genuine hip-hop.

### Dirty Beaches — Berlin  *(hypnagogic pop / rockabilly / post-punk)*
| cos | neighbor | genres | |
|----:|---|---|:--:|
| +0.707 | Corridor — Supermercado | post-punk, art rock, jangle pop | ✓ |
| +0.634 | Daniel Johnston — An Incoherent Speech | freak folk, indie pop, indie rock | ~ |
| +0.627 | Boards of Canada — Bocuma | trip-hop, downtempo, idm | ~ |
| +0.624 | Daniel Johnston — Scrambled Eggs | freak folk, indie pop, indie rock | ~ |
| +0.623 | Landon Caldwell — What Seems Like An End | ambient jazz, tape music, film score | ~ |
| +0.610 | Daniel Johnston — Wicked World | freak folk, indie pop, indie rock | ~ |

> Lo-fi/hypnagogic texture cluster — Corridor (post-punk) is a strong match; the Daniel Johnston run is lo-fi-adjacent but genre-loose.

### Ativin — Scissors  *(experimental rock / math rock)*
| cos | neighbor | genres | |
|----:|---|---|:--:|
| +0.459 | Codeine — Promise Of Love | shoegaze, indie rock, slowcore | ~ |
| +0.448 | Black Sabbath — After Forever | stoner rock | ~ |
| +0.426 | Codeine — Realize | post-rock, slowcore, shoegaze | ✓ |
| +0.425 | 90 Day Men — A National Car Crash | post-punk, post-rock, math rock | ✓ |
| +0.423 | Jim O'Rourke — There's Hell in Hello | indie rock, american primitivism, avant-folk | ~ |
| +0.413 | Bill Callahan — Naked Souls | indie folk, alternative country, americana | ⚠ |

> 90 Day Men is the standout cross-catalog hit (both midwest math/post-rock) — note this is *mutual* (Ativin appears under 90 Day Men too).

---

## Verdict (cross-catalog, seed-artist excluded)

| | self-inclusive | **cross-catalog (honest)** | Δ |
|---|---:|---:|---:|
| whiten_l2 | 0.488 | **0.142** | −71% |
| towers_baseline | 0.318 | **0.084** | −74% |
| MERT lead over towers | +54% | **+69%** | wider |

1. **The self-match inflation was real and large** (~3.4×). Your call to exclude the seed artist was correct and necessary — the previous lists were flattered by every artist trivially matching its own catalogue.

2. **MERT's advantage over the towers got *wider*, not narrower** (+69% vs +54%). The towers lean even harder on same-artist similarity, so they collapse further when it's removed. MERT retains more genuine cross-artist structure. This strengthens the case for folding MERT.

3. **The cassette-fi worry is answered:** Black Flag pulled Dinosaur Jr. (genre-adjacent loud guitar rock), not random tape-hiss. MERT keys on idiom, not recording medium. The residual concern is *clustering* — one tight production neighborhood can sweep a seed's top-6 (all-Dinosaur-Jr., all-Chihiro-Yamanaka, all-Daniel-Johnston) — but the generator's per-artist cap neutralizes that downstream.

4. **Failure modes are concentrated, not random:** hazy/lo-fi seeds (Bowery Electric, Ciccone Youth, Destroyer) and genre-sparse neighborhoods (classical) are where it drifts — exactly where a learned audio embedding has least to grab and where genre⊗sonic fusion is meant to backstop it.

5. **Absolute 0.142 is low but expected** for cross-artist genre-maxsim (different artists share only partial genre tags). The right read is relative: 0.142 vs 0.084 floor, with ~60–70% genre-plausible neighbors by eye — workable input for the fusion + beam, not a finished playlist on its own.

**Net:** `whiten_l2` confirmed as the transform, and MERT confirmed as a real upgrade over the towers under the honest cross-catalog test. The folding decision still waits on full-library extraction + refit; the listen test is the final gate.
