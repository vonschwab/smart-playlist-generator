# Playlist trajectory control — literature review & conclusions

**Date:** 2026-06-24
**Purpose:** Ground the control-surface redesign ("roam corridors", spec `docs/superpowers/specs/2026-06-24-roam-corridors-control-surface-design.md`) in the data-science literature before committing the mechanism. The redesign proposed a *soft deviation penalty* to control how far playlist bridges may roam from the region the seeds define. This doc records what four independent literatures say about that choice, what they correct, and the refined mechanism we adopt.

**Method:** four parallel literature sweeps (trajectory optimization / MPC; discrete relevance–diversity selection; controllable movement in learned embedding spaces; MIR playlist sequencing), each judged against our specific setting (discrete pick-from-pool via beam search, fixed seed anchors, never-fail, worst-edge-critical). Citations collected at the end.

---

## TL;DR — the verdict

The soft-corridor **concept is validated** by all four literatures; it is the *only* mechanism that directly models "controlled deviation from an anchor region," and MIR work confirms the whole anchor-to-anchor-path framing. But the **naive form is wrong in three correctable ways**, and each fix targets a failure mode we already have:

1. **Knob = corridor *width* (radius / `k`), not penalty *weight* (λ).** Width is smooth and monotonic; λ→deviation is non-linear and pool-density-sensitive. Never-fail comes from *relaxing the width*, not tuning λ.
2. **Worst edge needs a *minimax / min-bottleneck* term, not an additive penalty.** Additive penalties optimize the average and will trade one terrible edge for several mild ones — violating "the worst edge defines the experience."
3. **The reference must be *on-manifold* (a kNN-graph geodesic over real tracks), not a straight-line interpolation.** A linear chord between seeds cuts through low-density "holes" where the nearest real track is perceptually wrong — i.e. our timbre-ceiling failure. Plus: **hubness-correct the distances (mutual proximity)** so hub tracks don't colonize every bridge.

Optional, complementary: a mild MMR-style within-playlist repulsion if bridges cluster on near-identical tracks.

**Refined mechanism we adopt:** a *soft, width-controlled corridor* around an *on-manifold kNN-graph reference path* between piers, with a *minimax worst-edge guard* and *hubness-corrected* distances.

---

## 1. Trajectory optimization / MPC — controlled deviation from a reference

**Mechanisms.** Exterior (quadratic) penalty methods; augmented Lagrangian; **tube-based robust MPC** (a nominal "spine" + an invariant "tube" the trajectory stays in; the tube radius is the tightness knob; recursive feasibility guaranteed — Mayne et al. 2005); **funnel MPC / prescribed-performance control** (a time-varying error funnel around the reference; the stage cost diverges at the funnel boundary; the funnel width *is* the knob — Berger et al.); trust-region methods (a radius `δ` bounding the step, shrunk on failure/grown on success); discrete analog = **constrained beam search with hard feasibility cuts** (reject partial solutions that provably can't satisfy a constraint).

**Findings.**
- An additive soft penalty in a *sum* objective (which beam search is) optimizes the **average/cumulative**, not the worst case. It will trade one very bad edge for several slightly-penalized ones. (Confirmed adversarially across penalty-method, NMT-beam-search, and constrained-beam-search sources.)
- The established always-feasible formulations (tube/funnel MPC) parameterize tightness as a **width**, not a penalty weight, and relax the width when infeasible. The width is the smooth knob; a penalty weight is not (doubling λ does not halve deviation).
- Pure exterior penalty does *not* guarantee constraint satisfaction (only in the limit p→∞). **This does not bite us** — we *want* a soft preference that relaxes, not a hard constraint to converge to; with the penalty → 0 at the corridor boundary and width-relaxation in the cascade, never-fail holds by construction.

**Verdict for us:** soft penalty is fine *as the soft component*, but (a) make the knob the width, (b) add a minimax/bottleneck term for the worst edge, (c) never-fail via width-relaxation (which our cascade already does).

## 2. Discrete relevance ↔ diversity selection

**Mechanisms.** **MMR** (Maximal Marginal Relevance — `λ·Rel − (1−λ)·max_sim(selected)`; Carbonell & Goldstein 1998); **DPP** (determinantal point processes — set-global pairwise repulsion); **Diverse Beam Search** (inter-group sequence diversity; Vijayakumar 2016); **stochastic / temperature beam search** (sample ∝ exp(score/T); Kool 2019); submodular coverage; UCB exploration bonuses.

**Findings.**
- Only the **soft deviation penalty** directly models *candidate-to-anchor-region* distance. MMR penalizes similarity to *already-selected* items (within-playlist diversity); DPP is pairwise repulsion; DBS diversifies the returned hypotheses; temperature is spatially blind. None of them is a corridor.
- To make MMR emulate a corridor you'd have to redefine its `Rel()` as "distance to seed centroid" — at which point you've rebuilt the soft anchor-penalty inside MMR's slot and discarded MMR's diversity term. So MMR is not a substitute.
- MMR/DPP repulsion *is* complementary: it prevents picking five near-identical tracks in a row, which the corridor does not address.

**Verdict for us:** the soft deviation penalty is the **correct primitive** for the roam corridor; it is additive on the beam edge score, composes natively, and never starves (penalty → 0 at the boundary). Add a mild within-playlist MMR repulsion only if clustering appears.

## 3. Controllable movement in a learned embedding space

**Mechanisms.** **kNN-graph shortest path → manifold geodesic** (Alamgir & von Luxburg, ICML 2012 — Dijkstra over a *distance-weighted* kNN graph converges to the true geodesic; unweighted graphs converge to a *detrimental* distance, so edge weighting is mandatory); **mutual proximity** for hubness (Flexer, Schnitzer & Stevens — rescales distance by neighbor symmetry, killing the high-dimensional pathology where a few "hub" tracks dominate everyone's kNN list; validated *specifically on content-based audio similarity*); geodesic vs linear interpolation under a learned metric (a straight chord cuts through the empty interior and exits the manifold; density-aware metrics route through where data actually is — Metric Flow Matching, NeurIPS 2024); SLERP (a band-aid for the hollow-shell problem, not a fix for non-spherical manifolds).

**The key warning (most important finding in the whole review).** Moving *linearly* through a high-dimensional learned space passes through **off-manifold, low-density "holes"** the model never saw, where points are unrealistic and distances are untrustworthy. In our discrete setting this is the **worst edge**: a linear/centroid target between two seeds lands in a sparse pocket of MERT space, and the nearest *real* track to that synthetic point is perceptually wrong — **exactly our documented timbre-ceiling coarseness (Metallica ≈ YYY in sparse regions; [[project_timbre_embedding_ceiling]]).** The prescription is unambiguous: **don't route to synthetic midpoints; route through real, dense, on-manifold items**, which a kNN-graph path does by construction, and protect it with hubness correction.

**Verdict for us:** "penalize deviation from a reference region" is sound **only if the reference is on-manifold**. Define the reference/progress against a **distance-weighted kNN-graph geodesic over the real track pool** (corridor width ≈ `k` / edge threshold), and **apply mutual-proximity hubness correction to the MERT distances** before building neighbors. Penalizing deviation from a linear chord would re-import the off-manifold worst-edge failure.

## 4. MIR / music-recsys playlist sequencing

**Prior systems.** Flexer & Schnitzer, "Playlist Generation Using Start and End Songs" (ISMIR 2008) — the seminal anchor-to-anchor path via shortest-path over pairwise acoustic distance; **STRAW** (random walk between styles, edge exists only below a distance threshold τ); **Bittner et al., "Automatic Playlist Sequencing and Transitions"** (ISMIR 2017, Spotify) — sequencing as a shortest-Hamiltonian-path weighted by acoustic/tonal/tempo continuity, validated by professional curators; **DJ-MC** (RL agent scoring song utility *and* transition reward separately; AAMAS 2015); **mood-dynamic playlists** (interpolate a path through the arousal–valence circumplex — direct precedent for an energy arc).

**Findings.**
- The pier-bridge framing (fixed anchors + beam-searched bridges + soft transition cost) is the canonical, most-defensible approach. Beam search is the standard practical approximation to the intractable Hamiltonian-path problem.
- **Hard admission thresholds (τ) disconnect the similarity graph → no path** — the STRAW failure, identical to our own cascade detonations from hard gates. Soft penalties in the objective are the literature's answer. Direct support for our "soft over hard" rule ([[feedback_generation_time_budget]]).
- **Greedy nearest-neighbor walks drift** away from the seed style; beam width is the documented mitigation (we already do this).
- **Average similarity is a misleading metric; the worst consecutive pair defines the experience.** Bittner's min-Hamiltonian-path is structurally a *min-bottleneck (widest-path)* problem — independent confirmation that the worst edge is the right optimization target.
- Multi-pier (several anchors) is *more* structured than most prior work (mostly two anchors) — a strength, not a risk.

**Verdict for us:** prior MIR work strongly supports the design and independently corroborates refinements #2 (worst-edge = min-bottleneck) and #3 (soft over hard, drift mitigation via beam).

---

## 5. Convergence with existing project decisions

This is not a fork. [[project_genre_steering_two_system]] (decided 2026-06-16) independently landed on **"a geodesic router over real populated genres + a mutual-proximity neighborhood metric"** — for the *genre* dimension. The literature here says the same two ideas (kNN-geodesic routing + mutual proximity) are the right treatment for the *sonic* dimension too. The unifying principle for the whole redesign: **every corridor routes over a hubness-corrected kNN graph of real tracks, against an on-manifold reference, with a min-bottleneck worst-edge guard and a width knob.**

## 6. The refined mechanism (what the spec adopts)

A per-dimension **roam corridor** = a *soft, width-controlled* preference around an *on-manifold kNN-graph reference path* between consecutive piers:
- **Knob:** corridor width (radius / `k`), smooth + monotonic; never-fail by relaxing width in the existing cascade.
- **Reference:** a distance-weighted kNN-graph geodesic over the real candidate pool (not a linear chord); every bridge hop is a real on-manifold track.
- **Distances:** mutual-proximity (hubness) corrected, at least for the MERT sonic dimension.
- **Worst-edge guard:** a minimax / min-bottleneck term (or per-step single-edge cut), so the corridor never averages away one jarring transition.
- **Soft penalty:** the deviation-beyond-width term remains, as the soft (never-starving) corridor; penalty → 0 at the boundary.
- **Optional:** mild within-playlist MMR repulsion if near-identical clustering appears.

## 7. Open questions for calibration
- kNN `k` per dimension and per mode (literature: `k ≈ 20–30` typical for music; small = narrow/safe, large = wide/discovery).
- The width→`k`/radius mapping and the soft-penalty slope beyond the boundary.
- Whether mutual proximity is needed on genre/energy too, or sonic suffices.
- Robust energy band (p10–p90 of seed arousal vs raw min/max) when one seed is an energy outlier.
- The exact minimax-vs-additive blend for the worst-edge guard.

---

## Citations
- Mayne, Seron, Raković (2005), "Robust model predictive control of constrained linear systems with bounded disturbances" (tube MPC).
- Berger et al., "Funnel MPC for nonlinear systems with relative degree one," arXiv 2107.03284.
- "Corridor MPC: Towards Optimal and Safe Trajectory Tracking" (ResearchGate 363313282).
- Nocedal & Wright, *Numerical Optimization*, Ch. 17 (penalty / augmented Lagrangian).
- "Neural Sequence Generation with Constraints via Beam Search with Cuts" (AAAI SOCS 2024).
- Carbonell & Goldstein (1998), MMR (SIGIR'98).
- Chen et al. (2018), "Fast Greedy MAP Inference for DPP," arXiv 1709.05135; Liu et al. (2022), "DPP Likelihoods for Sequential Recommendation," arXiv 2204.11562 (SIGIR'22).
- Vijayakumar et al. (2016), "Diverse Beam Search," arXiv 1610.02424.
- Kool et al. (2019), "Stochastic Beams and Where to Find Them," ICML'19.
- Alamgir & von Luxburg (2012), "Shortest path distance in random k-NN graphs," ICML'12.
- Flexer, Schnitzer & Stevens, "Mutual proximity graphs for improved reachability in music recommendation" (JNMR 2017; ISMIR 2011).
- "Metric Flow Matching" (NeurIPS 2024); Chen & Lipman (2023), "Flow Matching on General Geometries."
- Flexer, Schnitzer, Gasser, Widmer (2008), "Playlist Generation using Start and End Songs," ISMIR'08.
- Ferraro, Bogdanov et al. (2019), "Random Playlists Smoothly Commuting Between Styles" (STRAW), ACM.
- Bittner et al. (2017), "Automatic Playlist Sequencing and Transitions," ISMIR'17 (Spotify Research).
- Liebman, Saar-Tsechansky, Stone (2015), "DJ-MC: A Reinforcement-Learning Agent for Music Playlist Recommendation," arXiv 1401.1880 (AAMAS'15).
- Chotas & Bailey (2016), "Towards Playlist Generation Algorithms Using RNNs Trained on Within-Track Transitions," arXiv 1606.02096.
- "Generating Smooth Mood-Dynamic Playlists with Audio Features and KNN" (Springer 2024); "Mood Dynamic Playlist: Interpolating a Musical Path Between Emotions" (2022).
