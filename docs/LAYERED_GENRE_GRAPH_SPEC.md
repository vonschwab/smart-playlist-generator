# Layered Genre Graph Specification

Status: Draft for future implementation

Last updated: 2026-06-04

## 1. Problem Statement

Playlist Generator currently stores and uses genres mostly as flat tags. A track or release can carry multiple tags, and those tags are converted into genre vectors for candidate admission, scoring, transition quality, and genre-aware DJ bridge routing. This has already improved with IDF-weighted genre admission, vector-mode DJ genre routing, and coverage bonuses that make rare or specific tags like `shoegaze`, `slowcore`, or `jangle pop` matter more than broad tags like `rock` or `pop`.

The remaining problem is that one flat genre representation is doing too many jobs.

Broad tags are noisy, but still useful. `rock`, `pop`, `electronic`, `folk`, and `jazz` should usually not prove a tight match by themselves, but they do provide a larger musical neighborhood. They are useful for fallback context, safe bridge regions, and avoiding incoherent family jumps.

Niche tags are precise, but sparse. A release tagged `slowcore` or `american primitivism` has a stronger musical identity than a release tagged only `rock`, but strict matching on leaf tags can over-isolate the playlist when the library has few exact overlaps.

Similar-but-different niche genres need explicit connection. `shoegaze` and `dream pop`, `jangle pop` and `twee pop`, or `avant-folk` and `indie folk` are not identical, but they are musically close enough that the system should understand a route between them. A flat vector can detect overlap but cannot explain why two non-identical terms should connect.

Cross-genre transitions should be conditional. `indie pop` to `synth-pop` can work when the tracks share pop-family context, synth-heavy production, danceable rhythm, or compatible timbre. `indie pop` to `techno` should not be admitted just because both have a loose `pop` or `electronic` neighborhood unless there is a strong intermediate bridge path or very strong sonic and facet evidence.

The current single genre similarity score conflates:

- Family or broad-neighborhood affinity.
- Precise leaf or niche compatibility.
- Descriptor/facet compatibility.
- Cross-genre bridge permission.
- Confidence in the source evidence.
- Penalties for broad-only or noisy metadata.

The Layered Genre Graph separates those concerns so genre can guide both tight transitions and dynamic genre-spanning playlists.

## 2. Design Philosophy

Core principle: genre should not be only a label system. It should be a routing substrate for playlist generation.

The system should preserve multi-genre signatures and use them as structured routing information. A release is not simply "rock" or "indie pop"; it may sit in a neighborhood shaped by family membership, leaf genres, scene context, production texture, and sonic behavior.

The intended model:

- Broad genre/family = safe larger musical neighborhood.
- Niche genre/style = tight compatibility and transition lock.
- Facets = mood, texture, instrumentation, era, production, scene, and contextual attributes.
- Bridge edges = justified movement between related but non-identical genres.
- Sonic, pace, and transition evidence = proof that the bridge actually works musically.

This should be a graph or DAG, not a strict tree. A genre may have multiple parents or relationships:

- `jangle pop` can live under `indie pop`, `pop rock`, `pop family`, and `rock family`.
- `ambient americana` can connect to `ambient`, `american primitivism`, `folk`, and `country/roots`.
- `dance-punk` can bridge `post-punk`, `punk`, `dance`, and `electronic`.
- `shoegaze` can sit near `dream pop`, `noise pop`, `indie rock`, and `ambient pop`.

The graph should be explainable and versioned. It does not need to be perfect. It needs to be practical, auditable, and tunable.

## 3. Taxonomy Layers

### A. Family Layer

Examples:

- `rock`
- `pop`
- `electronic`
- `folk`
- `jazz`
- `hip hop`
- `punk`
- `metal`
- `ambient/new age`
- `r&b/soul`
- `country/roots`
- `classical/modern composition`
- `experimental`

Purpose:

- Broad neighborhood context.
- Fallback context for sparse releases.
- Safe bridge region for related genres.
- Low-weight compatibility feature.
- Not strong enough alone to admit candidates in `strict` or `narrow` modes.

Family tags should usually be inferred from more specific observed genres. If a release has `jangle pop`, the system can infer `pop family` and possibly `rock family`, but those inferred memberships should be lower weight than the observed `jangle pop` leaf tag.

Family tags should not dominate candidate admission. A candidate whose only overlap is `rock`, `indie`, `alternative`, or `pop` should fail `strict` and usually fail `narrow` unless separate sonic and facet evidence is unusually strong.

### B. Genre/Style Layer

Examples:

- `indie pop`
- `jangle pop`
- `twee pop`
- `synth-pop`
- `slowcore`
- `shoegaze`
- `dream pop`
- `post-punk`
- `dance-punk`
- `avant-folk`
- `american primitivism`
- `ambient americana`
- `ambient dub`

Purpose:

- Primary musical identity.
- High-weight IDF matching.
- Precision in `strict` and `narrow` modes.
- Top-K signature extraction for coverage behavior.
- Main input for genre-aware bridge routing.

Leaf genres should retain strong influence. The current IDF behavior is valuable and should remain: rarer genre/style terms should matter more than broad families when they are supported by reliable evidence.

### C. Facet Layer

Examples:

- Mood: `melancholic`, `warm`, `bleak`, `euphoric`
- Texture: `lo-fi`, `noisy`, `pastoral`, `reverb-heavy`, `minimal`
- Instrumentation: `acoustic`, `synth-heavy`, `female vocals`, `instrumental`, `drum machine`
- Production: `diy`, `home-recorded`, `polished`, `sample-based`
- Era: `1980s`, `1990s`, `2000s`
- Function: `danceable`, `sleep`, `study`, `meditative`
- Scene/context: `anacortes`, `kranky`, `c86`, `pc music`

Purpose:

- Explain compatibility across genre boundaries.
- Validate bridge movement.
- Improve transition scoring without pretending descriptors are genres.
- Support future GUI review workflows that separate "is this a genre?" from "is this useful metadata?"

Facets should not be treated as genre/style tags. For example:

- `lo-fi` is usually a production or texture facet unless it appears in a canonical phrase like `lo-fi indie` or `lo-fi bedroom pop`.
- `japanese` is a region/context facet, not a genre.
- `female vocals` is a vocal facet, not a genre.
- `1980s` is an era facet, not a genre.

### D. Bridge Layer

Examples:

- `indie pop` <-> `synth-pop`
- `jangle pop` <-> `twee pop`
- `shoegaze` <-> `dream pop`
- `avant-folk` <-> `indie folk`
- `ambient americana` <-> `american primitivism`
- `post-punk` <-> `dance-punk`

Purpose:

- Encode valid cross-genre movement.
- Keep movement weighted and explainable.
- Support dynamic playlists without collapsing into hub genres.
- Allow non-identical niche genres to connect through parent families, sibling relationships, or curated bridge edges.
- Activate only when supported by sonic and facet evidence.

Bridge edges are permissions, not guarantees. A bridge edge says "this transition can be valid if the tracks also sound compatible." Sonic, pace, timbre, rhythm, harmony, facet overlap, and transition quality still need to support the move.

## 4. Proposed Data Model

First implementation should be sidecar-only. It must not mutate main metadata tables directly.

The schema below is intended for the AI genre enrichment sidecar database or a new sidecar-owned taxonomy database. Table names are sketches and can be namespaced during implementation, for example `genre_graph_canonical_genres`.

### `canonical_genres`

Canonical registry for family, genre, subgenre, and microgenre terms.

```sql
CREATE TABLE canonical_genres (
    genre_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    kind TEXT NOT NULL CHECK (kind IN ('family', 'genre', 'subgenre', 'microgenre')),
    specificity_score REAL NOT NULL DEFAULT 0.5,
    status TEXT NOT NULL DEFAULT 'active'
        CHECK (status IN ('active', 'deprecated', 'alias_only', 'review')),
    taxonomy_version TEXT NOT NULL
);
```

Notes:

- `specificity_score` should generally increase from family to microgenre.
- `status = 'alias_only'` means the term should normalize to another canonical term and should not appear as a vector dimension.
- `status = 'review'` means the term is not ready for automatic generation behavior.

### `genre_aliases`

Maps source variants and spelling variants to canonical genres.

```sql
CREATE TABLE genre_aliases (
    alias TEXT PRIMARY KEY,
    canonical_genre_id INTEGER NOT NULL REFERENCES canonical_genres(genre_id),
    source TEXT NOT NULL,
    confidence REAL NOT NULL DEFAULT 1.0
);
```

Examples:

- `synth pop` -> `synth-pop`
- `hip-hop` -> `hip hop`
- `lofi indie` -> `lo-fi indie`

Aliases should be source-aware when needed. Some source terms may be too ambiguous to normalize automatically.

### `genre_edges`

Graph edges between canonical genres.

```sql
CREATE TABLE genre_edges (
    source_genre_id INTEGER NOT NULL REFERENCES canonical_genres(genre_id),
    target_genre_id INTEGER NOT NULL REFERENCES canonical_genres(genre_id),
    edge_type TEXT NOT NULL CHECK (
        edge_type IN (
            'is_a',
            'sibling',
            'fusion_of',
            'bridge_to',
            'scene_related',
            'derived_from'
        )
    ),
    weight REAL NOT NULL DEFAULT 1.0,
    confidence REAL NOT NULL DEFAULT 1.0,
    source TEXT NOT NULL,
    notes TEXT,
    PRIMARY KEY (source_genre_id, target_genre_id, edge_type)
);
```

Edge semantics:

- `is_a`: parent or family propagation, such as `jangle pop is_a indie pop`.
- `sibling`: closely related terms under a shared parent.
- `fusion_of`: a genre combines two or more families or genres.
- `bridge_to`: curated cross-genre movement permission.
- `scene_related`: connected by scene, label, geography, or era, but not necessarily direct genre lineage.
- `derived_from`: historical or stylistic derivation.

Edges may be directional. `bridge_to` should normally be directional unless the relationship is clearly symmetric.

### `canonical_facets`

Canonical descriptor/facet registry.

```sql
CREATE TABLE canonical_facets (
    facet_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    facet_type TEXT NOT NULL CHECK (
        facet_type IN (
            'mood',
            'texture',
            'instrumentation',
            'production',
            'era',
            'region',
            'function',
            'vocal',
            'scene'
        )
    ),
    status TEXT NOT NULL DEFAULT 'active'
        CHECK (status IN ('active', 'deprecated', 'alias_only', 'review'))
);
```

Examples:

- `melancholic`, type `mood`
- `lo-fi`, type `production`
- `reverb-heavy`, type `texture`
- `drum machine`, type `instrumentation`
- `1980s`, type `era`
- `anacortes`, type `scene`

### `release_genre_assignments`

Genre assignments for album/release-level enrichment.

```sql
CREATE TABLE release_genre_assignments (
    release_id TEXT,
    album_id TEXT,
    genre_id INTEGER NOT NULL REFERENCES canonical_genres(genre_id),
    assignment_layer TEXT NOT NULL CHECK (
        assignment_layer IN (
            'observed_leaf',
            'inferred_parent',
            'inferred_family',
            'model_prior',
            'human'
        )
    ),
    confidence REAL NOT NULL DEFAULT 0.0,
    source_reliability REAL NOT NULL DEFAULT 0.0,
    evidence_count INTEGER NOT NULL DEFAULT 0,
    rejected_by_user INTEGER NOT NULL DEFAULT 0,
    provenance_json TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (release_id, album_id, genre_id, assignment_layer)
);
```

Implementation notes:

- Existing sidecar release keys may be used in `release_id` if the codebase does not yet have stable release IDs.
- `album_id` should be retained where available for compatibility with existing metadata.
- Human rejects must override all automated evidence.
- `observed_leaf` means a source directly supplied the genre/style.
- `inferred_parent` and `inferred_family` must be derived from graph propagation, not treated as observed facts.

### `release_facet_assignments`

Facet assignments for album/release-level enrichment.

```sql
CREATE TABLE release_facet_assignments (
    release_id TEXT,
    album_id TEXT,
    facet_id INTEGER NOT NULL REFERENCES canonical_facets(facet_id),
    confidence REAL NOT NULL DEFAULT 0.0,
    source TEXT NOT NULL,
    provenance_json TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (release_id, album_id, facet_id, source)
);
```

Facets can come from source tags, official text, model extraction, sonic-derived features, or human review. Source and provenance must be retained because facet trust varies heavily.

### `genre_bridge_rules`

Mode-specific rules for activating bridge movement.

```sql
CREATE TABLE genre_bridge_rules (
    source_genre_id INTEGER NOT NULL REFERENCES canonical_genres(genre_id),
    target_genre_id INTEGER NOT NULL REFERENCES canonical_genres(genre_id),
    required_family_min REAL NOT NULL DEFAULT 0.0,
    required_facet_overlap REAL NOT NULL DEFAULT 0.0,
    required_sonic_similarity REAL NOT NULL DEFAULT 0.0,
    required_transition_quality REAL NOT NULL DEFAULT 0.0,
    mode_allowed TEXT NOT NULL CHECK (
        mode_allowed IN ('strict', 'narrow', 'dynamic', 'discover')
    ),
    notes TEXT,
    PRIMARY KEY (source_genre_id, target_genre_id, mode_allowed)
);
```

Rules should be evaluated against the candidate mode. For example, a `dynamic` bridge rule should not automatically apply in `strict`.

## 5. Vector Representation

The layered graph should produce separate matrices/vectors. Do not collapse these into a single flat genre vector too early.

### `X_genre_leaf_idf`

Specific genre/style/subgenre/microgenre dimensions only.

Use cases:

- Precise similarity.
- Strict and narrow admission.
- DJ bridge vector routing.
- Top-K genre signature extraction.
- IDF-weighted rare taste signals.

Weighting:

- Observed reliable leaf genres should have the strongest weight.
- Human-approved leaf genres should have the strongest trust.
- Model prior genres should be capped unless corroborated.
- Last.fm-only AI-adjudicated unknowns should not enter this matrix automatically.

### `X_genre_family`

Inferred parent and family memberships.

Use cases:

- Neighborhood compatibility.
- Safe bridge region.
- Fallback for sparse releases.
- Broad context diagnostics.

Weighting:

- Lower weight than `X_genre_leaf_idf`.
- Inferred values should be marked and dampened.
- Family overlap alone should not pass `strict` or `narrow` admission.

### `X_genre_bridge`

Bridge affordance vector or graph-derived adjacency vector.

Use cases:

- Permission for cross-genre movement.
- Beam search route expansion.
- Explaining why non-identical genres can connect.

Possible construction:

- For each release, aggregate bridge-adjacent genres from its top leaf genres.
- Weight by bridge edge weight, edge confidence, and source genre confidence.
- Decay by graph distance.
- Keep direct leaf identity separate from bridge affordance.

### `X_facet`

Descriptor/facet vector.

Use cases:

- Mood/texture/production alignment.
- Bridge validation.
- Transition scoring.
- Explanations for cross-family moves.

Facets should not act as genres. They should modify permission and transition quality rather than define the candidate's genre identity.

### `X_genre_legacy`

Optional existing flat vector retained during migration.

Use cases:

- Backward compatibility.
- A/B testing.
- Config flag fallback.
- Guardrail while layered scoring is tuned.

### Parent Propagation

If a release has `jangle pop`, the graph can infer parents and families:

```text
observed_leaf:
  jangle pop

inferred_parent:
  indie pop
  pop rock

inferred_family:
  pop family
  rock family
```

Propagation rules:

- Propagate only through approved `is_a` and selected `fusion_of` edges.
- Store propagated assignments separately from observed source tags.
- Apply confidence decay per edge.
- Apply lower vector weights for inferred parents and families.
- Preserve provenance so diagnostics can explain inferred memberships.

Example propagation pseudocode:

```python
def propagate_genres(observed_leaf_assignments, graph, max_depth=3):
    assignments = []
    for observed in observed_leaf_assignments:
        assignments.append(observed.with_layer("observed_leaf"))

        frontier = [(observed.genre_id, observed.confidence, 0)]
        while frontier:
            genre_id, confidence, depth = frontier.pop()
            if depth >= max_depth:
                continue

            for edge in graph.out_edges(genre_id, edge_type={"is_a", "fusion_of"}):
                propagated_confidence = confidence * edge.weight * edge.confidence * 0.85
                layer = "inferred_family" if edge.target.kind == "family" else "inferred_parent"
                assignments.append(Assignment(
                    genre_id=edge.target.genre_id,
                    layer=layer,
                    confidence=propagated_confidence,
                    provenance={"source": observed.genre_id, "edge": edge.edge_type},
                ))
                frontier.append((edge.target.genre_id, propagated_confidence, depth + 1))

    return dedupe_keep_highest_confidence(assignments)
```

## 6. Scoring Model

The scoring model should expose separate components instead of one flat genre score.

Core components:

- `family_affinity`: broad neighborhood compatibility.
- `niche_similarity`: leaf genre/style similarity, IDF-weighted.
- `facet_alignment`: mood, texture, production, instrumentation, era, or scene alignment.
- `bridge_permission`: graph-validated permission to move between related but non-identical genres.
- `broad_only_penalty`: penalty when overlap is only broad tags.
- `unexplained_jump_penalty`: penalty for family or leaf jumps without bridge, facet, or sonic support.

Candidate-level genre score:

```python
genre_score = (
    w_family * family_affinity
    + w_leaf * niche_similarity
    + w_facet * facet_alignment
    + w_bridge * bridge_permission
    - w_broad_only * broad_only_penalty
    - w_jump * unexplained_jump_penalty
)
```

Suggested starting weights by mode:

| Mode | w_family | w_leaf | w_facet | w_bridge | w_broad_only | w_jump |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| strict | 0.10 | 0.65 | 0.10 | 0.05 | 0.40 | 0.50 |
| narrow | 0.15 | 0.55 | 0.15 | 0.10 | 0.30 | 0.35 |
| dynamic | 0.20 | 0.40 | 0.20 | 0.20 | 0.20 | 0.25 |
| discover | 0.20 | 0.30 | 0.25 | 0.25 | 0.10 | 0.15 |

These are starting points, not final tuning.

Transition-level genre score:

```python
edge_genre_score = (
    local_family_continuity
    + local_leaf_continuity
    + bridge_edge_bonus
    + facet_continuity
    - unexplained_family_jump_penalty
)
```

Broad tags should not be enough by themselves. A candidate with only `rock`, `indie`, `alternative`, or `pop` overlap should not pass `strict` or `narrow` genre admission unless sonic and facet evidence is strong enough to justify a configured exception.

Candidate admission should evaluate both score and explanation:

```python
def admit_candidate(seed_context, candidate, mode):
    components = layered_genre_components(seed_context, candidate)

    if mode in {"strict", "narrow"}:
        if components.broad_only and not components.has_valid_bridge_override:
            return Reject("broad_only", components)

    if components.niche_similarity >= threshold.leaf_min:
        return Admit("leaf_match", components)

    if components.bridge_permission and components.sonic_ok and components.facet_ok:
        return Admit("validated_bridge", components)

    if mode == "discover" and components.sonic_strong and components.facet_strong:
        return Admit("sonic_facet_discovery", components)

    return Reject("insufficient_genre_evidence", components)
```

## 7. Candidate Admission Behavior By Mode

The existing mode presets remain first-class. Layered genre behavior should preserve mode meaning while improving explainability.

### `strict`

Expected behavior:

- Requires strong leaf/niche similarity.
- Family match alone is insufficient.
- Bridge overrides are rare.
- Broad-only candidates should almost always be rejected.
- Sonic strength can help rank admitted candidates but should not routinely bypass missing genre evidence.

### `narrow`

Expected behavior:

- Same family plus good niche similarity can admit.
- Close sibling relationships can admit.
- Limited bridge edges are allowed when sonic and facet checks pass.
- Broad-only candidates should usually be rejected.

### `dynamic`

Expected behavior:

- Allows family and sibling movement.
- Bridge edges are allowed with sonic and facet support.
- Can intentionally move across related scenes or styles.
- Broad-only candidates can pass only when explained by stronger evidence.

### `discover`

Expected behavior:

- Allows broader graph traversal.
- Unknown or broad-only candidates may pass if sonic and facet evidence are strong.
- Bridge paths can be longer, but should still be explainable.
- Noise and human rejects remain excluded.

### `off`

Expected behavior:

- Ignore genre layers entirely.
- Do not compute genre admission gates.
- Sonic and other non-genre systems continue as configured.

### Metadata Quality Behavior Table

| Metadata quality | Definition | strict | narrow | dynamic | discover | off |
| --- | --- | --- | --- | --- | --- | --- |
| rich | Multiple reliable leaf genres plus family/facet support | Admit when leaf match or close bridge exists | Admit with leaf, sibling, or close bridge | Admit with bridge/facet/sonic support | Admit unless other gates fail | Ignore |
| usable | One or two reliable leaf genres, some inferred family | Admit only on strong leaf match | Admit on leaf or close sibling | Admit on family plus bridge/facet support | Admit with sonic/facet support | Ignore |
| broad_only | Only broad tags like `rock`, `pop`, `indie`, `alternative` | Reject | Reject unless exceptional sonic/facet evidence | Conditional, penalized | Conditional, lower penalty | Ignore |
| unknown | No reliable genre layers | Reject by genre | Reject by genre | Usually reject unless sonic/facet discovery path is enabled | Conditional discovery path | Ignore |

## 8. Bridge Validation

A weak leaf match can be overridden only when all required bridge conditions are met:

- There is a known bridge edge or strong graph adjacency.
- Sonic similarity clears the mode-specific threshold.
- Facet overlap clears the mode-specific threshold.
- Transition quality clears the mode-specific threshold.
- No hard conflict exists.

Bridge validation should be local to the transition or seed context, not just global candidate admission. A candidate might be valid near one track and invalid near another.

Pseudocode:

```python
def validate_bridge(from_release, to_release, mode, graph, sonic, facets, transition):
    bridge = graph.best_bridge(from_release.leaf_genres, to_release.leaf_genres, mode)
    if bridge is None:
        return BridgeDecision(False, reason="no_bridge_edge")

    rule = bridge.rule_for_mode(mode)
    if rule is None:
        return BridgeDecision(False, reason="bridge_not_allowed_in_mode")

    if family_affinity(from_release, to_release) < rule.required_family_min:
        return BridgeDecision(False, reason="family_affinity_below_threshold")

    if facet_overlap(from_release, to_release) < rule.required_facet_overlap:
        return BridgeDecision(False, reason="facet_overlap_below_threshold")

    if sonic.similarity < rule.required_sonic_similarity:
        return BridgeDecision(False, reason="sonic_similarity_below_threshold")

    if transition.quality < rule.required_transition_quality:
        return BridgeDecision(False, reason="transition_quality_below_threshold")

    if has_hard_conflict(from_release, to_release):
        return BridgeDecision(False, reason="hard_conflict")

    return BridgeDecision(True, edge=bridge.edge, reason="validated_bridge")
```

Examples:

- `indie pop` -> `synth-pop` is valid when there is pop-family overlap, synth-heavy or danceable facets, and good rhythm/timbre transition.
- `shoegaze` -> `dream pop` is valid when there is shared reverb-heavy texture, compatible density, and a known sibling or bridge edge.
- `avant-folk` -> `indie folk` is valid when acoustic/pastoral facets align and sonic transition quality is acceptable.
- `indie pop` -> `techno` should not be valid unless there is an intermediate dance/electronic bridge path or unusually strong sonic/facet evidence.

Bridge validation should produce diagnostics that explain which edge was used and which thresholds passed or failed.

## 9. Source And Noise Policy

Source evidence should be interpreted by reliability, specificity, and user feedback.

High priority:

- Official artist pages.
- Official label/release pages.
- Bandcamp release pages.
- Press kit or EPK genre text from artist/label.
- Official distributor/store metadata if traceable to the release owner.

Useful but normalized:

- Discogs.
- MusicBrainz.
- Local metadata.
- Existing curated library tags.

Useful but noisy:

- Last.fm.
- User-generated tags from streaming or social systems.

Last.fm policy:

- Last.fm can supply useful corroboration.
- Last.fm broad tags should have low reliability.
- Last.fm-only unknown tags adjudicated by AI should not become automatic genre/style assignments.
- Last.fm-only terms can become provisional only when they are already known canonical genre/style terms or have human review/corroboration.

Human review policy:

- Human rejects override all automated evidence.
- Human approvals can promote a tag into the correct layer.
- Human corrections should be stored as provenance and respected by future rebuilds.

Examples:

| Source tag | Target interpretation |
| --- | --- |
| `rock` | Broad family only |
| `indie` | Broad family/context only unless part of a canonical phrase |
| `jangle pop` | Genre/style |
| `lo-fi` | Facet unless part of a canonical genre phrase |
| `lo-fi bedroom pop` | Genre/style phrase with possible `lo-fi` production facet |
| `japanese` | Region/context facet, not genre |
| `seen live` | Reject/noise |
| `favorite` | Reject/noise |
| `spotify` | Reject/noise |
| `kranky` | Label/scene facet, not genre |
| `female vocals` | Vocal facet |
| `1980s` | Era facet |

Descriptors, places, formats, joke tags, trivia, labels, list tags, and personal tags must not become genre/style tags unless explicitly curated as a genre phrase.

## 10. Integration Points

### AI Genre Enrichment Sidecar DB

The first implementation should be sidecar-only. New taxonomy tables, assignments, facets, bridge edges, and provenance should live outside `data/metadata.db`.

Integration work:

- Normalize enriched source tags into canonical genres and facets.
- Store human rejects and approvals as hard overrides.
- Build release-level layered assignments.
- Keep taxonomy version in every derived assignment and artifact fingerprint.

### Artifact Building

Artifact building should emit layered matrices:

- `X_genre_leaf_idf`
- `X_genre_family`
- `X_genre_bridge`
- `X_facet`
- Optional `X_genre_legacy`

Artifacts must include:

- Taxonomy version.
- Source sidecar fingerprint.
- Graph edge version or hash.
- Matrix dimension vocabularies.

### `candidate_pool.py`

Candidate admission should replace the single flat genre gate with layered components while preserving a config fallback to legacy behavior.

Required changes:

- Compute `family_affinity`.
- Compute `niche_similarity`.
- Compute `facet_alignment`.
- Compute `bridge_permission`.
- Apply `broad_only_penalty`.
- Emit diagnostics for final admission decision.

### `pier_bridge_builder.py`

The pier-bridge beam search should use layered genre routing for transition and route scoring.

Required changes:

- Preserve existing multi-genre vector interpolation.
- Add bridge-aware transition bonuses.
- Penalize unexplained family jumps.
- Use facet continuity to validate cross-genre motion.
- Keep "worst transition matters" diagnostics.

### `segment_pool_builder.py`

Segment pools should receive the same layered metadata and admission reasons used by candidate pools.

Required changes:

- Avoid broad-only pools in strict/narrow modes.
- Expose bridge-enabled candidates separately in diagnostics.
- Preserve recency and diversity behavior.

### Diagnostics And Auditing

Diagnostics should be first-class. If the system cannot explain why genre admitted a candidate or bridge, the behavior should not ship.

Required outputs:

- Per-candidate layered scores.
- Per-transition bridge explanation.
- Source evidence quality.
- Human override involvement.
- Taxonomy version.

### GUI Review Workflow

GUI review should make layer distinctions explicit:

- Genre/style approval.
- Facet approval.
- Reject/noise.
- Alias/canonical correction.
- Human override for release signature.

The GUI should not blindly graduate AI-adjudicated cache terms into canonical vocabulary. Human-reviewed graduation should be the default.

## 11. Diagnostics

Diagnostics must support both admission debugging and playlist quality auditing.

### Candidate Admission Diagnostics

For rejected and admitted candidates, diagnostics should include:

- Candidate track/release ID.
- Source genre signature.
- Candidate genre signature.
- `family_affinity`.
- `niche_similarity`.
- `facet_alignment`.
- `bridge_permission`.
- `broad_only_penalty`.
- `unexplained_jump_penalty`.
- Final genre decision: admitted or rejected.
- Reason code, such as `leaf_match`, `broad_only`, `validated_bridge`, `unknown_metadata`, `human_reject`.
- Source evidence quality.
- Whether human override was involved.
- Taxonomy version.

Example JSON shape:

```json
{
  "candidate_id": "track_123",
  "decision": "rejected",
  "reason": "broad_only",
  "family_affinity": 0.72,
  "niche_similarity": 0.04,
  "facet_alignment": 0.31,
  "bridge_permission": 0.0,
  "broad_only_penalty": 0.40,
  "unexplained_jump_penalty": 0.10,
  "source_evidence_quality": "low",
  "human_override": false,
  "taxonomy_version": "genre-graph-v1"
}
```

### Bridge Transition Diagnostics

For bridge transitions, diagnostics should include:

- From track/release ID.
- To track/release ID.
- From genre signature.
- To genre signature.
- Bridge edge used, if any.
- Graph path used, if any.
- Sonic similarity.
- Facet overlap.
- Transition quality.
- Whether the jump was explained or penalized.
- Failed thresholds when rejected.

Example JSON shape:

```json
{
  "from_id": "track_a",
  "to_id": "track_b",
  "from_signature": ["indie pop", "jangle pop"],
  "to_signature": ["synth-pop", "new wave"],
  "bridge_edge": {
    "source": "indie pop",
    "target": "synth-pop",
    "edge_type": "bridge_to",
    "weight": 0.78
  },
  "sonic_similarity": 0.66,
  "facet_overlap": 0.52,
  "transition_quality": 0.71,
  "decision": "accepted",
  "reason": "validated_bridge"
}
```

## 12. Migration Plan

Migration should be phased. The goal is to add layered behavior without a giant rewrite and without losing the current flat-vector fallback.

### Phase 1: Taxonomy Schema And Registry

Deliverables:

- Sidecar taxonomy schema.
- Canonical genre registry.
- Canonical facet registry.
- Alias normalization.
- Parent/family propagation.
- Taxonomy versioning.
- No generation behavior change.

Validation:

- Existing tests pass.
- New report can show layered assignments for a sample release.
- Human rejects remain respected.

### Phase 2: Layered Vectors And Artifacts

Deliverables:

- Build `X_genre_leaf_idf`.
- Build `X_genre_family`.
- Build `X_genre_bridge`.
- Build `X_facet`.
- Retain optional `X_genre_legacy`.
- Artifact fingerprint includes taxonomy and sidecar graph versions.
- Audit/report command for matrix coverage and sparsity.

Validation:

- Legacy generation remains available.
- Layered artifact can be built without mutating main metadata.
- Diagnostics can show release signatures and propagated families.

### Phase 3: Candidate Admission Uses Layered Score

Deliverables:

- Candidate admission can use layered genre components behind a config flag.
- Broad-only penalty.
- Mode-specific thresholds.
- Fix max-over-seeds behavior where needed so one broad match does not overpower a stronger niche mismatch.
- Candidate diagnostics include layered score components.

Validation:

- Strict/narrow reject broad-only candidates.
- Specific tags retain IDF-like influence.
- Dynamic/discover can admit validated bridge candidates.
- Legacy mode remains available.

### Phase 4: Pier-Bridge Beam Scoring

Deliverables:

- Pier-bridge beam scoring uses family, leaf, bridge, and facet layers.
- Bridge/facet validation for cross-genre moves.
- Explained and unexplained jump diagnostics.
- Transition-quality thresholds integrated with bridge rules.

Validation:

- Multi-seed bridge playlists avoid hub-genre collapse.
- Similar niche genres can connect through shared parents.
- Cross-family jumps require bridge plus sonic/facet support.
- Weakest-edge diagnostics explain bad transitions.

### Phase 5: GUI Review And Feedback Learning

Deliverables:

- GUI review supports genre/style, facet, alias, and reject decisions.
- Human edits/rejects apply to layered assignments.
- Replacement feedback can inform bridge edge weights or review queues.
- AI suggestions for bridge edges require human review before activation.

Validation:

- GUI never blindly promotes AI cache terms into canonical genres.
- Human rejects stay respected across rebuilds.
- Review workflow can explain what changed in taxonomy or assignment layers.

## 13. Acceptance Criteria

Concrete acceptance criteria:

- Flat genre behavior remains available behind a config flag during migration.
- Existing tests still pass.
- Taxonomy versioning exists and is included in derived assignments/artifacts.
- Human rejects remain respected by enrichment, vector building, and generation.
- Broad-only tags cannot dominate strict/narrow generation.
- Specific tags retain IDF-like influence.
- Similar niche genres can connect through shared parents.
- Cross-genre moves require bridge plus sonic/facet support.
- Diagnostics can explain why a candidate passed or failed.
- Bridge transitions can explain the graph edge or path used.
- Source evidence quality is visible in audit output.
- The first implementation is sidecar-only and does not mutate main metadata tables.
- GUI review separates genre/style, facet, alias, and reject decisions.

## 14. Open Questions

Open questions for implementation planning:

- Should family taxonomy be hand-curated YAML first, then promoted to SQLite?
- How much should AI be allowed to suggest bridge edges?
- Should AI-suggested bridge edges default to `review` status until human-approved?
- Should facets be inferred by AI, source tags, sonic features, or all three?
- How should regional scene tags be represented when they are musically meaningful but not genres?
- How should label or scene tags like `kranky`, `c86`, or `pc music` affect bridge routing?
- How should albums with intentionally eclectic genre signatures be handled?
- Should graph weights be globally curated or learned from user replacement feedback?
- Should replacement feedback adjust genre bridge weights, facet weights, or only diagnostics at first?
- How should the system cap graph traversal so dynamic playlists open up without becoming incoherent?
- Should bridge validation happen at candidate admission, transition scoring, or both?
- What UI affordance is needed for users to understand "this is a facet, not a genre"?

## Appendix A: Implementation Guardrails

- Do not replace the existing generator behavior in one step.
- Do not remove legacy genre vectors until layered behavior has clear diagnostics and test coverage.
- Do not let broad tags act as strong evidence in strict/narrow modes.
- Do not promote AI-adjudicated unknown tags into canonical genres without review.
- Do not treat facets as genre/style dimensions.
- Do not mutate `data/metadata.db` during early phases.
- Preserve sonic and genre fusion. Layered genre should improve the interaction, not replace sonic evidence.

## Appendix B: Example Layered Signatures

### Duster - `Stratosphere`

Leaf candidates:

- `slowcore`
- `space rock`
- `dream pop`
- `post-rock`

Families:

- `rock family`
- `pop family`
- `ambient/experimental neighborhood`

Facets:

- `lo-fi`
- `melancholic`
- `minimal`
- `reverb-heavy`

Bridge affordances:

- `slowcore` <-> `dream pop`
- `dream pop` <-> `shoegaze`
- `space rock` <-> `post-rock`

### Mount Eerie - `Sauna`

Leaf candidates:

- `indie folk`
- `avant-folk`
- `psychedelic folk`
- `drone`

Families:

- `folk family`
- `experimental family`
- `ambient/experimental neighborhood`

Facets:

- `lo-fi`
- `bleak`
- `minimal`
- `acoustic`

Bridge affordances:

- `avant-folk` <-> `indie folk`
- `drone` <-> `ambient`
- `psychedelic folk` <-> `freak folk`

### Stereolab - `Dots and Loops`

Leaf candidates:

- `indie pop`
- `krautrock`
- `lounge pop`
- `post-rock`
- `electronic pop`

Families:

- `pop family`
- `rock family`
- `electronic family`
- `experimental family`

Facets:

- `motorik`
- `retro-futurist`
- `warm`
- `synth-heavy`

Bridge affordances:

- `indie pop` <-> `synth-pop`
- `krautrock` <-> `post-rock`
- `lounge pop` <-> `art pop`

