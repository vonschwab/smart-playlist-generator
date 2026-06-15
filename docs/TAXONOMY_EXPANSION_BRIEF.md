# Taxonomy Expansion Brief

**File:** `data/layered_genre_taxonomy.yaml`
**Version being expanded:** `0.1.1-layered-seed-reviewed`
**Purpose of this doc:** Brief for an AI session dedicated to expanding the taxonomy to fill coverage gaps discovered during a 30-album enrichment test.

---

## What the taxonomy is

A structured, layered genre system used to assign genres to albums and drive playlist similarity routing. It is **not a flat tag list** — it's a graph where each genre entry has:

- A **kind** (`family`, `umbrella`, `genre`, `subgenre`, `microgenre`, `facet`, `alias`, `reject`)
- A **role** (`family`, `umbrella`, `leaf`, `modifier`, `context`, `alias`, `reject`)
- **Parent edges** that declare relationships to parents (with edge types: `is_a`, `family_context`, `scene_adjacent`, `fusion_of`, `style_modifier`, `bridge_to`, `alias_of`)
- A **specificity_score** (0.0–1.0; families ~0.05, subgenres ~0.75–0.88)
- A **status** (`active`, `review`, `deprecated`, `alias_only`, `rejected`)
- A **source_policy** (`broad_context`, `source_supported`, `review_context`, `facet`, etc.)

The taxonomy lives in YAML as a list of `records`. There are also `bridge_rules` (cross-genre routing rules) and `facets` (texture/mood/production/instrumentation modifiers).

**Key constraint:** No genre/family relationship should be inferred from token containment alone (e.g., `dream pop` must not imply `pop` as a parent unless an explicit curated edge exists). Every parent edge must be deliberately placed.

---

## Current taxonomy at a glance

### Families (14 total)

| Family | Notes |
|--------|-------|
| `rock` | Broad |
| `pop` | Broad |
| `electronic` | Broad |
| `folk` | Broad |
| `jazz` | Broad |
| `hip hop` | Broad |
| `punk` | Broad |
| `metal` | Broad |
| `ambient/new age` | Broad |
| `r&b/soul` | Broad |
| `country/roots` | Broad |
| `classical/modern composition` | Broad — **no leaves attached** |
| `indie/alternative` | Curated context family |
| `experimental/avant-garde` | Broad context |

### Active leaf genres (kind = genre/subgenre/microgenre, status = active)

**Indie / Alternative / Pop-Rock cluster:**
`indie rock`, `alternative rock`, `indie pop`, `jangle pop`, `twee pop`, `noise pop`, `art pop`, `experimental pop`, `avant-garde pop`, `synth-pop`, `dream pop`, `shoegaze`, `slowcore`, `post-rock`, `space rock`, `slacker rock`, `psychedelic rock`, `experimental rock`, `neo-psychedelia`, `psychedelic pop`, `ambient pop`, `art rock`, `progressive rock`, `noise rock`, `garage rock`, `post-punk`, `dance-punk`, `new wave`, `krautrock`, `indietronica`

**Folk / Roots cluster:**
`indie folk`, `folk rock`, `psychedelic folk`, `avant-folk`, `american primitivism`, `ambient americana`, `americana`, `country`

**Electronic cluster:**
`downtempo`, `dub`, `ambient dub`, `electronica` (umbrella)

**Jazz cluster:**
`modal jazz`, `jazz fusion`, `avant-garde jazz`, `bebop`, `post-bop`

**R&B/Soul cluster:**
`funk` (only leaf; borders jazz)

**Hip hop cluster:**
`lo-fi hip hop` (status: review — only entry)

**Metal cluster:**
`black metal` (only entry)

**Facets (25):**
`lo-fi`, `noisy`, `pastoral`, `acoustic`, `synth-heavy`, `danceable`, `melancholic`, `minimal`, `warm`, `reverb-heavy`, `female vocals`, `instrumental`, `drum machine`, `diy`, `drone`, `spacious`, `motorik`, `bright`, `guitar-forward`, `slow`, `1980s` (era), `canada`, `japanese`, `welsh` (regions), `c86` (scene)

**Review-status leaves** (valid but not yet fully admitted):
`space rock revival`, `college rock`, `drone rock`, `kosmische musik`, `pop rock`

**Total: ~76 canonical genres + 25 facets + ~30 aliases**

---

## Gaps discovered during enrichment testing

The following were found by running full AI genre enrichment on 30 albums spanning multiple genres, then trying to build graph assignments. Albums with 0 or near-0 assignments reveal where the taxonomy is completely missing a territory.

---

### Gap 1 — R&B / Soul: entire leaf layer absent

**Affected albums:** Aretha Franklin — *Amazing Grace* (0 assignments), Al Green — *Call Me* (1 family-only assignment)

The `r&b/soul` family has no leaves under it except `funk` (which is more jazz-adjacent). The AI enrichment correctly identified these albums as soul/gospel, but the taxonomy had nowhere to put those terms.

**Missing terms needed (at minimum):**

| Term | Kind | Notes |
|------|------|-------|
| `soul` | genre/umbrella | Broad soul umbrella — parent: `r&b/soul` |
| `gospel` | genre | Parent: `r&b/soul` |
| `classic soul` | subgenre | Parent: `soul` |
| `deep soul` | subgenre | Parent: `soul` |
| `southern soul` | subgenre | Parent: `soul` |
| `soul jazz` | subgenre | Parent: `jazz` + `soul`; bridge to r&b/soul |
| `contemporary r&b` | genre | Parent: `r&b/soul` |
| `blues` | family or umbrella | Needed as its own node — `soul blues`, `gospel blues` want a home |

**Note:** `soul` as a standalone term is currently absent entirely. The enrichment pipeline generated it as an AI suggestion for Aretha Franklin but it mapped to nothing.

---

### Gap 2 — Jazz: core mainstream subgenres missing

**Affected albums:** Ahmad Jamal — *Ahmad's Blues* (1 family-only), A Tribe Called Quest — *Midnight Marauders* (1 family-only), Antonio Carlos Jobim — *Wave* (sparse)

The taxonomy has `modal jazz`, `bebop`, `post-bop`, `jazz fusion`, `avant-garde jazz` — these are specific subgenres. But mainstream/accessible jazz forms that AI sources consistently apply are absent.

**Missing terms needed:**

| Term | Kind | Notes |
|------|------|-------|
| `hard bop` | subgenre | Parent: `jazz`; adjacent to `bebop` and `post-bop` |
| `cool jazz` | subgenre | Parent: `jazz` |
| `soul jazz` | subgenre | Parent: `jazz`; bridge to `r&b/soul` |
| `piano jazz` | genre | Parent: `jazz`; useful for Ahmad Jamal context |
| `straight-ahead jazz` | genre/umbrella | Parent: `jazz`; broad but real — covers hard bop / post-bop territory |
| `bossa nova` | genre | Parent: `jazz`; bridge to `folk` or `classical/modern composition` |
| `samba` | genre | Parent: `country/roots` or standalone; related to `bossa nova` |
| `latin jazz` | genre | Parent: `jazz`; bridge to bossa nova / samba |
| `jazz blues` | genre | Parent: `jazz` + bridge to `blues` |
| `swing` | genre | Parent: `jazz` |
| `big band` | genre | Parent: `jazz` |

---

### Gap 3 — Hip hop: almost completely empty

**Affected albums:** A Tribe Called Quest — *Midnight Marauders* (1 family-only)

The `hip hop` family has only `lo-fi hip hop` (status: review) as a leaf. That's one entry for an entire genre family. AI enrichment produces specific hip hop subgenres that have nowhere to go.

**Missing terms needed:**

| Term | Kind | Notes |
|------|------|-------|
| `rap` | umbrella | Alias-to or parent to hip hop subgenres; many sources use "rap" as primary tag |
| `east coast hip hop` | subgenre | Parent: `hip hop` |
| `west coast hip hop` | subgenre | Parent: `hip hop` |
| `conscious hip hop` | subgenre | Parent: `hip hop` |
| `alternative hip hop` | genre | Parent: `hip hop`; bridge to `indie/alternative` |
| `abstract hip hop` | subgenre | Parent: `hip hop`; adjacent to `alternative hip hop` |
| `jazz rap` | subgenre | Parent: `hip hop`; bridge to `jazz` |
| `hardcore hip hop` | subgenre | Parent: `hip hop` |
| `boom bap` | subgenre | Parent: `hip hop`; classic ATCQ production style |
| `instrumental hip hop` | subgenre | Parent: `hip hop` |
| `trap` | genre | Parent: `hip hop` |
| `uk hip hop` | subgenre | Parent: `hip hop`; relevant for Archy Marshall territory |
| `grime` | genre | Parent: `hip hop`; London-specific; Archy Marshall territory |

---

### Gap 4 — Electronic: IDM, techno, and modern electronic missing

**Affected albums:** Aphex Twin — *Syro* (sparse), Autechre — *Amber* (sparse), AFX — *Analord 01* (sparse)

The electronic family has only `downtempo`, `dub`, `ambient dub`, and `electronica` (umbrella) as usable leaves. It's missing the bulk of electronic music taxonomy.

**Missing terms needed:**

| Term | Kind | Notes |
|------|------|-------|
| `IDM` | genre | "Intelligent dance music"; parent: `electronic`; Aphex Twin, Autechre |
| `ambient techno` | subgenre | Parent: `electronic` + `ambient/new age`; Aphex Twin territory |
| `techno` | genre | Parent: `electronic` |
| `acid techno` | subgenre | Parent: `techno`; AFX |
| `acid house` | subgenre | Parent: `electronic` |
| `house` | genre | Parent: `electronic` |
| `electro` | genre | Parent: `electronic`; distinct from electronica |
| `breakbeat` | genre | Parent: `electronic` |
| `jungle` | genre | Parent: `electronic` |
| `drum and bass` | genre | Parent: `electronic` |
| `drill and bass` | subgenre | Parent: `drum and bass`; Aphex Twin/AFX |
| `glitch` | genre | Parent: `electronic` + `experimental/avant-garde` |
| `microsound` | subgenre | Parent: `electronic` + `experimental/avant-garde` |
| `electroacoustic` | genre | Parent: `electronic` + `classical/modern composition` |
| `modular synthesis` | facet | Production facet |

---

### Gap 5 — Punk / Hardcore: leaf layer thin

**Affected albums:** 48 Chairs — *70% Paranoid* (sparse), Angry Angles — *Angry Angles* (sparse)

The `punk` family has `post-punk`, `dance-punk`, `garage rock` (with punk bridge). But it's missing the core punk subgenres.

**Missing terms needed:**

| Term | Kind | Notes |
|------|------|-------|
| `punk rock` | genre | Parent: `punk`; or alias — but currently no canonical node |
| `hardcore punk` | genre | Parent: `punk` |
| `post-hardcore` | genre | Parent: `punk` + `indie/alternative` |
| `emo` | genre | Parent: `punk` + `indie/alternative`; American Football |
| `midwest emo` | subgenre | Parent: `emo` |
| `anarcho-punk` | subgenre | Parent: `punk` |
| `no wave` | genre | Parent: `experimental/avant-garde` + `punk`; scene_adjacent to post-punk |
| `math rock` | genre | Parent: `rock`; adjacent to `post-rock` and `emo`; American Football |

---

### Gap 6 — Classical / Composition: zero leaves

The `classical/modern composition` family has no leaf genres under it at all. It exists as a family but is a dead end.

**Missing terms needed (start conservative):**

| Term | Kind | Notes |
|------|------|-------|
| `modern classical` | genre | Parent: `classical/modern composition` |
| `neoclassical` | genre | Parent: `classical/modern composition` + `ambient/new age` |
| `contemporary classical` | genre | Parent: `classical/modern composition` |
| `minimalist composition` | subgenre | Parent: `classical/modern composition`; Philip Glass territory |
| `chamber music` | genre | Parent: `classical/modern composition` |
| `orchestral` | genre | Parent: `classical/modern composition` |
| `art song` | genre | Parent: `classical/modern composition`; ANOHNI territory |

---

### Gap 7 — Common pop/rock subgenres: review terms that never resolve

These terms appeared frequently in AI enrichment output as `review` (not accepted) because they either aren't in the taxonomy or are flagged `status: review`. They're real, well-established genres.

| Term | Current status | What's needed |
|------|---------------|---------------|
| `chamber pop` | Absent | Add as subgenre; parent: `pop` + `indie/alternative`; adjacent to `baroque pop` |
| `power pop` | Absent | Add as subgenre; parent: `pop` + `rock` + `indie pop` |
| `baroque pop` | Absent | Add as subgenre; parent: `pop`; adjacent to `chamber pop` |
| `pop rock` | Present but `status: review` | Promote to `active` or clarify when it should accept |
| `indie dance` | Absent | Parent: `indie/alternative` + `electronic` |
| `sophisti-pop` | Absent | Parent: `pop`; 80s polished pop |

---

### Gap 8 — R&B / Soul adjacents: avant-soul and related

**Affected albums:** ANOHNI — *My Back Was a Bridge for You to Cross* (0 assignments)

ANOHNI sits at the intersection of avant-garde, soul, and art song. The taxonomy has nothing for this territory.

**Missing terms needed:**

| Term | Kind | Notes |
|------|------|-------|
| `avant-soul` | subgenre | Parent: `r&b/soul` + `experimental/avant-garde` |
| `art song` | genre | Parent: `classical/modern composition` or `pop` |
| `torch song` | subgenre | Parent: `r&b/soul` or `pop` |
| `neo soul` | genre | Parent: `r&b/soul` |

---

### Gap 9 — Folk: freak folk, dream folk, neofolk

**Affected albums:** Animal Collective — *Fall Be Kind* (rich, but some terms fell to review)

The `folk` family has `indie folk`, `folk rock`, `psychedelic folk`, `avant-folk`, `american primitivism`, `ambient americana`. Missing:

| Term | Kind | Notes |
|------|------|-------|
| `freak folk` | subgenre | Parent: `folk` + `experimental/avant-garde`; Animal Collective adjacent |
| `dream folk` | subgenre | Parent: `folk` + `ambient/new age`; adjacent to `indie folk` |
| `neofolk` | genre | Parent: `folk` + `experimental/avant-garde`; distinct from freak folk |
| `sadcore` | genre | Parent: `folk` + `indie/alternative`; adjacent to `slowcore` |

---

### Gap 10 — French / Global pop styles

**Affected albums:** Air — *Moon Safari* (sparse), Antonio Carlos Jobim — *Wave* (sparse)

| Term | Kind | Notes |
|------|------|-------|
| `french pop` | genre | Parent: `pop`; distinct cultural scene |
| `french house` | subgenre | Parent: `house` (electronic); Air territory |
| `space age pop` | genre | Parent: `pop` + `ambient/new age`; lounge/exotica aesthetic |
| `lounge` | genre | Parent: `pop` or `ambient/new age`; context-heavy but real |
| `bossa nova` | genre | (also listed in Gap 2; Jobim) |
| `mpb` | genre | Música Popular Brasileira; parent: `country/roots` or standalone |

---

### Gap 11 — Blues: entire family absent

`blues` as a family or umbrella does not exist. Several genres want it as a parent:

| Term | Kind | Notes |
|------|------|-------|
| `blues` | family or umbrella | Required to root `soul blues`, `jazz blues`, `electric blues`, etc. |
| `electric blues` | genre | Parent: `blues` |
| `soul blues` | subgenre | Parent: `blues` + `r&b/soul` |
| `gospel blues` | subgenre | Parent: `blues` + `r&b/soul` |
| `delta blues` | subgenre | Parent: `blues` |
| `chicago blues` | subgenre | Parent: `blues` |

---

## Summary of gaps by severity

### Critical (0-assignment albums, major families essentially empty)
1. **R&B/Soul leaf layer** — `soul`, `gospel`, `classic soul`, `deep soul` needed
2. **Electronic depth** — `IDM`, `techno`, `acid techno`, `glitch`, `electro`, `breakbeat`, `drum and bass` needed
3. **Hip hop leaf layer** — entire subgenre tree missing (boom bap, jazz rap, east coast, alternative hip hop)
4. **Classical/composition** — `modern classical`, `neoclassical`, `minimalist composition` needed
5. **Blues** — family node + core leaves needed

### High (thin coverage, common enrichment terms falling off)
6. **Jazz mainstream** — `hard bop`, `cool jazz`, `soul jazz`, `bossa nova`, `straight-ahead jazz`
7. **Punk depth** — `punk rock`, `hardcore punk`, `emo`, `math rock`, `post-hardcore`
8. **Common pop subgenres** — `chamber pop`, `power pop`, `baroque pop` (all absent)
9. **Avant-soul / ANOHNI territory** — `avant-soul`, `neo soul`, `art song`

### Medium (enrichment noise, review terms not resolving)
10. Folk extensions — `freak folk`, `dream folk`, `sadcore`
11. French/global — `french pop`, `space age pop`, `lounge`, `bossa nova`, `mpb`
12. UK specific — `grime`, `uk hip hop`, `no wave`

---

## How to add a new record

Every record added must follow this YAML structure. Copy and adapt from an existing similar entry:

```yaml
- name: <canonical name, lowercase>
  kind: <family|umbrella|genre|subgenre|microgenre|facet|alias|reject>
  role: <family|umbrella|leaf|modifier|context|alias|reject>
  status: <active|review>
  facet_type: null
  specificity_score: <0.0–1.0; families=0.05, umbrellas=0.25–0.45, genres=0.55–0.70, subgenres=0.70–0.88>
  canonical_target: null
  parent_edges:
  - target: <parent name>
    edge_type: <is_a|family_context|scene_adjacent|fusion_of|style_modifier|bridge_to>
    weight: <0.0–1.0>
    confidence: 0.85
    notes: null
  secondary_roles: []
  reject_reason: null
  alias_policy: null
  source_policy: <broad_context|source_supported|review_context|facet>
  possible_context_target: null
  notes: <one sentence or null>
```

**Rules:**
- Use `is_a` for a direct subgenre-of relationship (e.g. `hard bop is_a jazz`)
- Use `family_context` for broad family membership (weight typically 0.5–0.75)
- Use `scene_adjacent` for related-but-not-parent genres (weight typically 0.25–0.55)
- Leaf genres that are well-established get `status: active`; uncertain or niche ones get `status: review`
- `specificity_score` for a new subgenre should be ~0.72–0.82; for a new umbrella ~0.35–0.55
- `source_policy: source_supported` for most leaf genres; `broad_context` only for family-level entries

**Alias records** (for spelling variants, hyphens, etc.) look like:

```yaml
- name: <alternate spelling>
  kind: alias
  role: alias
  status: alias_only
  facet_type: null
  specificity_score: null
  canonical_target: <canonical name>
  parent_edges: []
  secondary_roles: []
  reject_reason: null
  alias_policy:
    type: plain
    notes: Orthographic alias only.
  source_policy: null
  possible_context_target: null
  notes: Orthographic alias only.
```

---

## What NOT to add

- Don't add **artist names**, **release titles**, **place names** as genres (use `reject` records if they show up as noise)
- Don't add **era tags** as genres (`1990s`, `1980s` → use facet/era_context if needed)
- Don't add **platform names** or **user-list tags** (`spotify`, `seen live`, `favorites`)
- Don't add **pure adjective/descriptors** that aren't established genre names (`warm`, `sad` → these belong as facets if needed)
- Don't add `indie` as a standalone genre — it's already explicitly rejected as source_noise
- Don't infer parent relationships from word containment — `dream pop` does NOT get a `pop` parent just because the word "pop" is in it. Every edge must be deliberately placed

---

## Reference: edge types and when to use them

| Edge type | Meaning | Example |
|-----------|---------|---------|
| `is_a` | Direct subgenre-of | `hard bop is_a jazz` |
| `family_context` | Member of broader family | `shoegaze family_context rock` |
| `scene_adjacent` | Related scene, not parent | `shoegaze scene_adjacent dream pop` |
| `fusion_of` | Genre that blends two others | `ambient dub fusion_of ambient + dub` |
| `style_modifier` | Modified by another style | `experimental rock style_modifier experimental` |
| `bridge_to` | Cross-genre routing bridge | `indie pop bridge_to synth-pop` (with sonic/facet conditions) |
| `alias_of` | Different name for same thing | (use `kind: alias` records instead) |
