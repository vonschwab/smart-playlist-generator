# Insights-Report Follow-Ups (2026-07-06) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Institutionalize the four accepted recommendations from the 2026-07-06 insights review: a prior-art-before-design rule (project CLAUDE.md), collapse/health-claim triggers in the evaluation-methodology skill, a new genre-adjudication skill, and a sub-agent visibility rule (global CLAUDE.md).

**Architecture:** Pure documentation/skill changes — no code, no tests to run. Three files live in this repo (`CLAUDE.md`, two files under `.claude/skills/`, which IS git-tracked); one file is the user's global `C:\Users\Dylan\.claude\CLAUDE.md` (outside the repo — edited, never committed). Each repo task ends in its own commit.

**Tech Stack:** Markdown + YAML frontmatter (skills). Python one-liner only to validate frontmatter parses.

## Global Constraints

- **Shared-checkout discipline (this session is on the shared main checkout):** other sessions have in-flight work (`git status` shows modified `src/playlist/pier_bridge*`, `data/layered_genre_taxonomy.yaml`, `docs/superpowers/plans/2026-07-04-always-publish-policy-flip.md`, etc.). NEVER `git add -A` / `git add -u`. Stage explicit paths only, re-check `git status` immediately before each commit, and commit with `git commit --only <paths> -m ...` so another session's staged files are never swept in.
- **Do not touch** any file listed as modified in `git status` that this plan does not name. They are another session's work.
- Never use `--no-verify`; if a hook fails, diagnose it.
- The global `C:\Users\Dylan\.claude\CLAUDE.md` is NOT in this repo — Task 4 edits it but has no commit step.
- Skill files follow the repo's existing convention: YAML frontmatter with `name` + `description` only, then markdown body (see `.claude/skills/evaluation-methodology/SKILL.md`).
- Match the existing prose style of each file being edited (bold-lead bullets in CLAUDE.md session discipline; numbered checklist + red-flag bullets in the skill).
- Sub-agent executors: use `model: sonnet` per the global sub-agent model policy.

---

### Task 1: Prior-art-before-design rule in project CLAUDE.md

**Files:**
- Modify: `CLAUDE.md` (repo root, "Session discipline" section, ~line 35)
- Commit also includes: `docs/superpowers/plans/2026-07-06-insights-followups.md` (this plan)

**Interfaces:**
- Consumes: nothing.
- Produces: a new session-discipline bullet later tasks may reference by name ("Prior art before design").

**Context for the implementer:** The 2026-07-06 insights review found Claude repeatedly designed fixes from first principles (e.g. a fusion-policy fix) instead of reading existing handoff/design docs, causing user interrupts. The existing bullet "Project concepts: search before answering" covers *questions*; nothing covers *designs*. This rule fills that gap. The `reuse-first` skill covers code reuse; this bullet covers decisions/design prior art — keep them distinct.

- [ ] **Step 1: Apply the edit**

In `CLAUDE.md`, find this exact bullet in the "Session discipline" list:

```markdown
- **Project concepts: search before answering.** Asked what something project-specific does, grep code/docs first — never infer from the name.
```

Replace it with (i.e. append the new bullet directly after it):

```markdown
- **Project concepts: search before answering.** Asked what something project-specific does, grep code/docs first — never infer from the name.
- **Prior art before design.** Before designing any fix, feature, or investigation for a subsystem, check the auto-memory index (MEMORY.md) and `docs/` (HANDOFF_*, INCIDENT_*, `superpowers/specs/`, `superpowers/plans/`) for prior decisions on it — design from prior art, not first principles. First-principles redesigns of already-decided subsystems are a recurring interrupt cause (insights review 2026-07-06). The `reuse-first` skill covers the code half of this rule.
```

- [ ] **Step 2: Verify**

Run: `git -C C:\Users\Dylan\Desktop\PLAYLIST_GENERATOR_V3 diff CLAUDE.md`
Expected: exactly one hunk, adding exactly one bullet line after "Project concepts: search before answering." No other changes.

- [ ] **Step 3: Commit (explicit paths only)**

```bash
git -C C:\Users\Dylan\Desktop\PLAYLIST_GENERATOR_V3 status --short
# confirm no unexpected staged files, then:
git -C C:\Users\Dylan\Desktop\PLAYLIST_GENERATOR_V3 add CLAUDE.md docs/superpowers/plans/2026-07-06-insights-followups.md
git -C C:\Users\Dylan\Desktop\PLAYLIST_GENERATOR_V3 diff --cached --name-only
# expected: exactly these 2 files, then:
git -C C:\Users\Dylan\Desktop\PLAYLIST_GENERATOR_V3 commit --only -m "docs(claude-md): prior-art-before-design session rule (insights 2026-07-06)" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>" -- CLAUDE.md docs/superpowers/plans/2026-07-06-insights-followups.md
```

---

### Task 2: Collapse/health-claim triggers in the evaluation-methodology skill

**Files:**
- Modify: `.claude/skills/evaluation-methodology/SKILL.md` (3 edits: description line 3, pre-flight checklist after item 6 ~line 17, red flags after ~line 33)

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces: pre-flight items **7** and **8** (numbering continues the existing 1–6).

**Context for the implementer:** The 2026-06-24 "MERT collapse" false alarm (see `docs/INCIDENT_2026-06-24_MERT_COLLAPSE_REBUTTAL.md` and memory `project_mert_collapse_incident`) happened AFTER this skill existed, because the skill's triggers only cover A-vs-B evaluations — a "space X has collapsed" health claim didn't fire it. Three specific process failures: (a) two probes disagreed (rank 31 vs rank 21,525 on the same bytes) and the scary one was trusted without reconciling; (b) no random-pair null baseline was ever computed; (c) the probe measured raw cosine against a fixed target instead of the runtime's own metric (rank via max-over-seed-tracks) — the cosine drop was the *intended* effect of whitening. The skill's own "Maintenance protocol" section mandates exactly this kind of extension: add the failure to the pre-flight checklist and a matching red flag.

- [ ] **Step 1: Extend the frontmatter description (trigger fix)**

Find this exact line (line 3):

```
description: Use when designing, running, or reporting any similarity/ranking/A-B evaluation — sonic or genre audition, whitening or transform validation, artifact A/B comparison, threshold or floor calibration — or before presenting any quantitative conclusion that one embedding, matrix, or config beats another. Also use when building or extending an evaluation/audition harness.
```

Replace with:

```
description: Use when designing, running, or reporting any similarity/ranking/A-B evaluation — sonic or genre audition, whitening or transform validation, artifact A/B comparison, threshold or floor calibration — or before presenting any quantitative conclusion that one embedding, matrix, or config beats another. Also use when building or extending an evaluation/audition harness, and BEFORE reporting that a space, artifact, or metric has collapsed, regressed, or silently degraded — a health/collapse claim is an evaluation and needs the same controls.
```

- [ ] **Step 2: Add pre-flight items 7 and 8**

Find this exact text (checklist item 6):

```markdown
6. **Check the metric isn't circular.** Don't validate a space with a metric computed in that same space (genre QC scored inside the same enriched-genre space said everything was fine while the tags were junk). At least one arm of the evidence must be independent: human ears, held-out labels, a different modality.
```

Replace with:

```markdown
6. **Check the metric isn't circular.** Don't validate a space with a metric computed in that same space (genre QC scored inside the same enriched-genre space said everything was fine while the tags were junk). At least one arm of the evidence must be independent: human ears, held-out labels, a different modality.
7. **Reconcile contradictory probes before trusting either.** If two measurements of the same space disagree, the contradiction IS the finding — resolve it before reporting anything. The 2026-06-24 "MERT collapse" false alarm shipped a catastrophic conclusion from the scary probe (rank 21,525) while a healthy probe (rank 31) on the same bytes sat unreconciled; the scary probe had a selection bug (first-track instead of max-over-seed-tracks).
8. **Health claims need a null baseline and the runtime's own metric.** "Collapsed/degraded" is an evaluation: compare same-artist (or golden-neighbor) stats against a **random-pair baseline** (the missing control in the MERT false alarm — healthy gap was +0.214 vs random), and measure what the runtime actually consumes (rank via max over seed tracks), never raw cosine against a fixed target — whitening intentionally recenters cosines to ~0, so a cosine "drop" is expected, not damage.
```

- [ ] **Step 3: Add matching red flags**

Find this exact text (last red-flag bullet):

```markdown
- "Preliminary / directionally promising" framing of an unsound result → an unsound result is not reportable at any confidence level.
```

Replace with:

```markdown
- "Preliminary / directionally promising" framing of an unsound result → an unsound result is not reportable at any confidence level.
- "Two probes disagree — the alarming one must be right" → reconcile first; same bytes + corrected method may be healthy.
- "Raw cosine dropped after whitening → collapse" → whitening recenters cosines; check rank/discrimination against a random-pair baseline instead.
- "One ad-hoc number justifies an incident report" → a collapse/regression claim gets the full pre-flight plus a second independent probe before any incident doc, memory, or recovery plan is written.
```

- [ ] **Step 4: Verify frontmatter still parses and diff is clean**

Run:

```bash
python -c "import yaml,io; t=io.open(r'C:\Users\Dylan\Desktop\PLAYLIST_GENERATOR_V3\.claude\skills\evaluation-methodology\SKILL.md',encoding='utf-8').read(); fm=t.split('---')[1]; d=yaml.safe_load(fm); print('OK:', sorted(d.keys()))"
```

Expected: `OK: ['description', 'name']`

Run: `git -C C:\Users\Dylan\Desktop\PLAYLIST_GENERATOR_V3 diff .claude/skills/evaluation-methodology/SKILL.md`
Expected: 3 hunks (description, checklist, red flags); no other sections touched.

- [ ] **Step 5: Commit (explicit paths only)**

```bash
git -C C:\Users\Dylan\Desktop\PLAYLIST_GENERATOR_V3 status --short
git -C C:\Users\Dylan\Desktop\PLAYLIST_GENERATOR_V3 add .claude/skills/evaluation-methodology/SKILL.md
git -C C:\Users\Dylan\Desktop\PLAYLIST_GENERATOR_V3 diff --cached --name-only
# expected: exactly this 1 file, then:
git -C C:\Users\Dylan\Desktop\PLAYLIST_GENERATOR_V3 commit --only -m "docs(skills): eval-methodology fires on collapse/health claims (MERT false-alarm lessons)" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>" -- .claude/skills/evaluation-methodology/SKILL.md
```

---

### Task 3: Create the genre-adjudication skill

**Files:**
- Create: `.claude/skills/genre-adjudication/SKILL.md`

**Interfaces:**
- Consumes: the contract in `src/ai_genre_enrichment/album_adjudicator.py` (`ADJUDICATOR_INSTRUCTIONS` lines 31–48, `ADJUDICATOR_RESPONSE_SCHEMA` lines 67–103, `LAYERS`/`FACET_TYPES` lines 23–28). The skill MIRRORS that contract; the .py file stays the single source of truth for the pipeline path.
- Produces: a session-invocable skill standardizing the interactive adjudication sessions (23+ sessions in the last month re-specified this schema implicitly each time).

**Context for the implementer:** Dylan pastes batches of 2–16 release payloads into Claude Code sessions and expects structured JSON back, matching the pipeline's `album-adjudicator-response-v1` schema. The skill encodes: the schema, the core rules (specific genres only, facets separated, user-file-tag floor, escalate over guess), batch output format, and the "verify a term against the taxonomy before rejecting it" lesson (Kankyo Ongaku). It must NOT duplicate the pipeline's evolution — it points at the .py source of truth and instructs re-checking on drift.

- [ ] **Step 1: Write the skill file**

Create `.claude/skills/genre-adjudication/SKILL.md` with exactly this content:

````markdown
---
name: genre-adjudication
description: Use when the user pastes one or more album/release payloads for genre adjudication or classification — batches of JSON payloads with artist/album/evidence fields expecting structured genre JSON back — or asks to adjudicate, classify, or review the genre identity of specific releases in an interactive session.
---

# Genre adjudication (interactive sessions)

Dylan pastes batches of release payloads (typically 2–16) and expects structured JSON conforming to the pipeline's `album-adjudicator-response-v1` contract. **Source of truth:** `src/ai_genre_enrichment/album_adjudicator.py` (`ADJUDICATOR_INSTRUCTIONS`, `ADJUDICATOR_RESPONSE_SCHEMA`). If this skill and that file disagree, the .py file wins — update this skill.

## Output contract

Return ONE fenced JSON array, one element per payload, **in payload order**. Each element is one response object:

```json
{
  "release_key": "<echo the payload's album_id or release_key if present, else \"<artist> — <album>\">",
  "genres": [{"term": "shoegaze", "confidence": 0.9, "layer": "core"}],
  "facets": [{"term": "female vocals", "facet_type": "vocal"}],
  "escalate": false,
  "escalate_reason": "",
  "overall_confidence": 0.85,
  "warnings": []
}
```

- `layer` ∈ {`core`, `secondary`}. Core = primary identity (~2–4); secondary = real but lesser element. Total genres ~3–6; fewer for focused releases.
- `facet_type` ∈ {`mood`, `texture`, `instrumentation`, `production`, `era`, `region`, `function`, `vocal`, `scene`, `format`, `rhythm`} (the taxonomy's facet enum).
- `release_key` is an interactive-batch convenience for unambiguous mapping — the strict pipeline schema omits it. If the paste specifies its own output format, that wins.
- No prose between payloads' results, no per-genre rationale, no chain-of-thought. A short summary AFTER the JSON block is fine.

## Core rules (mirrors the pipeline prompt)

1. **Tight and specific, no broad parents.** State what the release ACTUALLY IS — the 3–6 genres a knowledgeable listener would name. Broad parents (rock, pop, jazz, electronic, hip hop, folk, indie rock, alternative rock, experimental) are derived downstream from the genre graph; never include them. Shoegaze, not "rock"; ethio-jazz, not "world music"; trip-hop, not "downtempo".
2. **Genres ≠ facets.** Mood/texture/instrumentation/production/era/region/function/vocal/scene/format/rhythm descriptors (instrumental, lo-fi, acoustic, orchestral, 1970s, japanese, live, female vocals, drone) go in `facets`, never `genres`.
3. **User file tags are ground truth.** Every SPECIFIC `user_file_tags` genre MUST appear in `genres`, OR set `escalate: true` and name the omitted tag in `escalate_reason`. Silently dropping a specific user file tag is the single worst error. Broad-parent file tags (e.g. "rock") may be dropped without escalating.
4. **This release, not the artist.** Source tags are often artist-level and identical across albums — give THIS release its own identity. Never infer genre from artist name, nationality, language, album-title aesthetics, or demographic cues alone.
5. **Escalate over guess.** Set `escalate: true` when the release identity is ambiguous, evidence is thin and you'd be guessing, or a correct file tag would be dropped. Lower `confidence`/`overall_confidence` for sparse evidence. An escalation is a good outcome, not a failure.
6. **No web search** unless Dylan explicitly asks; never claim an external source says something it wasn't shown to say.
7. **Verify before rejecting a term.** Before declaring a proposed term bogus, check `data/layered_genre_taxonomy.yaml` (grep the term and its aliases) — legitimate niche subgenres (e.g. Kankyo Ongaku) have been wrongly challenged. Rare > common when expressing taste (design principle 12). Canonical spelling is normalized downstream — use canonical names where known.

## Related

- Pipeline path (batch, non-interactive): `adjudicate`/`apply` stages — `docs/genre_adjudication/ANALYZE_ADJUDICATE_STAGE.md`.
- Where accepted genres land and how to read them back: the **genre-data-authority** skill.
- Adding/editing taxonomy terms the adjudication surfaces: the **taxonomy-growth** skill.
````

- [ ] **Step 2: Verify frontmatter parses**

Run:

```bash
python -c "import yaml,io; t=io.open(r'C:\Users\Dylan\Desktop\PLAYLIST_GENERATOR_V3\.claude\skills\genre-adjudication\SKILL.md',encoding='utf-8').read(); fm=t.split('---')[1]; d=yaml.safe_load(fm); print('OK:', d['name'])"
```

Expected: `OK: genre-adjudication`

- [ ] **Step 3: Verify contract matches the source of truth**

Read `src/ai_genre_enrichment/album_adjudicator.py:23-29` and `:67-103`, then confirm all three of these in the skill file match exactly — fix the skill (never the .py) on any mismatch:

1. `layer` enum = `LAYERS` = `{core, secondary}`
2. `facet_type` enum = `FACET_TYPES` = `{mood, texture, instrumentation, production, era, region, function, vocal, scene, format, rhythm}`
3. Response object keys = `ADJUDICATOR_RESPONSE_SCHEMA["required"]` = `genres, facets, escalate, escalate_reason, overall_confidence, warnings` (the skill's extra `release_key` is documented as interactive-only, on top of these)

- [ ] **Step 4: Commit (explicit paths only)**

```bash
git -C C:\Users\Dylan\Desktop\PLAYLIST_GENERATOR_V3 status --short
git -C C:\Users\Dylan\Desktop\PLAYLIST_GENERATOR_V3 add .claude/skills/genre-adjudication/SKILL.md
git -C C:\Users\Dylan\Desktop\PLAYLIST_GENERATOR_V3 diff --cached --name-only
# expected: exactly this 1 file, then:
git -C C:\Users\Dylan\Desktop\PLAYLIST_GENERATOR_V3 commit --only -m "docs(skills): genre-adjudication skill — standardize interactive adjudication contract" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>" -- .claude/skills/genre-adjudication/SKILL.md
```

---

### Task 4: Sub-agent status-reporting rule in global CLAUDE.md

**Files:**
- Modify: `C:\Users\Dylan\.claude\CLAUDE.md` ("Sub-agent model policy" section, after the `CLAUDE_CODE_SUBAGENT_MODEL` bullet, ~line 28)
- **No commit** — this file is outside any repo.

**Interfaces:**
- Consumes: nothing.
- Produces: a global visibility rule for all future sessions.

**Context for the implementer:** The insights review flagged "opaque background execution" as a recurring friction — sub-agent builds ran silently and Dylan lost visibility mid-build. The fix is a standing contract: every delegated task reports back a one-line outcome in the main conversation.

- [ ] **Step 1: Apply the edit**

In `C:\Users\Dylan\.claude\CLAUDE.md`, find this exact bullet (end of "Sub-agent model policy"):

```markdown
- Do not set `CLAUDE_CODE_SUBAGENT_MODEL` (it would statically override this dynamic routing).
```

Replace with:

```markdown
- Do not set `CLAUDE_CODE_SUBAGENT_MODEL` (it would statically override this dynamic routing).
- **Sub-agent work is never silent.** When each sub-agent completes, relay a one-line outcome to me in the main conversation (what it did / changed / found — not its raw output). For multi-task builds, give a brief plan-of-record at each phase boundary. Opaque background execution is a known friction (insights 2026-07-06); visibility is part of the delegation contract.
```

- [ ] **Step 2: Verify**

Re-read the "Sub-agent model policy" section of `C:\Users\Dylan\.claude\CLAUDE.md` and confirm: the new bullet appears exactly once, the section has no duplicated lines, and no other section changed.

---

## Execution order

Tasks 1 → 2 → 3 → 4, matching the order agreed with Dylan. Tasks are independent (no shared files), but the order is a user instruction — keep it.

## Out of scope (explicitly deferred)

- The committed discrimination probe as a publish-gate in the fold script (flagged in `project_mert_collapse_incident` memory as "useful outcome") — separate feature, needs its own spec.
- The insights report's "default new features to False" suggestion — **rejected**: contradicts design principle 22 ("Activate fixes; never default to legacy").
- Re-running /insights on a non-overlapping window — operational note, not a code task.
