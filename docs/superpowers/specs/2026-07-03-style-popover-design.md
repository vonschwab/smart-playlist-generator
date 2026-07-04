# Unified "Style" popover (genre-tag picker) — design spec

**Date:** 2026-07-03
**Status:** approved design, pre-plan
**Feature group:** genre-tag steering ([[project_tag_steering]]) — GUI polish

## Problem

The genre-tag chips currently render as an inline `<Cell>` inside Row 1 of
`web/src/components/GenerateControls.tsx` (added ~`:360-389`). Being a flex item in the
wrapping row, the chips **elastically displace** the rest of Row 1 — the Generate/New Seeds
buttons and neighbouring controls shift as chips appear and as the button label widens to
"Generating…". Dylan: "sloppy… genres pop out and displace other stuff." (Evidence: the
Torrey screenshot, 2026-07-03.)

## Decision (locked with Dylan, 2026-07-03)

A **unified "Style" popover**: replace the Style Spread dropdown with a fixed-width "Style ▾"
trigger; clicking opens a **floating card** (absolute-positioned overlay — does not reflow
Row 1) that holds BOTH the Style Spread selector and the genre chips. Rationale: Style Spread
and genre-lean are both "which slice of this artist's style," so they belong in one place; and
an overlay eliminates displacement entirely. Chosen over a separate "Genres" popover button
and over a dedicated fixed row.

## 1. Component

New presentational component `web/src/components/StylePopover.tsx` (GenerateControls is
already large; one responsibility per file).

**Props (all owned by GenerateControls — state stays where it flows into `submit()`):**
```ts
{
  artistVariety: string;
  onVarietyChange: (v: string) => void;
  artistTags: { name: string; release_count: number; confidence: number }[];
  steeringTags: string[];
  onToggleTag: (name: string) => void;
  tagsFetched: boolean;
}
```
`StylePopover` owns ONLY its open/closed boolean (+ its own outside-click / Escape handling).

## 2. Trigger

Replaces the Style Spread `<Cell>` (`GenerateControls.tsx:312-326`). A fixed-width button
rendered via the existing `<Cell>` helper:
- Label `Style` + a chevron (`▾`).
- A small accent count-badge shown only when `steeringTags.length > 0` (e.g. a `#5eead4`
  pill with the count). The badge is a fixed, self-contained element — it does not reflow the
  row (the whole picker lives in one Cell).
- `aria-expanded={open}`, `aria-controls` pointing at the card.
- Visible only in artist mode (Style Spread already is — it's inside the `mode === "artist"`
  block).

## 3. The floating card

- `absolute`, anchored under the trigger, left-aligned, `z-20` (must sit ABOVE Row 2
  cohesion/genre/sonic, which the autocomplete dropdown's `z-10` would collide with — use a
  higher z). Width ~280px. Reuses the popover idiom already in this file (the artist
  autocomplete `<ul>` at `:241-265`: `absolute … bg-[#16181d] border border-[#23262d] rounded
  shadow-xl`).
- **Spread section:** the existing `artistVariety` `<select>` (focused / balanced /
  sprawling), relabeled "Spread" inside the card; `onChange` → `onVarietyChange`.
- **"Lean toward genres" section:** the genre chips — same toggle behaviour, ≤3 cap, accent
  styling as today (`:363-386`). Preserve `data-testid="steering-chips"` on the chip
  container. States:
  - `artistTags.length > 0` → the chips.
  - `tagsFetched && artistTags.length === 0` → hint "No published genres for this artist —
    run enrichment publish to enable tag steering." (moved from `GenerateControls.tsx:395-...`).
  - not yet fetched → a subtle "Loading…" line (optional; acceptable to render nothing).

## 4. Close behaviour

- Outside-click closes (extend the existing outside-click effect at
  `GenerateControls.tsx:129-139`, or a self-contained `useEffect` + `ref` inside StylePopover).
- `Escape` closes.
- Selecting a chip or changing spread does NOT auto-close (multi-select is expected).

## 5. State & data flow — UNCHANGED

`steeringTags`, `artistTags`, `tagsFetched`, the fetch-on-confirmed-artist effect, and the
reset-on-artist-change all stay in GenerateControls. `submit()` still sends `steering_tags`
(when artist + ≥1 selected) and `artist_variety` exactly as today. **No change to
`api.ts`, `types.ts`, schemas, policy, worker, or config.** Pure front-end DOM relocation.
Remove the old inline chips `<Cell>` (`:360-389`) and the old empty-state hint block.

## 6. Testing

- `npm --prefix web run build` — `tsc -b` clean, no type errors (this is the gate; there is
  no unit test for this component).
- Preserve `data-testid="steering-chips"` so any existing Playwright selector still resolves.
- Live acceptance (Dylan): in artist mode, pick an artist → "Style ▾" shows; open it → spread
  + chips; select 2 → badge shows ②; **Row 1 and the Generate button do not move** as chips
  appear or while generating. Front-end only — no worker restart, just a browser refresh
  after the build.

## Out of scope

- Any backend/policy/schema/worker change.
- Changing what the chips or spread DO (steering behaviour is unchanged — this is layout only).
- Non-artist modes (the trigger is artist-only).

## Key references

- Inline chips to remove: `GenerateControls.tsx:360-389`; empty-state hint to move: the block
  immediately after it.
- Style Spread cell to replace with the trigger: `:312-326`.
- Popover idiom to reuse: autocomplete `<ul>` `:241-265`; outside-click effect `:129-139`.
- Chip state/fetch (stays in GenerateControls): `useState` block ~`:106-110`, fetch effect,
  `toggleSteeringTag`, and the `submit()` body's `steering_tags` line ~`:203`.
