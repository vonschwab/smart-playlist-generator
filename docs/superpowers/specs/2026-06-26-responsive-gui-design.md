# Responsive GUI — Design

**Date:** 2026-06-26
**Branch:** `feat/responsive-gui`
**Status:** Approved (brainstorming complete) — pending implementation plan

## Problem

The web GUI (`web/`) is hard-coded for a wide desktop. Below a high resolution, elements
collapse and become unusable — most visibly the Artist text box in the toolbar, which is
squeezed toward zero width while the fixed-width selects beside it hold their size.

Audit findings:

- **No responsive system exists at all.** Zero `sm:`/`md:`/`lg:` breakpoints and zero
  media queries across the entire `web/src` tree.
- **`Shell.tsx`** is a fixed `h-screen`, 3-pane *drag-resizable* layout
  (Jobs │ Center │ Diagnostics) plus a bottom Logs pane, built on
  `react-resizable-panels`. Drag-to-resize does not degrade to touch/narrow screens.
- **`GenerateControls.tsx`** is three dense single-line `flex` rows of "cells" with **no
  wrapping**. As width shrinks, the `grow` cell (Artist input) is crushed while fixed
  selects hold width — the reported bug.
- Everything is sized in hard pixels (`text-[9px]`, `text-[11px]`), tuned for a wide
  monitor; unreadable and below touch-target size on a phone.

## Goal

Graceful down to phone. The desktop power-tool layout is preserved at full width, but the
UI must **never collapse** and must stay **readable and usable at any width down to a
phone**. On a phone, viewing/reviewing a generated playlist is the primary job; full
control-tweaking may be condensed behind disclosure.

Explicitly **not** a full mobile-first rebuild — phone is a supported viewport, not a
co-equal design target.

## Core strategy: container queries, not viewport breakpoints

The UI reflows based on **container width** using Tailwind v4 container queries
(`@container` / `@sm:` / `@md:` / `@max-md:`), **not** viewport width (`md:`/`lg:`).

Confirmed via context7: container queries are **built into Tailwind v4 core** — no plugin
to install. (`@tailwindcss/container-queries` was merged into core in v4.)

**Why this is correct for this architecture specifically:** the panels are
drag-resizable. A viewport-based fix would still let the Artist box collapse the moment a
user drags the center panel narrow on a wide monitor. Container queries make each region
respond to *its own* width — the durable fix.

Exactly **one** viewport breakpoint is used, and only for the top-level shell swap.

## Components

### 1. Shell (`Shell.tsx`) — one viewport breakpoint, two layouts

- **≥ 1024px (`lg`):** unchanged. Current 3-pane resizable layout + bottom Logs pane.
  Desktop power-tool fully preserved.
- **< 1024px:** `react-resizable-panels` is dropped entirely (drag-resize is meaningless
  on touch). Render a **single-column shell**: header → condensed controls → playlist
  (fills remaining height) → **bottom tab bar** with tabs **Playlist · Jobs · Diag ·
  Logs**. The active tab determines what fills the column; Playlist is the default tab.
- Implementation: a `useMediaQuery('(min-width: 1024px)')` hook selects `<DesktopShell>`
  vs `<MobileShell>`. **Both consume the same region nodes** (`jobs`, `center`, `right`,
  `logs`) that `App.tsx` already passes to `Shell` as props — the actual panel content is
  authored once, not duplicated.

**Boundaries:** `Shell` owns layout only. It does not know what the regions contain. The
media-query hook is a standalone unit in `web/src/lib/`.

### 2. Toolbar (`GenerateControls.tsx`) — container-query reflow

Wrap the toolbar in `@container`; three tiers driven by *its own* width:

- **Wide:** today's dense 3-row bar, unchanged.
- **Medium:** cells **wrap** onto more rows instead of squeezing. This is the immediate
  bug fix — the `grow` Artist cell stops being crushed because its neighbors wrap away.
- **Narrow/phone:** collapse to a **primary bar** (Mode · Artist/Genre input · ▸ Generate ·
  ↻ New Seeds) plus a **"More controls ▾"** disclosure holding rows 2–3 (cohesion, genre,
  sonic, pace, bangers; freshness, within, played, skip-recent-seeds, artist gap, artist
  diversity) and the artist-mode extras (artist presence, style spread, include
  collaborations, popular seeds). Defaults are sensible, so a phone user can generate
  without ever opening the disclosure.

**Decision (confirmed with user):** advanced controls go behind a disclosure on phone,
not always-visible-wrapped.

### 3. TrackTable (`TrackTable.tsx`) — column priority

Container-query–driven column visibility (not a separate mobile table):

- **Always:** `#` · Title/Artist · `T`.
- **Genre chips:** move under the title on narrow, or hide.
- **Last.fm rank:** first column to drop.

### 4. Type scale & touch targets

- Replace hard-coded `text-[9px]`/`text-[11px]` with small **semantic utilities** that
  bump up one notch on touch/narrow containers (readable on a phone, above tap-target
  size).
- Interactive rows/buttons reach ~40px touch targets on small screens.

### 5. Quality floor (`frontend-design` skill, build phase)

Visible keyboard focus; `prefers-reduced-motion` respected; no element collapses or
overflows at any supported width.

## Verification

Non-negotiable evidence step. **Playwright** (already in `web` devDependencies) drives the
running app at **375 / 768 / 1024 / 1440px**, screenshots each, and each is checked for
collapse/overflow. No success claim without screenshots.

## Non-goals / YAGNI

- No new runtime dependencies and no new Claude Code plugins. (Investigated: container
  queries + a media-query shell swap + Playwright cover it. `vaul` for drawers was
  considered and rejected — the tab-bar pattern needs no drawer library.)
- No component-library extraction / `DesignSync` push.
- No redesign of the desktop layout at `lg`+ — it is preserved as-is.
- Not a mobile-first rebuild; no native-app gestures beyond tab switching.

## Tooling notes (for the build phase)

- **context7** — current docs for Tailwind v4 container queries, Radix, react-resizable-panels.
- **frontend-design** skill — visual quality-floor pass.
- **Playwright** — multi-viewport screenshot verification.

## Risks / open considerations

- The `< 1024px` shell renders a different React tree than `≥ 1024px`. Region content must
  be authored to mount cleanly in both (no assumptions about parent being a resizable
  `Panel`). Verify the Diagnostics and Logs panels render standalone.
- `react-resizable-panels` persists sizes by `id`; ensure unmounting it below `lg` and
  remounting above `lg` restores saved sizes rather than resetting.
- Tab-bar state (which secondary region is shown on mobile) is ephemeral UI state — keep
  it local, do not persist to localStorage.
