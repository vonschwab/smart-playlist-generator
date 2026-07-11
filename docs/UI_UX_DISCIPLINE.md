# GUI design discipline

How the web GUI should look and behave, on every screen size. This is the UI counterpart to the
`web-gui` skill (which covers wiring/architecture traps). Grounded in Apple HIG, WCAG 2.2,
WebKit/MDN documentation, and NN/g research — distilled 2026-07-09; evidence and current
violations in `docs/UI_AUDIT_2026-07-09.md`.

**Scope trigger:** read this before designing or changing anything under `web/src/`.

## North star

1. **The GUI is an instrument panel, not a marketing site.** Dense, dark, legible, honest.
   Beautiful means disciplined — one type scale, one token palette, one accent doing its job —
   not gloss.
2. **The iPhone is a first-class surface, not a degraded desktop.** Dylan drives real
   generations from the phone over LAN. Every feature must be *usable* there, not merely
   reachable.
3. **The worst screen defines the experience** (same as playlists: floor quality, not average).
   One unreadable label or unreachable button on the phone breaks the tool.
4. **The phone lies about failure.** iOS Safari kills sockets, zooms viewports, and moves the
   viewport under you without any console to notice. Anything that can silently die on mobile
   needs an explicit recovery path.

## Hard rules

Each rule is verifiable. "Verify" tells you how; the sweep in *Verification harness* below
automates most of them.

### Touch & input

- **T1 — Primary controls ≥ 44×44 CSS px; nothing interactive below 24×24.** 44px for anything
  tapped often or blind (Generate, tab bars, player transport, dialog primary buttons); 24px is
  the WCAG 2.5.8 legal floor for dense secondary controls (table-row icons), and an undersized
  control's 24px circle may not overlap a neighbor's. *Verify:* `measurements.json` from the
  sweep — no `smallTargets` entry under 24, none under 44 for primary/nav/transport.
- **T2 — Every `<input>`/`<select>`/`<textarea>` computes to ≥ 16px font-size on touch**, or iOS
  zooms the page on focus. A `@media (pointer: coarse)` override is acceptable; the computed
  size is what counts. *Verify:* grep new inputs for sub-`text-base` sizes without a coarse
  override; focus each new field once on the phone.
- **T3 — No hover-only or right-click-only affordances.** Everything reachable via `:hover`
  reveal (`opacity-0 group-hover:*`) or `onContextMenu` must have an always-visible (or
  coarse-pointer-visible) tap path. Long-press is an enhancement, never the only path.
  *Verify:* `grep -rn "group-hover\|onContextMenu" web/src` — each hit needs a visible touch
  equivalent.
- **T4 — Custom controls suppress the gray tap flash only if they replace it**: pair
  `-webkit-tap-highlight-color: transparent` with an explicit `:active` style. Never zero
  feedback on tap.
- **T5 — Numeric fields declare `inputMode`** (`numeric`/`decimal`) so the phone shows the right
  keyboard.

### Viewport & layout (iOS Safari)

- **V1 — Full-height containers use `dvh`, never bare `vh`/`h-screen`.** Safari's `100vh` is the
  *largest* viewport; the bottom of an `h-screen` app sits under the toolbar. *Verify:*
  `grep -rn "h-screen\|100vh" web/src` → only `dvh` (with `vh` fallback) allowed.
- **V2 — Edge-pinned bars respect safe areas.** With `viewport-fit=cover` in the viewport meta,
  any bar touching a screen edge (mobile tab bar, mini player) pads with
  `env(safe-area-inset-*)`. *Verify:* grep `safe-area-inset` next to every `fixed`/bottom bar.
- **V3 — Floating/portaled elements clamp to the viewport.** Popovers and dropdowns computed
  from `getBoundingClientRect` must clamp against `window.innerWidth/innerHeight` (and portal
  out of `overflow-hidden` ancestors). *Verify:* open each popover from its extreme trigger
  positions at 390px.
- **V4 — Dialogs fit the phone + keyboard:** `max-h-[85dvh] overflow-y-auto` on content; assume
  the keyboard eats ~40% of height when an input is focused. Modals are reserved for blocking or
  destructive confirmation; mid-task option surfaces prefer non-modal bottom sheets (NN/g).
- **V5 — No document-level horizontal overflow at 390px, ever.** `scrollWidth −
  clientWidth ≤ 1`. The existing `responsive.spec.ts` + sweep assert this — keep them green.
- **V6 — Independent scroll panes set `overscroll-behavior: contain`** so inner scroll ends
  don't rubber-band the shell or trigger pull-to-refresh.

### Lifecycle (the phone lies)

- **L1 — Treat a backgrounded tab's WebSocket as dead.** Every job-driven surface needs (a)
  reconnect-with-backoff on `close`/`error`, and (b) a poll reconcile on
  `visibilitychange`/`pageshow` (`useJobReconcile` is the house pattern — the main Generate flow
  must use it too). *Verify:* background Safari 30s mid-generation; the UI must recover without
  reload.
- **L2 — `audio.play()` stays in the synchronous call stack of a user gesture**, and its promise
  rejection is handled. Queue auto-advance is best-effort on iOS, never assumed.
- **L3 — Job state renders from reconcilable server state, not accumulated WS events**, so a
  missed event window can always be healed by one poll.

### Color & type

- **C1 — Theme tokens are the only colors.** No raw hex/rgb in components (JS color maps use
  `var(--color-*)`); severity uses `warn`/`danger`/`info` tokens (add the token if it's missing
  — don't invent a hex). A color used twice is a token. *Verify:* `web/src/designTokens.test.ts`
  (runs in the unit suite) fails the build on any raw hex or stock palette class.
- **C2 — Contrast floor: 4.5:1 body text, 3:1 large text and UI glyphs**, computed against the
  actual token hex. Every text token (`text`/`muted`/`faint`) holds 4.5:1 on `panel` and `bg`;
  status colors hold 3:1. *Verify:* `designTokens.test.ts` computes the ratios from the live
  `@theme` values — a token change that breaks contrast fails the suite.
- **C3 — No pure `#000` surfaces / pure `#fff` long-form text** (halation on OLED during long
  log-reading sessions). Current tokens already comply — keep it that way.
- **C4 — The type scale is the named scale.** Steps: `text-2xs` (10px — the only sanctioned
  sub-xs step, for dense read-only data; the coarse-pointer media query raises its theme
  variable to 12px so phones never render smaller) then the standard `xs`/`sm`/…. No
  `text-[Npx]` arbitrary sizes — `designTokens.test.ts` fails on them. Micro-type never sits on
  interactive controls.
- **C5 — Focus stays visible.** `:focus-visible` ring ≥ 3:1 against surroundings (global rule in
  `index.css` — don't `outline: none` past it). `prefers-reduced-motion` stays honored globally.

### Components & states

- **S1 — One recipe per control kind.** Primary button, ghost button, chip, input: single shared
  className/component, not per-file reinventions. Divergence is a bug even when the rendered
  pixels happen to match.
- **S2 — Every panel designs its three states: empty, loading, error.** Empty states name the
  action that fills them ("No jobs yet — generate a playlist"); a black void is a defect. Errors
  go through one `friendlyError()` path — no raw `Error: …` strings, no bare HTTP codes.
- **S3 — User vocabulary only.** Pipeline/system nouns (`muq`, `genre-sim`, `artifacts`,
  `job_id`, worker/bridge) never appear unglossed. Buttons name their action ("Save changes",
  not "Submit"); the same action keeps the same name everywhere; pluralization is computed,
  never "Track(s)".
- **S4 — Navigation stays visible: bottom tab bar ≤ 5 destinations**; overflow goes in an
  explicit "More" tab, not a hamburger. Progressive disclosure caps at 2 levels — anything used
  every session surfaces at level 1.
- **S5 — On the phone, the payoff owns the viewport.** After Generate, the playlist — not the
  controls — gets the screen; controls collapse behind disclosure. Primary actions live in the
  bottom thumb zone; destructive actions live *out* of it.

### App identity

- **I1 — The served HTML owns its identity:** real `<title>`, real favicon, `theme-color` +
  `apple-mobile-web-app-status-bar-style` matching the dark theme, 180×180 `apple-touch-icon`,
  and a manifest with `display: standalone` (the phone entry point is Add-to-Home-Screen, not a
  bookmark).
- **I2 — Local-first includes fonts.** No render-blocking third-party CDN fetches; self-host
  Spline Sans / JetBrains Mono with `font-display: swap`.

## Verification harness

- **The sweep** (screenshots + target/type/overflow measurements at 390×844, chromium + WebKit):

  ```
  npm --prefix web exec -- playwright test mobile-audit --config web/playwright.config.ts
  $env:PW_BROWSER = 'webkit'; npm --prefix web exec -- playwright test mobile-audit --config web/playwright.config.ts
  ```

  Output: `web/test-results/mobile-audit[-webkit]/` (`*.png` + `measurements.json`). Run it —
  and **look at the screenshots** — before and after any GUI change that touches layout.
- **Existing gates:** `web/tests/responsive.spec.ts` (overflow at 4 widths), unit tests
  (`npm --prefix web run test`), and the `web-gui` skill's build/restart trap catalog.
- **What automation can't see** (dynamic toolbar, focus zoom, safe areas, backgrounding, audio
  gestures): once per significant change, exercise the real flow on the actual iPhone over LAN.

## Tooling

Installed support: `frontend-design` plugin (aesthetic direction), this doc (rules), the sweep
(evidence). Recommended additions (surveyed 2026-07-09, verified to exist):

```
claude mcp add playwright -- cmd /c npx @playwright/mcp@latest
claude mcp add playwright-iphone -- cmd /c npx @playwright/mcp@latest --browser webkit --device "iPhone 15"
claude plugin install chrome-devtools-mcp@claude-plugins-official
npx skills add vercel-labs/agent-skills   # web-design-guidelines static checker
```

## Maintenance

Like the `web-gui` skill, this is a living index: when a rule proves wrong or a new phone trap
costs a debugging cycle, amend the rule here in the same PR that fixes it.
