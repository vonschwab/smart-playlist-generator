# Responsive GUI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the web GUI graceful down to phone — never collapses, stays readable and usable at any width — while preserving the desktop power-tool layout unchanged at ≥1024px.

**Architecture:** One viewport media query (`min-width:1024px`) swaps a desktop 3-pane resizable shell for a single-column mobile shell with a bottom tab bar. Everything *inside* a region reflows with Tailwind v4 **container queries** (`@container`/`@max-md:`), so components respond to their own (possibly drag-resized) width rather than the viewport. No new dependencies.

**Tech Stack:** React 19, TypeScript, Vite 6, Tailwind CSS v4 (config-less, via `@tailwindcss/vite`, container queries in core), `react-resizable-panels` v4, `@tanstack/react-table` v8, Vitest + `@testing-library/react` (jsdom) for unit tests, Playwright for multi-viewport e2e.

## Global Constraints

- No new runtime dependencies and no new Claude Code plugins. Container queries are Tailwind v4 core (`@container`, `@sm:`…`@7xl:`, `@max-md:` etc.) — do **not** install `@tailwindcss/container-queries` or `vaul`.
- Desktop layout at `≥1024px` is preserved as-is. Only the `<1024px` path is new.
- The desktop↔mobile swap uses exactly **one** viewport media query: `(min-width: 1024px)`. All other responsiveness uses container queries.
- Tailwind container-query breakpoints (width of the nearest `@container` ancestor): `@sm`=384px, `@md`=448px, `@lg`=512px, `@3xl`=768px. This plan uses the `@md` (448px) threshold for the toolbar disclosure and the Last.fm column.
- All files live under `web/`. Run npm from repo root with `--prefix web` (per project convention).
- Unit tests run in jsdom, which does **not** evaluate container queries or layout. Test *behavior and class wiring* in unit tests; verify *visual reflow* only in Playwright.
- Every commit message ends with the trailer:
  `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`
- Commands: test `npm --prefix web run test`; single file `npm --prefix web run test -- <substr>`; lint `npm --prefix web run lint`; build `npm --prefix web run build`; e2e `npm --prefix web run test:e2e`.

---

## File Structure

- **Create** `web/src/lib/useMediaQuery.ts` — `useMediaQuery(query: string): boolean` hook (Task 1).
- **Create** `web/src/lib/useMediaQuery.test.ts` — unit tests for the hook (Task 1).
- **Modify** `web/src/components/Shell.tsx` — split into desktop body + mobile body w/ bottom tab bar (Task 2).
- **Create** `web/src/components/Shell.test.tsx` — mobile-mode tab-switching tests (Task 2).
- **Modify** `web/src/components/GenerateControls.tsx` — `@container`, wrapping rows, "More controls" disclosure (Task 3).
- **Create** `web/src/components/GenerateControls.test.tsx` — disclosure toggle test (Task 3).
- **Modify** `web/src/components/TrackTable.tsx` — `@container` wrapper + `meta.cellClass` column hiding (Task 4).
- **Modify** `web/src/components/QualityStats.tsx` — wrap the stats row (Task 4).
- **Create** `web/src/components/TrackTable.test.tsx` — Last.fm column carries hide-class (Task 4).
- **Modify** `web/src/index.css` — focus-visible ring + reduced-motion + min readable type on touch (Task 5).
- **Create** `web/tests/responsive.spec.ts` — Playwright 375/768/1024/1440 verification + screenshots (Task 6).

Dependency order: Task 1 → Task 2 (uses the hook). Tasks 3, 4, 5 are independent of each other and of Task 2. Task 6 verifies everything and runs last.

---

### Task 1: `useMediaQuery` hook

**Files:**
- Create: `web/src/lib/useMediaQuery.ts`
- Test: `web/src/lib/useMediaQuery.test.ts`

**Interfaces:**
- Consumes: nothing.
- Produces: `export function useMediaQuery(query: string): boolean` — reactive, SSR/jsdom-safe (returns `false` when `window.matchMedia` is unavailable), updates on the matchMedia `change` event.

- [ ] **Step 1: Write the failing test**

Create `web/src/lib/useMediaQuery.test.ts`:

```ts
import { describe, it, expect, vi, afterEach } from "vitest";
import { renderHook, act } from "@testing-library/react";
import { useMediaQuery } from "./useMediaQuery";

// Controllable matchMedia mock: returns a media query list whose `matches`
// reflects a closure variable, and captures the registered change handler.
function installMatchMedia(initial: boolean) {
  let current = initial;
  let handler: ((e: unknown) => void) | null = null;
  window.matchMedia = vi.fn().mockImplementation((query: string) => ({
    get matches() { return current; },
    media: query,
    addEventListener: (_type: string, cb: (e: unknown) => void) => { handler = cb; },
    removeEventListener: vi.fn(),
  })) as unknown as typeof window.matchMedia;
  return {
    flip(v: boolean) { current = v; handler?.({}); },
  };
}

afterEach(() => { vi.restoreAllMocks(); });

describe("useMediaQuery", () => {
  it("returns the initial match state", () => {
    installMatchMedia(true);
    const { result } = renderHook(() => useMediaQuery("(min-width: 1024px)"));
    expect(result.current).toBe(true);
  });

  it("updates when the media query changes", () => {
    const ctl = installMatchMedia(false);
    const { result } = renderHook(() => useMediaQuery("(min-width: 1024px)"));
    expect(result.current).toBe(false);
    act(() => ctl.flip(true));
    expect(result.current).toBe(true);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npm --prefix web run test -- useMediaQuery`
Expected: FAIL — `useMediaQuery` cannot be imported / is not a function.

- [ ] **Step 3: Write minimal implementation**

Create `web/src/lib/useMediaQuery.ts`:

```ts
import { useEffect, useState } from "react";

// Reactive viewport media query. Safe under jsdom/SSR (no matchMedia) where it
// reports false. Used only for the desktop↔mobile shell swap; everything else
// responds to container width, not the viewport.
export function useMediaQuery(query: string): boolean {
  const [matches, setMatches] = useState<boolean>(() => {
    if (typeof window === "undefined" || !window.matchMedia) return false;
    return window.matchMedia(query).matches;
  });

  useEffect(() => {
    if (typeof window === "undefined" || !window.matchMedia) return;
    const mql = window.matchMedia(query);
    const onChange = () => setMatches(mql.matches);
    onChange();
    mql.addEventListener("change", onChange);
    return () => mql.removeEventListener("change", onChange);
  }, [query]);

  return matches;
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npm --prefix web run test -- useMediaQuery`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add web/src/lib/useMediaQuery.ts web/src/lib/useMediaQuery.test.ts
git commit -m "feat(web): add useMediaQuery hook for responsive shell swap

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Shell — desktop/mobile swap + bottom tab bar

**Files:**
- Modify: `web/src/components/Shell.tsx` (entire file rewritten below)
- Test: `web/src/components/Shell.test.tsx`

**Interfaces:**
- Consumes: `useMediaQuery` from `../lib/useMediaQuery` (Task 1).
- Produces: unchanged public API — `Shell({ topBar, jobs, center, right, logs }: { topBar: ReactNode; jobs: ReactNode; center: ReactNode; right: ReactNode; logs: ReactNode })`. `App.tsx` needs no change.
- Mobile DOM contract (relied on by Task 6): a `<nav data-testid="mobile-tabbar">` with buttons `data-testid` `tab-mobile-playlist|jobs|diag|logs`; default active region is `center`.

- [ ] **Step 1: Write the failing test**

Create `web/src/components/Shell.test.tsx`:

```tsx
import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen, fireEvent, cleanup } from "@testing-library/react";
import { Shell } from "./Shell";

// Force the mobile shell: matchMedia('(min-width:1024px)') -> false.
function forceMobile() {
  window.matchMedia = vi.fn().mockImplementation((query: string) => ({
    matches: false,
    media: query,
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
  })) as unknown as typeof window.matchMedia;
}

afterEach(() => { cleanup(); vi.restoreAllMocks(); });

function renderShell() {
  return render(
    <Shell
      topBar={<div>TOPBAR</div>}
      jobs={<div data-testid="stub-jobs">JOBS</div>}
      center={<div data-testid="stub-center">CENTER</div>}
      right={<div data-testid="stub-right">RIGHT</div>}
      logs={<div data-testid="stub-logs">LOGS</div>}
    />,
  );
}

describe("Shell (mobile)", () => {
  it("renders the bottom tab bar and defaults to the center region", () => {
    forceMobile();
    renderShell();
    expect(screen.getByTestId("mobile-tabbar")).toBeTruthy();
    expect(screen.getByTestId("stub-center")).toBeTruthy();
    // Secondary regions are not mounted until their tab is selected.
    expect(screen.queryByTestId("stub-jobs")).toBeNull();
  });

  it("switches the active region when a tab is tapped", () => {
    forceMobile();
    renderShell();
    fireEvent.click(screen.getByTestId("tab-mobile-jobs"));
    expect(screen.getByTestId("stub-jobs")).toBeTruthy();
    expect(screen.queryByTestId("stub-center")).toBeNull();

    fireEvent.click(screen.getByTestId("tab-mobile-diag"));
    expect(screen.getByTestId("stub-right")).toBeTruthy();

    fireEvent.click(screen.getByTestId("tab-mobile-logs"));
    expect(screen.getByTestId("stub-logs")).toBeTruthy();
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npm --prefix web run test -- Shell`
Expected: FAIL — no `mobile-tabbar` (current Shell always renders the resizable desktop layout).

- [ ] **Step 3: Write minimal implementation**

Replace the entire contents of `web/src/components/Shell.tsx` with:

```tsx
import { Group, Panel, Separator } from "react-resizable-panels";
import { useState, type ReactNode } from "react";
import { useMediaQuery } from "../lib/useMediaQuery";

// Separator className: v4 uses data-separator for hover/active; no resize-handle-state attr
const handleV =
  "w-1.5 bg-bg hover:bg-accent transition-colors cursor-col-resize data-[disabled]:cursor-default";
const handleH =
  "h-1.5 bg-bg hover:bg-accent transition-colors cursor-row-resize data-[disabled]:cursor-default";

interface ShellProps {
  topBar: ReactNode;
  jobs: ReactNode;
  center: ReactNode;
  right: ReactNode;
  logs: ReactNode;
}

export function Shell(props: ShellProps) {
  // Single viewport breakpoint: drag-resizable panes only make sense with a pointer
  // and real width. Below it we render a touch-friendly single-column shell.
  const isDesktop = useMediaQuery("(min-width: 1024px)");
  return (
    <div className="h-screen flex flex-col bg-bg text-text">
      <header className="flex flex-wrap items-center justify-between gap-2 px-4 py-2.5 bg-panel border-b border-border">
        {props.topBar}
      </header>
      {isDesktop ? <DesktopBody {...props} /> : <MobileBody {...props} />}
    </div>
  );
}

function DesktopBody(props: ShellProps) {
  return (
    <Group orientation="vertical" className="flex-1" id="pg-vert">
      <Panel defaultSize={78} minSize={40}>
        <Group orientation="horizontal" id="pg-horiz">
          <Panel
            defaultSize={16}
            minSize={10}
            collapsible
            className="bg-panel border-r border-border"
          >
            {props.jobs}
          </Panel>
          <Separator className={handleV} />
          <Panel defaultSize={62} minSize={30} className="bg-bg overflow-hidden">
            {props.center}
          </Panel>
          <Separator className={handleV} />
          <Panel
            defaultSize={22}
            minSize={14}
            collapsible
            className="bg-panel border-l border-border"
          >
            {props.right}
          </Panel>
        </Group>
      </Panel>
      <Separator className={handleH} />
      <Panel
        defaultSize={22}
        minSize={8}
        collapsible
        className="bg-[#0c0e12] border-t border-border"
      >
        {props.logs}
      </Panel>
    </Group>
  );
}

type Region = "playlist" | "jobs" | "diag" | "logs";

function MobileBody(props: ShellProps) {
  const [region, setRegion] = useState<Region>("playlist");
  const active =
    region === "playlist" ? props.center
    : region === "jobs" ? props.jobs
    : region === "diag" ? props.right
    : props.logs;

  const tabs: Array<{ id: Region; label: string }> = [
    { id: "playlist", label: "Playlist" },
    { id: "jobs", label: "Jobs" },
    { id: "diag", label: "Diag" },
    { id: "logs", label: "Logs" },
  ];

  return (
    <>
      <div className="flex-1 min-h-0 overflow-hidden bg-bg">{active}</div>
      <nav
        data-testid="mobile-tabbar"
        className="flex shrink-0 border-t border-border bg-panel"
      >
        {tabs.map((t) => (
          <button
            key={t.id}
            data-testid={`tab-mobile-${t.id}`}
            onClick={() => setRegion(t.id)}
            aria-current={region === t.id}
            className={[
              "flex-1 py-3 text-xs font-medium transition-colors",
              region === t.id ? "text-accent" : "text-faint hover:text-muted",
            ].join(" ")}
          >
            {t.label}
          </button>
        ))}
      </nav>
    </>
  );
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npm --prefix web run test -- Shell`
Expected: PASS (2 tests).

- [ ] **Step 5: Verify no regression in the rest of the unit suite**

Run: `npm --prefix web run test`
Expected: PASS (all existing tests + the new ones).

- [ ] **Step 6: Commit**

```bash
git add web/src/components/Shell.tsx web/src/components/Shell.test.tsx
git commit -m "feat(web): mobile single-column shell with bottom tab bar

Below 1024px, drop react-resizable-panels for a one-column layout whose
bottom tab bar swaps Playlist/Jobs/Diag/Logs. Desktop layout unchanged.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: GenerateControls — wrapping rows + "More controls" disclosure

**Files:**
- Modify: `web/src/components/GenerateControls.tsx`
- Test: `web/src/components/GenerateControls.test.tsx`

**Interfaces:**
- Consumes: nothing new.
- Produces: unchanged props/behavior. New DOM contract: outer wrapper has class `@container`; a disclosure button `data-testid="more-controls"` with `aria-expanded`; the advanced region is `data-testid="advanced-controls"` and carries class `@max-md:hidden` only while collapsed.

Background: the bug is that Row 1 is `flex items-center` with **no wrap**, so the `grow` Artist cell is crushed. Fix = let rows wrap, and on a narrow *container* (<448px) tuck Rows 2–3 + artist extras behind a disclosure.

- [ ] **Step 1: Write the failing test**

Create `web/src/components/GenerateControls.test.tsx`:

```tsx
import { describe, it, expect, afterEach } from "vitest";
import { render, screen, fireEvent, cleanup } from "@testing-library/react";
import { GenerateControls } from "./GenerateControls";

afterEach(() => { cleanup(); localStorage.clear(); });

function renderControls() {
  return render(
    <GenerateControls
      mode="artist"
      onModeChange={() => {}}
      seedTrackIds={[]}
      seedDisplays={[]}
      onSubmit={() => {}}
      busy={false}
    />,
  );
}

describe("GenerateControls disclosure", () => {
  it("renders a collapsed advanced region with a More controls toggle", () => {
    renderControls();
    const toggle = screen.getByTestId("more-controls");
    expect(toggle.getAttribute("aria-expanded")).toBe("false");
    // Collapsed: advanced region carries the container-query hide class.
    expect(screen.getByTestId("advanced-controls").className).toContain("@max-md:hidden");
  });

  it("expands and collapses when the toggle is clicked", () => {
    renderControls();
    const toggle = screen.getByTestId("more-controls");
    fireEvent.click(toggle);
    expect(toggle.getAttribute("aria-expanded")).toBe("true");
    expect(screen.getByTestId("advanced-controls").className).not.toContain("@max-md:hidden");
    fireEvent.click(toggle);
    expect(toggle.getAttribute("aria-expanded")).toBe("false");
    expect(screen.getByTestId("advanced-controls").className).toContain("@max-md:hidden");
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npm --prefix web run test -- GenerateControls`
Expected: FAIL — no `more-controls` / `advanced-controls` elements.

- [ ] **Step 3a: Add the disclosure state**

In `web/src/components/GenerateControls.tsx`, add a `showMore` state alongside the other `useState`/`useLocalStorage` hooks (e.g. immediately after the `const [seedEpoch, setSeedEpoch] = useState(0);` line near the top of the component body):

```tsx
  // On a narrow *container* (<@md = 448px) Rows 2–3 + artist extras collapse
  // behind this toggle. At >=448px they are always shown regardless of state.
  const [showMore, setShowMore] = useState(false);
```

- [ ] **Step 3b: Mark the toolbar as a container and let Row 1 wrap**

Change the outer wrapper (currently
`<div className="border border-[#23262d] rounded-none border-x-0 border-t-0 overflow-hidden">`)
to add `@container`:

```tsx
    <div className="@container border border-[#23262d] rounded-none border-x-0 border-t-0 overflow-hidden">
```

Change the Row 1 container (currently
`<div className="flex items-center bg-[#16181d] border-b border-[#1e2128]">`)
to wrap:

```tsx
      <div className="flex flex-wrap items-center bg-[#16181d] border-b border-[#1e2128]">
```

- [ ] **Step 3c: Insert the disclosure toggle at the end of Row 1**

Inside Row 1, immediately **after** the `mode === "artist"` "↻ New Seeds" `Cell` block and **before** the closing `</div>` of Row 1, add:

```tsx
        <Cell>
          <button
            type="button"
            data-testid="more-controls"
            aria-expanded={showMore}
            onClick={() => setShowMore((v) => !v)}
            className="@md:hidden border border-[#23262d] text-[#8b939d] text-[11px] px-3 py-[4px] rounded whitespace-nowrap"
          >
            {showMore ? "Less controls ▴" : "More controls ▾"}
          </button>
        </Cell>
```

(The `@md:hidden` makes this button appear only when the toolbar container is narrower than 448px.)

- [ ] **Step 3d: Wrap Rows 2 & 3 in the collapsible advanced region**

Wrap **both** Row 2 (`flex items-center bg-[#13151a] …`) and Row 3 (`flex items-center bg-[#111317]`) in a single `<div>` that toggles the hide class. Place the opening tag immediately before Row 2 and the closing tag immediately after Row 3's closing `</div>`:

```tsx
      <div
        data-testid="advanced-controls"
        className={showMore ? "" : "@max-md:hidden"}
      >
        {/* ── ROW 2: cohesion + matching ─────────────────────────────── */}
        <div className="flex flex-wrap items-center bg-[#13151a] border-b border-[#1e2128]">
          {/* …existing Row 2 cells unchanged… */}
        </div>

        {/* ── ROW 3: freshness + spacing ─────────────────────────────── */}
        <div className="flex flex-wrap items-center bg-[#111317]">
          {/* …existing Row 3 cells unchanged… */}
        </div>
      </div>
```

Notes:
- Also change Row 2 and Row 3's own `flex items-center` to `flex flex-wrap items-center` (shown above) so their cells wrap on medium containers instead of squeezing.
- Leave every existing `Cell`/`Lbl`/`select` inside Rows 2 and 3 exactly as-is; only the wrapper `<div>` and the `flex-wrap` change.
- The artist-mode extras (artist presence, style spread, include collaborations, popular seeds) stay in Row 1; with `flex-wrap` they wrap to a second line on medium widths. They are intentionally *not* moved into the disclosure to keep Row 1 self-contained — confirm at Playwright time they wrap acceptably; if Row 1 is too tall on phones, a follow-up can move them, but that is out of scope here.

- [ ] **Step 4: Run test to verify it passes**

Run: `npm --prefix web run test -- GenerateControls`
Expected: PASS (2 tests).

- [ ] **Step 5: Typecheck/lint the changed file**

Run: `npm --prefix web run lint`
Expected: no new errors.

- [ ] **Step 6: Commit**

```bash
git add web/src/components/GenerateControls.tsx web/src/components/GenerateControls.test.tsx
git commit -m "feat(web): toolbar wraps + collapses advanced controls on narrow containers

@container + flex-wrap stops the Artist input from being crushed; below the
@md (448px) container width, Rows 2-3 tuck behind a More controls disclosure.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: TrackTable column priority + QualityStats wrap

**Files:**
- Modify: `web/src/components/TrackTable.tsx`
- Modify: `web/src/components/QualityStats.tsx`
- Test: `web/src/components/TrackTable.test.tsx`

**Interfaces:**
- Consumes: nothing new.
- Produces: unchanged props. New DOM contract: the table is wrapped in a `@container` div; the `Last.fm` column header + cells carry class `@max-md:hidden` (drop first on narrow). QualityStats row wraps instead of overflowing.

- [ ] **Step 1: Write the failing test**

Create `web/src/components/TrackTable.test.tsx`:

```tsx
import { describe, it, expect, afterEach } from "vitest";
import { render, screen, cleanup } from "@testing-library/react";
import { TrackTable } from "./TrackTable";
import { PlayerProvider } from "../contexts/PlayerContext";
import type { TrackOut } from "../lib/types";

afterEach(() => cleanup());

function track(overrides: Partial<TrackOut> = {}): TrackOut {
  return {
    position: 0,
    artist: "Phoebe Bridgers",
    title: "Scott Street",
    album: "Stranger in the Alps",
    duration_ms: 1000,
    file_path: "/x.mp3",
    genres: ["indie folk"],
    popularity_rank: 2,
    ...overrides,
  };
}

describe("TrackTable responsive columns", () => {
  it("hides the Last.fm column on narrow containers", () => {
    render(
      <PlayerProvider>
        <TrackTable tracks={[track()]} />
      </PlayerProvider>,
    );
    const header = screen.getByText("Last.fm");
    expect(header.className).toContain("@max-md:hidden");
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npm --prefix web run test -- TrackTable`
Expected: FAIL — `Last.fm` header has no `@max-md:hidden` class.

- [ ] **Step 3a: Carry a per-column hide class via TanStack `meta`**

In `web/src/components/TrackTable.tsx`, add `meta` to the popularity column. Change the popularity column definition (currently `col.accessor("popularity_rank", { header: "Last.fm", cell: … })`) to include:

```tsx
          col.accessor("popularity_rank", {
            header: "Last.fm",
            meta: { cellClass: "@max-md:hidden" },
            cell: (c) => {
              const r = c.getValue();
              return (
                <span
                  className="font-mono text-[11px] text-faint"
                  title="Last.fm popularity rank within the artist's top tracks (lower = more popular)"
                >
                  {r == null ? "—" : `#${r}`}
                </span>
              );
            },
          }),
```

- [ ] **Step 3b: Apply `meta.cellClass` to header and body cells, and wrap the table in a container**

Replace the component's returned `<table>…</table>` block so the `<th>` and `<td>` append the meta class, and the whole table is wrapped in an `@container` div. Replace from `return (` (the final return) through the closing `);`:

```tsx
  return (
    <div className="@container">
      <table className="w-full text-left" data-testid="track-table">
        <thead>
          {table.getHeaderGroups().map((hg) => (
            <tr key={hg.id} className="border-b border-border">
              {hg.headers.map((h) => {
                const extra = (h.column.columnDef.meta as { cellClass?: string } | undefined)?.cellClass ?? "";
                return (
                  <th
                    key={h.id}
                    onClick={h.column.getToggleSortingHandler()}
                    className={`px-3 py-2 text-[9px] uppercase tracking-wide text-faint cursor-pointer select-none ${extra}`}
                  >
                    {flexRender(h.column.columnDef.header, h.getContext())}
                  </th>
                );
              })}
            </tr>
          ))}
        </thead>
        <tbody>
          {table.getRowModel().rows.map((r) => {
            const isCurrent = player.current?.rating_key === r.original.rating_key;
            return (
              <tr
                key={r.id}
                onContextMenu={(e) => {
                  e.preventDefault();
                  onContextAction?.(r.original, r.index, e.clientX, e.clientY);
                }}
                className={`group border-b border-[#181b21] ${
                  isCurrent ? "bg-[#15202b]" : "odd:bg-panel2 hover:bg-[#15202b]"
                }`}
              >
                {r.getVisibleCells().map((cell) => {
                  const extra = (cell.column.columnDef.meta as { cellClass?: string } | undefined)?.cellClass ?? "";
                  return (
                    <td key={cell.id} className={`px-3 py-2 align-top ${extra}`}>
                      {flexRender(cell.column.columnDef.cell, cell.getContext())}
                    </td>
                  );
                })}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
```

Note: TanStack's `columnDef.meta` is typed as `unknown`-ish via module augmentation; the inline `as { cellClass?: string } | undefined` cast avoids needing a global `ColumnMeta` augmentation. Leave the empty-state early return (`if (tracks.length === 0) …`) unchanged.

- [ ] **Step 3c: Make QualityStats wrap instead of overflow**

In `web/src/components/QualityStats.tsx`, change the outer row (currently
`<div className="flex items-center gap-5 px-3 py-2 border-b border-border bg-panel2">`) to:

```tsx
    <div className="flex flex-wrap items-center gap-x-5 gap-y-2 px-3 py-2 border-b border-border bg-panel2">
```

And change the export-buttons wrapper (currently `<div className="ml-auto flex gap-2">`) to keep it pushed right but allow it to wrap onto its own line:

```tsx
      <div className="ml-auto flex gap-2 shrink-0">
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npm --prefix web run test -- TrackTable`
Expected: PASS.

- [ ] **Step 5: Run the full unit suite**

Run: `npm --prefix web run test`
Expected: PASS (all).

- [ ] **Step 6: Commit**

```bash
git add web/src/components/TrackTable.tsx web/src/components/QualityStats.tsx web/src/components/TrackTable.test.tsx
git commit -m "feat(web): drop Last.fm column + wrap stats on narrow containers

Container-query column hiding via TanStack column meta; QualityStats row
wraps instead of overflowing.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Global quality floor — focus ring, reduced motion, readable type on touch

**Files:**
- Modify: `web/src/index.css`

**Interfaces:**
- Consumes: nothing.
- Produces: global CSS only; no API change. Verified visually in Task 6 + lint/build.

This is CSS-only (jsdom cannot meaningfully test it), so its check is build + the Playwright pass in Task 6. No unit test.

- [ ] **Step 1: Append the quality-floor rules**

Append to `web/src/index.css` (after the existing `body { … }` line):

```css
/* ── Responsive / accessibility quality floor ──────────────────────────── */

/* Visible keyboard focus everywhere (mouse clicks stay ring-free). */
:focus-visible {
  outline: 2px solid var(--color-accent);
  outline-offset: 1px;
  border-radius: 2px;
}

/* Honor reduced-motion: kill non-essential transitions/animations. */
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.01ms !important;
    scroll-behavior: auto !important;
  }
}

/* On coarse pointers (touch), lift the tiny desktop type to a legible floor
   so the dense toolbar/table stay readable on a phone. Targets the project's
   smallest hard-coded sizes without touching desktop rendering. */
@media (pointer: coarse) {
  body { font-size: 14px; }
  [class*="text-[9px]"]  { font-size: 11px; }
  [class*="text-[10px]"] { font-size: 12px; }
  [class*="text-[11px]"] { font-size: 13px; }
}
```

- [ ] **Step 2: Verify the production build still compiles**

Run: `npm --prefix web run build`
Expected: build succeeds (tsc + vite), no CSS errors.

- [ ] **Step 3: Commit**

```bash
git add web/src/index.css
git commit -m "feat(web): a11y/responsive quality floor (focus ring, reduced motion, touch type)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Playwright multi-viewport verification (the evidence gate)

**Files:**
- Create: `web/tests/responsive.spec.ts`

**Interfaces:**
- Consumes: the running app (Playwright `webServer` from `playwright.config.ts` builds `dist`, serves on `:8771` with the fake worker). Relies on DOM contracts: `data-testid="seed-input"`, `mobile-tabbar`, `tab-mobile-*` (Task 2), `track-table` (existing).
- Produces: screenshots under `web/test-results/responsive/` + pass/fail assertions.

The fake worker returns a 2-track playlist (see `tests/generate.spec.ts`). This test exercises the real reflow at four widths and is the success gate for the whole feature.

- [ ] **Step 1: Write the test**

Create `web/tests/responsive.spec.ts`:

```ts
import { test, expect, type Page } from "@playwright/test";

const WIDTHS = [375, 768, 1024, 1440] as const;

async function generate(page: Page) {
  await page.getByTestId("seed-input").fill("Acetone");
  await page.getByRole("button", { name: /Generate/ }).click();
  await expect(page.getByTestId("track-table")).toBeVisible({ timeout: 15000 });
}

for (const width of WIDTHS) {
  test(`responsive @ ${width}px: no collapse, no overflow`, async ({ page }) => {
    await page.setViewportSize({ width, height: 900 });
    await page.goto("/");
    await expect(page.getByText("Playlist Generator")).toBeVisible();

    // 1) The Artist input must never collapse (the original bug).
    const seed = page.getByTestId("seed-input");
    await expect(seed).toBeVisible();
    const box = await seed.boundingBox();
    expect(box, "seed input has a layout box").not.toBeNull();
    expect(box!.width, `seed input width @ ${width}px`).toBeGreaterThan(120);

    // 2) No horizontal overflow of the document.
    const overflow = await page.evaluate(
      () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
    );
    expect(overflow, `horizontal overflow @ ${width}px`).toBeLessThanOrEqual(1);

    // 3) Shell mode matches the breakpoint.
    if (width < 1024) {
      await expect(page.getByTestId("mobile-tabbar")).toBeVisible();
    } else {
      await expect(page.getByTestId("mobile-tabbar")).toHaveCount(0);
    }

    // 4) Generate works and the table renders without overflowing either.
    await generate(page);
    const overflowAfter = await page.evaluate(
      () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
    );
    expect(overflowAfter, `overflow after generate @ ${width}px`).toBeLessThanOrEqual(1);

    await page.screenshot({ path: `test-results/responsive/${width}.png`, fullPage: true });
  });
}

test("mobile tab bar swaps regions @ 375px", async ({ page }) => {
  await page.setViewportSize({ width: 375, height: 900 });
  await page.goto("/");
  await page.getByTestId("tab-mobile-jobs").click();
  await expect(page.getByTestId("jobs-panel")).toBeVisible();
  await page.getByTestId("tab-mobile-logs").click();
  await expect(page.getByTestId("log-panel")).toBeVisible();
});
```

- [ ] **Step 2: Run the responsive spec**

Run: `npm --prefix web run test:e2e -- responsive`
Expected: PASS (5 tests). If a width fails on overflow or input width, fix the offending component's wrap/container classes and re-run before proceeding. Screenshots land in `web/test-results/responsive/`.

- [ ] **Step 3: Inspect the four screenshots**

Open `web/test-results/responsive/375.png`, `768.png`, `1024.png`, `1440.png`. Confirm visually: Artist input full-width and usable at 375/768; advanced controls collapsed behind "More controls" at 375 and visible at 768+; desktop 3-pane intact at 1024/1440; no clipped/overlapping elements. (This is the human-auditable evidence the task is done.)

- [ ] **Step 4: Run the full e2e suite to confirm no regression**

Run: `npm --prefix web run test:e2e`
Expected: PASS (existing specs + responsive). The desktop-shell specs must still pass (they run at the Playwright default viewport, ≥1024px → desktop shell).

- [ ] **Step 5: Commit**

```bash
git add web/tests/responsive.spec.ts
git commit -m "test(web): Playwright multi-viewport responsive verification (375/768/1024/1440)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:**
- Container queries not viewport breakpoints → Tasks 3 (`@container`/`@md`/`@max-md`), 4 (`@container`/`@max-md`); single viewport query confined to Task 1/2. ✓
- Shell: ≥1024 unchanged, <1024 single column + bottom tab bar (Playlist·Jobs·Diag·Logs) → Task 2. ✓
- Toolbar three tiers (wide / wrap / disclosure) → Task 3 (flex-wrap = medium wrap tier; `@md` disclosure = narrow tier; unchanged = wide). ✓ Disclosure-on-phone decision honored.
- TrackTable column priority (drop Last.fm first) → Task 4. ✓ (Genres already render inline under the title and wrap; no move needed.)
- Type & touch targets → Task 5 (`pointer: coarse` type floor; tab-bar buttons `py-3` ≈ 40px in Task 2). ✓
- Quality floor (focus, reduced motion) → Task 5. ✓
- Playwright verification at 375/768/1024/1440 + screenshots → Task 6. ✓
- No new deps/plugins → Global Constraints + Non-goals. ✓
- Risk: regions render standalone below lg — DiagnosticsPanel/LogPanel/JobsPanel are already self-contained (`h-full`/`p-*`/`overflow-auto`, no resizable-Panel assumptions); Task 2 mounts them directly. ✓
- Risk: react-resizable-panels size persistence across remount — current code sets no `autoSaveId`, so crossing the breakpoint resets to `defaultSize` (acceptable; noted in spec). ✓

**Placeholder scan:** No TBD/TODO; every code step shows full code; commands have expected output. ✓

**Type consistency:** `useMediaQuery(query: string): boolean` consistent across Tasks 1–2. `Region` union and `tab-mobile-${id}` testids consistent in Task 2 + Task 6. `meta.cellClass` shape (`{ cellClass?: string }`) identical in column def and both th/td reads (Task 4). DOM testids referenced in Task 6 (`seed-input`, `mobile-tabbar`, `tab-mobile-*`, `track-table`, `jobs-panel`, `log-panel`) all exist (existing) or are introduced in Task 2. ✓
