# M3U8 export naming + rename-before-export

**Date:** 2026-06-25
**Status:** Design — awaiting approval
**Scope:** Front-end only (`web/src`). No Python/worker/API change.

## Problem

Exporting a playlist as `.m3u8` produces a file named `playlist.m3u8` every time, with
no chance to rename. The Plex export already does the right thing — an
`ExportPlexDialog` with an editable name field pre-filled in `Artist — Date` format.
M3U8 should match: same default name, and a rename step before download.

## Current state (as built)

- **M3U8** (`App.tsx:271`): `onExportM3U8={() => playlist && downloadM3U8(playlist.tracks)}`.
  `downloadM3U8` (`web/src/lib/m3u.ts`) takes `filename = "playlist.m3u8"` — hardcoded,
  one-click instant download. The playlist is built and downloaded entirely client-side
  (a `Blob` + `<a download>`); nothing touches the worker or any endpoint.
- **Plex** (`App.tsx:329`): `ExportPlexDialog` with `defaultName={defaultPlexName()}`.
  `defaultPlexName` (`App.tsx:176`) = `` `${playlist?.tracks[0]?.artist ?? "Playlist"} — ${YYYY-MM-DD}` ``
  using `new Date().toISOString().slice(0,10)`. The dialog resets `name` to the default
  each time it opens.

## Design

### 1. Shared default-name helper

Extract the Plex naming logic into a pure helper so the two exports can't drift:

`web/src/lib/playlistName.ts`
```ts
import type { PlaylistOut } from "./types";

/** Default playlist name: "<first track artist> — <YYYY-MM-DD>". Matches Plex. */
export function defaultPlaylistName(playlist: PlaylistOut | null): string {
  const date = new Date().toISOString().slice(0, 10);
  const artist = playlist?.tracks[0]?.artist ?? "Playlist";
  return `${artist} — ${date}`;
}
```

`App.tsx`'s `defaultPlexName` is replaced by `defaultPlaylistName(playlist)`, and the new
M3U8 dialog uses the same helper. Separator stays the em-dash `—` (confirmed: M3U8 and
Plex names byte-identical).

### 2. Filename sanitization

The entered name becomes a real download filename, so sanitize it (the Plex *title* does
not need this — only the file does). In `web/src/lib/m3u.ts`:

```ts
/** Make a user string safe as a filename and ensure a single .m3u8 extension. */
export function toM3U8Filename(name: string): string {
  const cleaned = name
    .replace(/[/\\:*?"<>|]/g, "")   // strip filesystem-illegal chars
    .replace(/\s+/g, " ")            // collapse whitespace
    .trim();
  const base = cleaned || "playlist";
  return base.toLowerCase().endsWith(".m3u8") ? base : `${base}.m3u8`;
}
```

`downloadM3U8(tracks, name)` runs the raw name through `toM3U8Filename`. Em-dash `—` is a
valid filename character on Windows/macOS/Linux and is preserved.

### 3. `ExportM3U8Dialog` component

`web/src/components/ExportM3U8Dialog.tsx`, mirroring `ExportPlexDialog`'s markup/styling
(Radix `Dialog`, same classes), but **no network call** — confirm just builds + downloads:

- Props: `{ open, onOpenChange, tracks, defaultName }`.
- `useEffect` resets the editable `name` to `defaultName` when `open` flips true (same as Plex).
- Confirm button disabled when `!name.trim()`; on click → `downloadM3U8(tracks, name)` then
  `onOpenChange(false)`. No async/`status` state needed (download is synchronous).
- Title "Export as M3U8"; label "Playlist name"; `data-testid="export-m3u8-dialog"`,
  input `data-testid="m3u8-name"`.

### 4. Wiring (`App.tsx`)

- Add `const [m3u8Open, setM3u8Open] = useState(false)`.
- `onExportM3U8={() => setM3u8Open(true)}` (was an instant download).
- Mount `<ExportM3U8Dialog open={m3u8Open} onOpenChange={setM3u8Open}
  tracks={playlist?.tracks ?? []} defaultName={defaultPlaylistName(playlist)} />`.

## Testing

- **Unit (Vitest):** `defaultPlaylistName` (first-track artist + ISO date; `null`/empty
  playlist → `Playlist — <date>`); `toM3U8Filename` (illegal-char strip, whitespace
  collapse, empty → `playlist.m3u8`, idempotent extension, em-dash preserved).
- **Component/web test:** ↓ M3U8 opens the dialog, field shows the default name, editing +
  confirm triggers a download with the expected filename (stub `URL.createObjectURL` /
  anchor click) and closes the dialog.
- Build `web/dist` (`npm --prefix web run build`) so the served GUI picks it up. No
  serve_web restart required (client-side only).

## Out of scope / non-goals

- Plex export flow is unchanged except for sourcing its default name from the shared helper.
- No date+time disambiguation for same-artist-same-day collisions — the rename field covers
  that manually (matches Plex behavior). YAGNI unless requested.
- No backend, worker, or API changes.
```
