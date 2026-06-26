import type { TrackOut } from "./types";

/** Build an M3U8 playlist string from tracks (uses local file_path per track). */
export function buildM3U8(tracks: TrackOut[]): string {
  const lines = ["#EXTM3U"];
  for (const t of tracks) {
    const seconds = Math.round((t.duration_ms ?? 0) / 1000);
    lines.push(`#EXTINF:${seconds},${t.artist} - ${t.title}`);
    lines.push(t.file_path);
  }
  return lines.join("\n") + "\n";
}

/**
 * Turn a user-entered playlist name into a safe download filename, ensuring a
 * single `.m3u8` extension. Filesystem-illegal characters are stripped; the
 * em-dash used by the default name is a valid filename character and kept.
 */
export function toM3U8Filename(name: string): string {
  const cleaned = name
    .replace(/[/\\:*?"<>|]/g, "") // strip filesystem-illegal characters
    .replace(/\s+/g, " ") // collapse internal whitespace
    .trim();
  const base = cleaned || "playlist";
  return base.toLowerCase().endsWith(".m3u8") ? base : `${base}.m3u8`;
}

/**
 * Trigger a browser download of the playlist as an .m3u8 file. `name` is the
 * user-facing playlist name (e.g. "Alvvays — 2026-06-25"); it is sanitized into
 * a filename via {@link toM3U8Filename}.
 */
export function downloadM3U8(tracks: TrackOut[], name = "playlist"): void {
  const blob = new Blob([buildM3U8(tracks)], { type: "audio/x-mpegurl" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = toM3U8Filename(name);
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}
