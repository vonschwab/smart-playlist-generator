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

/** Trigger a browser download of the playlist as an .m3u8 file. */
export function downloadM3U8(tracks: TrackOut[], filename = "playlist.m3u8"): void {
  const blob = new Blob([buildM3U8(tracks)], { type: "audio/x-mpegurl" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}
