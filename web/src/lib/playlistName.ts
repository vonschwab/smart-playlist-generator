import type { PlaylistOut } from "./types";

/**
 * Default playlist name shared by the M3U8 and Plex exports:
 * "<first track artist> — <YYYY-MM-DD>" (em-dash, matching Plex).
 * Falls back to "Playlist" when there is no playlist or it has no tracks.
 */
export function defaultPlaylistName(playlist: PlaylistOut | null): string {
  const date = new Date().toISOString().slice(0, 10);
  const artist = playlist?.tracks[0]?.artist ?? "Playlist";
  return `${artist} — ${date}`;
}
