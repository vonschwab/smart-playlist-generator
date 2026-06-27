import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { defaultPlaylistName } from "./playlistName";
import type { PlaylistOut, TrackOut } from "./types";

function track(artist: string): TrackOut {
  return {
    position: 0,
    artist,
    title: "T",
    album: "",
    duration_ms: 0,
    file_path: "/x.mp3",
    genres: [],
  };
}

function playlist(...artists: string[]): PlaylistOut {
  return {
    name: "",
    track_count: artists.length,
    tracks: artists.map(track),
    metrics: {},
  };
}

beforeEach(() => {
  vi.useFakeTimers();
  vi.setSystemTime(new Date("2026-06-25T12:00:00Z"));
});
afterEach(() => vi.useRealTimers());

describe("defaultPlaylistName", () => {
  it("formats as '<first track artist> — <YYYY-MM-DD>' (em-dash)", () => {
    expect(defaultPlaylistName(playlist("Alvvays", "Slowdive"))).toBe("Alvvays — 2026-06-25");
  });

  it("falls back to 'Playlist' when the playlist is null", () => {
    expect(defaultPlaylistName(null)).toBe("Playlist — 2026-06-25");
  });

  it("falls back to 'Playlist' when there are no tracks", () => {
    expect(defaultPlaylistName(playlist())).toBe("Playlist — 2026-06-25");
  });
});
