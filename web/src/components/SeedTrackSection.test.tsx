import { describe, it, expect, afterEach, vi } from "vitest";
import { render, screen, cleanup } from "@testing-library/react";
import { SeedTrackSection } from "./SeedTrackSection";
import { PlayerProvider } from "../contexts/PlayerContext";
import type { SeedTrack } from "../lib/types";

vi.mock("../lib/api", () => ({
  api: {
    searchTracks: vi.fn(async () => ({ items: [], has_more: false })),
    trackGenres: vi.fn(async () => ({})),
  },
}));

afterEach(() => cleanup());

const seed: SeedTrack = {
  track_id: "t1",
  title: "Sundown",
  artist: "Acetone",
  album: "York Blvd",
  duration_ms: 1000,
  file_path: "/x.mp3",
  genres: ["slowcore"],
};

describe("SeedTrackSection touch affordances", () => {
  it("keeps the remove-seed button visible on coarse pointers", () => {
    render(
      <PlayerProvider>
        <SeedTrackSection tracks={[seed]} onAdd={() => {}} onRemove={() => {}} onClear={() => {}} />
      </PlayerProvider>,
    );
    // Hover-reveal is desktop-only; touch needs it visible (discipline T3).
    expect(screen.getByTitle("Remove seed").className).toContain("pointer-coarse:opacity-60");
  });
});
