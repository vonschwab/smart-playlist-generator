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
