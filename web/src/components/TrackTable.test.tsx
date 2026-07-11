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

  it("caps genre chips to 3 on the phone with a +N overflow pill", () => {
    // jsdom has no matchMedia -> useMediaQuery reports mobile.
    render(
      <PlayerProvider>
        <TrackTable tracks={[track({ genres: ["shoegaze", "dreampop", "slowcore", "noise pop", "c86"] })]} />
      </PlayerProvider>,
    );
    expect(screen.getByText("shoegaze")).toBeTruthy();
    expect(screen.getByText("slowcore")).toBeTruthy();
    // The 4th/5th fold into the overflow pill (not rendered as chips).
    expect(screen.queryByText("noise pop")).toBeNull();
    expect(screen.getByTestId("genre-overflow").textContent).toBe("+2");
  });

  it("keeps the row-actions kebab visible on coarse pointers", () => {
    render(
      <PlayerProvider>
        <TrackTable tracks={[track()]} />
      </PlayerProvider>,
    );
    // Hover-reveal is desktop-only affordance; touch needs it always visible
    // (docs/UI_UX_DISCIPLINE.md T3).
    expect(screen.getByTestId("kebab-btn").className).toContain("pointer-coarse:opacity-60");
  });
});
