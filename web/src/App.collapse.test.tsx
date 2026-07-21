import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import { render, screen, fireEvent, cleanup, act } from "@testing-library/react";

// S5 (docs/UI_UX_DISCIPLINE.md): on the phone, the payoff owns the viewport —
// once a playlist arrives, the controls collapse to a summary bar and the
// track list gets the screen. Desktop never collapses.

const fixtures = vi.hoisted(() => {
  const successJob = {
    job_id: "j-1",
    status: "success",
    stage: "done",
    playlist: {
      name: "Acetone — mix",
      track_count: 1,
      tracks: [
        {
          position: 0,
          rating_key: "t1",
          artist: "Acetone",
          title: "Sundown",
          album: "York Blvd",
          duration_ms: 1000,
          file_path: "x/y.flac",
          genres: [],
        },
      ],
      metrics: {},
    },
  };
  const impl: Record<string, ReturnType<typeof vi.fn>> = {
    generate: vi.fn(async () => ({ job_id: "j-1" })),
    job: vi.fn(async (): Promise<unknown> => successJob),
    jobs: vi.fn(async () => []),
    autocomplete: vi.fn(async () => ({ items: [], has_more: false })),
  };
  const api = new Proxy(impl, {
    get: (t, p: string) => (t[p] ??= vi.fn(async () => ({}))),
  });
  return { api: api as typeof impl };
});

vi.mock("./lib/api", () => ({ api: fixtures.api }));

import App from "./App";

class DeadWebSocket {
  onopen: (() => void) | null = null;
  onmessage: ((ev: { data: string }) => void) | null = null;
  onclose: (() => void) | null = null;
  onerror: (() => void) | null = null;
  readyState = 0;
  constructor(_url: string) {}
  send(_d: string) {}
  close() {}
}

async function generateToCompletion() {
  render(<App />);
  // Flush SP-1's setup gate: App renders a "Checking setup…" spinner until
  // api.getSetupStatus() resolves (mocked to {} -> not needs_setup). Until that
  // microtask flushes, the generator tree (seed-input) isn't mounted.
  await act(async () => {
    await vi.advanceTimersByTimeAsync(0);
  });
  fireEvent.change(screen.getByTestId("seed-input"), { target: { value: "Acetone" } });
  fireEvent.click(screen.getByRole("button", { name: "▸ Generate" }));
  // Two steps: flush api.generate's microtask (schedules the reconcile
  // interval), then advance past the first poll tick.
  await act(async () => {
    await vi.advanceTimersByTimeAsync(0);
  });
  await act(async () => {
    await vi.advanceTimersByTimeAsync(1500);
  });
}

beforeEach(() => {
  vi.useFakeTimers();
  vi.stubGlobal("WebSocket", DeadWebSocket);
  // jsdom lacks ResizeObserver; react-resizable-panels (desktop shell) needs it.
  vi.stubGlobal(
    "ResizeObserver",
    class {
      observe() {}
      unobserve() {}
      disconnect() {}
    },
  );
  // jsdom lacks scrollIntoView; the desktop shell mounts LogPanel eagerly.
  Element.prototype.scrollIntoView = () => {};
  localStorage.clear();
});

afterEach(() => {
  cleanup();
  vi.useRealTimers();
  vi.unstubAllGlobals();
  vi.clearAllMocks();
});

describe("mobile controls collapse (no matchMedia in jsdom -> mobile shell)", () => {
  it("collapses the controls into a summary bar when the playlist arrives", async () => {
    await generateToCompletion();
    // Controls stay mounted (form state survives) but hidden.
    expect(screen.getByTestId("seed-input").closest(".hidden")).not.toBeNull();
    expect(screen.getByTestId("controls-summary")).toBeTruthy();
    expect(screen.getByText(/Acetone — mix/)).toBeTruthy();
  });

  it("reopens the controls from the summary bar", async () => {
    await generateToCompletion();
    fireEvent.click(screen.getByTestId("controls-summary-open"));
    expect(screen.getByTestId("seed-input").closest(".hidden")).toBeNull();
    expect(screen.queryByTestId("controls-summary")).toBeNull();
  });
});

describe("desktop never collapses", () => {
  it("keeps controls expanded after generation when the viewport is desktop", async () => {
    window.matchMedia = vi.fn().mockImplementation((query: string) => ({
      matches: true,
      media: query,
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
    })) as unknown as typeof window.matchMedia;
    await generateToCompletion();
    expect(screen.getByTestId("seed-input").closest(".hidden")).toBeNull();
    expect(screen.queryByTestId("controls-summary")).toBeNull();
  });
});
