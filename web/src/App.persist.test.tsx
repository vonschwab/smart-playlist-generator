import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import { render, screen, fireEvent, cleanup, act } from "@testing-library/react";

// A generated playlist must survive a page reload (localStorage), and a loading
// indicator must show while a generation is in flight.

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

// Flush SP-1's setup gate: App shows a "Checking setup…" spinner until
// api.getSetupStatus() resolves (mocked to {} -> not needs_setup). The
// generator tree (seed-input) isn't mounted until that microtask flushes.
async function flushSetupGate() {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(0);
  });
}

async function startGenerate() {
  render(<App />);
  await flushSetupGate();
  fireEvent.change(screen.getByTestId("seed-input"), { target: { value: "Acetone" } });
  fireEvent.click(screen.getByRole("button", { name: "▸ Generate" }));
}

async function generateToCompletion() {
  await startGenerate();
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
  vi.stubGlobal(
    "ResizeObserver",
    class {
      observe() {}
      unobserve() {}
      disconnect() {}
    },
  );
  Element.prototype.scrollIntoView = () => {};
  localStorage.clear();
});

afterEach(() => {
  cleanup();
  vi.useRealTimers();
  vi.unstubAllGlobals();
  vi.clearAllMocks();
});

describe("playlist persistence across a reload", () => {
  it("restores the last playlist on a fresh mount without regenerating", async () => {
    await generateToCompletion();
    expect(screen.getByText("Sundown")).toBeTruthy();

    // Simulate a page reload: unmount, then mount a fresh App (localStorage
    // survives). No generation happens.
    cleanup();
    fixtures.api.generate.mockClear();
    render(<App />);
    await flushSetupGate();

    expect(screen.getByText("Sundown")).toBeTruthy();
    expect(fixtures.api.generate).not.toHaveBeenCalled();
  });
});

describe("loading indicator", () => {
  it("shows while generating and hides once the playlist arrives", async () => {
    await startGenerate();
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });
    expect(screen.getByTestId("generating-indicator")).toBeTruthy();

    await act(async () => {
      await vi.advanceTimersByTimeAsync(1500);
    });
    expect(screen.queryByTestId("generating-indicator")).toBeNull();
    expect(screen.getByText("Sundown")).toBeTruthy();
  });

  it("is absent on first load", async () => {
    render(<App />);
    // After the setup gate resolves (its own "Checking setup…" spinner also
    // uses generating-indicator), the GENERATION indicator must be absent
    // since nothing has been generated yet.
    await flushSetupGate();
    expect(screen.queryByTestId("generating-indicator")).toBeNull();
  });
});
