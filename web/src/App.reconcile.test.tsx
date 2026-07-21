import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import { render, screen, fireEvent, cleanup, act } from "@testing-library/react";
import type { JobOut } from "./lib/types";

// P0 regression (docs/UI_AUDIT_2026-07-09.md): iOS Safari kills the WebSocket in
// a backgrounded tab, so the `done` event never arrives. The Generate flow must
// recover from the authoritative job registry (useJobReconcile), like the tool
// panels already do — these tests deliver NO WebSocket event at all.

const fixtures = vi.hoisted(() => {
  const successJob = {
    job_id: "j-1",
    status: "success",
    stage: "done",
    playlist: {
      name: "Acetone — test",
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
          genres: ["slowcore"],
        },
      ],
      metrics: {},
    },
  };
  const failedJob = {
    job_id: "j-1",
    status: "failed",
    stage: "error",
    error: "seed artist not found",
    playlist: null,
  };
  const impl: Record<string, ReturnType<typeof vi.fn>> = {
    generate: vi.fn(async () => ({ job_id: "j-1" })),
    job: vi.fn(async (): Promise<unknown> => successJob),
    jobs: vi.fn(async () => []),
    autocomplete: vi.fn(async () => ({ items: [], has_more: false })),
  };
  // Any api method the wider component tree touches (blacklist fetch, genre
  // lookups, ...) resolves to a benign empty object instead of crashing.
  const api = new Proxy(impl, {
    get: (t, p: string) => (t[p] ??= vi.fn(async () => ({}))),
  });
  return { successJob, failedJob, api: api as typeof impl };
});

vi.mock("./lib/api", () => ({ api: fixtures.api }));

import App from "./App";

// Inert socket: connects, never delivers anything.
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

async function generateAndWaitForRecovery() {
  render(<App />);
  // Flush SP-1's setup gate: App shows a "Checking setup…" spinner until
  // api.getSetupStatus() resolves (mocked to {} -> not needs_setup). The
  // generator tree (seed-input) isn't mounted until that microtask flushes.
  await act(async () => {
    await vi.advanceTimersByTimeAsync(0);
  });
  fireEvent.change(screen.getByTestId("seed-input"), { target: { value: "Acetone" } });
  fireEvent.click(screen.getByRole("button", { name: "▸ Generate" }));
  // Let api.generate resolve and the pending job id land in state.
  await act(async () => {
    await vi.advanceTimersByTimeAsync(0);
  });
  expect(screen.getByRole("button", { name: "Generating…" })).toBeTruthy();
  // No WS `done` ever arrives; only the reconcile poll can recover.
  await act(async () => {
    await vi.advanceTimersByTimeAsync(1500);
  });
}

beforeEach(() => {
  vi.useFakeTimers();
  vi.stubGlobal("WebSocket", DeadWebSocket);
  localStorage.clear();
});

afterEach(() => {
  cleanup();
  vi.useRealTimers();
  vi.unstubAllGlobals();
  vi.clearAllMocks();
});

describe("generate flow reconcile backstop (dead WebSocket)", () => {
  it("recovers a finished generation and renders the playlist", async () => {
    fixtures.api.job.mockResolvedValue(fixtures.successJob as unknown as JobOut);
    await generateAndWaitForRecovery();
    expect(fixtures.api.job).toHaveBeenCalledWith("j-1");
    expect(screen.getByRole("button", { name: "▸ Generate" })).toBeTruthy();
    expect(screen.getByText("Sundown")).toBeTruthy();
  });

  it("surfaces a failed generation instead of hanging busy", async () => {
    fixtures.api.job.mockResolvedValue(fixtures.failedJob as unknown as JobOut);
    await generateAndWaitForRecovery();
    expect(screen.getByRole("button", { name: "▸ Generate" })).toBeTruthy();
    expect(screen.getByText(/seed artist not found/)).toBeTruthy();
  });
});
