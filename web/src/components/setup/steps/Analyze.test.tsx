import { describe, it, expect, afterEach, vi } from "vitest";
import { render, screen, cleanup, waitFor, act } from "@testing-library/react";
import { Analyze } from "./Analyze";
import { api } from "../../../lib/api";
import type { WsEvent } from "../../../lib/types";

// Same pattern as ToolsPanel: mock the WS/reconcile hooks and capture their
// callbacks so the test can drive a terminal job state directly, instead of
// standing up a real WebSocket + polling loop.
let wsHandler: ((e: WsEvent) => void) | null = null;

vi.mock("../../../lib/ws", () => ({
  useWorkerEvents: (cb: (e: WsEvent) => void) => {
    wsHandler = cb;
  },
}));
vi.mock("../../../lib/useJobReconcile", () => ({
  // Backstop path isn't exercised by these tests (the WS path covers the
  // finish() discrimination); keep it inert so it doesn't double-fire.
  useJobReconcile: () => {},
}));

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  wsHandler = null;
});

describe("Analyze", () => {
  it("shows a failure state — not the success text — when the analyze job's terminal state is a FAILURE", async () => {
    vi.spyOn(api, "analyzeLibrary").mockResolvedValue({ job_id: "job-fail" });
    vi.spyOn(api, "job").mockResolvedValue({
      job_id: "job-fail",
      status: "failed",
      stage: "muq",
      error: "MuQ extraction crashed: out of memory",
    } as any);

    render(<Analyze />);

    // Fire the WS "done" event once jobId has landed in state (analyzeLibrary
    // resolves asynchronously); waitFor retries this until the event isn't
    // filtered out and api.job's promise has resolved.
    await waitFor(() => {
      act(() => wsHandler?.({ type: "done", job_id: "job-fail" }));
      expect(screen.getByText(/MuQ extraction crashed/)).toBeTruthy();
    });

    expect(screen.queryByText(/Analysis finished/)).toBeNull();
    expect(screen.getByRole("alert").textContent).toMatch(/Analysis failed/);
  });

  it("shows the success handoff when the analyze job's terminal state is a SUCCESS", async () => {
    vi.spyOn(api, "analyzeLibrary").mockResolvedValue({ job_id: "job-ok" });
    vi.spyOn(api, "job").mockResolvedValue({
      job_id: "job-ok",
      status: "success",
      stage: "verify",
      error: null,
    } as any);

    render(<Analyze />);

    await waitFor(() => {
      act(() => wsHandler?.({ type: "done", job_id: "job-ok" }));
      expect(screen.getByText(/Analysis finished/)).toBeTruthy();
    });

    expect(screen.queryByRole("alert")).toBeNull();
  });
});
