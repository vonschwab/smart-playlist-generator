import { describe, it, expect, afterEach, vi } from "vitest";
import { render, screen, cleanup } from "@testing-library/react";
import { ToolsPanel } from "./ToolsPanel";

vi.mock("../lib/ws", () => ({ useWorkerEvents: () => {} }));
vi.mock("../lib/useJobReconcile", () => ({ useJobReconcile: () => {} }));
vi.mock("../lib/api", () => ({ api: {} }));

afterEach(() => cleanup());

describe("ToolsPanel stage labels", () => {
  it("glosses every machine stage name with a human description", () => {
    render(<ToolsPanel externalBusy={false} refreshJobs={() => {}} />);
    // S3: pipeline vocabulary (muq, genre-sim, …) never appears unglossed.
    for (const stage of ["scan", "muq", "genre-sim", "artifacts", "verify"]) {
      const el = screen.getByText(stage);
      const title = el.getAttribute("title") || el.closest("label")?.getAttribute("title");
      expect(title, `stage "${stage}" needs a title gloss`).toBeTruthy();
    }
  });
});
