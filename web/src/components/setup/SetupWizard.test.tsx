import { describe, it, expect, afterEach } from "vitest";
import { render, screen, fireEvent, cleanup } from "@testing-library/react";
import SetupWizard from "./SetupWizard";

// This project doesn't enable vitest `globals`, so RTL's own auto-cleanup
// (gated on a global `afterEach`) never registers — every test file that
// renders more than once needs this explicitly (see JobsPanel.test.tsx).
afterEach(() => cleanup());

const status = { state: "needs_setup", config_path: "/c", config_exists: false,
  music_directory: null, db_path: null, track_count: null, detail: "", checks: [] } as any;

describe("SetupWizard", () => {
  it("shows the rail with all steps and starts on Welcome", () => {
    render(<SetupWizard status={status} />);
    expect(screen.getByTestId("wizard-rail")).toBeTruthy();
    expect(screen.getByTestId("rail-step-welcome")).toBeTruthy();
    expect(screen.getByTestId("rail-step-analyze")).toBeTruthy();
    expect(screen.getByTestId("step-welcome")).toBeTruthy();
  });
  it("advances Welcome -> Environment on Next", () => {
    render(<SetupWizard status={status} />);
    fireEvent.click(screen.getByTestId("wizard-next"));
    expect(screen.getByTestId("step-environment")).toBeTruthy();
  });
  it("blocks Next on the music step until a folder is chosen", () => {
    render(<SetupWizard status={status} />);
    fireEvent.click(screen.getByTestId("wizard-next")); // -> environment
    fireEvent.click(screen.getByTestId("wizard-next")); // -> music
    expect(screen.getByTestId("wizard-next").hasAttribute("disabled")).toBe(true);
  });
});
