import { describe, it, expect, afterEach } from "vitest";
import { render, screen, cleanup } from "@testing-library/react";
import { JobsPanel } from "./JobsPanel";

afterEach(() => cleanup());

const noop = () => {};

describe("JobsPanel empty state", () => {
  it("names the action that fills the panel instead of a blank void", () => {
    render(<JobsPanel jobs={[]} onSelect={noop} onCancel={noop} onRerun={noop} onClear={noop} />);
    // S2: an empty screen is an invitation to act, not a black void.
    expect(screen.getByText(/No jobs yet/)).toBeTruthy();
  });
});
