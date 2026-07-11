import { describe, it, expect, afterEach, beforeAll } from "vitest";
import { render, screen, cleanup } from "@testing-library/react";
import { LogPanel } from "./LogPanel";

beforeAll(() => {
  // jsdom has no scrollIntoView; the component's autoscroll needs a stub.
  Element.prototype.scrollIntoView = () => {};
});

afterEach(() => cleanup());

describe("LogPanel empty state", () => {
  it("explains what appears here instead of a blank void", () => {
    render(<LogPanel lines={[]} />);
    expect(screen.getByText(/Logs stream here/)).toBeTruthy();
  });

  it("shows only the log lines once there are any", () => {
    render(<LogPanel lines={["INFO: hello"]} />);
    expect(screen.getByText("INFO: hello")).toBeTruthy();
    expect(screen.queryByText(/Logs stream here/)).toBeNull();
  });
});
