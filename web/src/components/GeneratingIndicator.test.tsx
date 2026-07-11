import { describe, it, expect, afterEach } from "vitest";
import { render, screen, cleanup } from "@testing-library/react";
import { GeneratingIndicator } from "./GeneratingIndicator";

afterEach(() => cleanup());

describe("GeneratingIndicator", () => {
  it("announces progress and surfaces the latest status line", () => {
    render(<GeneratingIndicator status="INFO: beam search…" />);
    expect(screen.getByTestId("generating-indicator")).toBeTruthy();
    expect(screen.getByText(/Building your playlist/)).toBeTruthy();
    expect(screen.getByText("INFO: beam search…")).toBeTruthy();
    // The live region is announced to assistive tech.
    expect(screen.getByRole("status")).toBeTruthy();
  });

  it("renders without a status line", () => {
    render(<GeneratingIndicator />);
    expect(screen.getByTestId("generating-indicator")).toBeTruthy();
    expect(screen.queryByText(/INFO:/)).toBeNull();
  });
});
