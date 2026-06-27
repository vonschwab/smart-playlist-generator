import { describe, it, expect, afterEach } from "vitest";
import { render, screen, fireEvent, cleanup } from "@testing-library/react";
import { GenerateControls } from "./GenerateControls";

afterEach(() => { cleanup(); localStorage.clear(); });

function renderControls() {
  return render(
    <GenerateControls
      mode="artist"
      onModeChange={() => {}}
      seedTrackIds={[]}
      seedDisplays={[]}
      onSubmit={() => {}}
      busy={false}
    />,
  );
}

describe("GenerateControls disclosure", () => {
  it("renders a collapsed advanced region with a More controls toggle", () => {
    renderControls();
    const toggle = screen.getByTestId("more-controls");
    expect(toggle.getAttribute("aria-expanded")).toBe("false");
    // Collapsed: advanced region carries the container-query hide class.
    expect(screen.getByTestId("advanced-controls").className).toContain("@max-md:hidden");
  });

  it("expands and collapses when the toggle is clicked", () => {
    renderControls();
    const toggle = screen.getByTestId("more-controls");
    fireEvent.click(toggle);
    expect(toggle.getAttribute("aria-expanded")).toBe("true");
    expect(screen.getByTestId("advanced-controls").className).not.toContain("@max-md:hidden");
    fireEvent.click(toggle);
    expect(toggle.getAttribute("aria-expanded")).toBe("false");
    expect(screen.getByTestId("advanced-controls").className).toContain("@max-md:hidden");
  });
});
