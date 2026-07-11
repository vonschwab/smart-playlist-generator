import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen, fireEvent, cleanup } from "@testing-library/react";
import { Shell } from "./Shell";

// Force the mobile shell: matchMedia('(min-width:1024px)') -> false.
function forceMobile() {
  window.matchMedia = vi.fn().mockImplementation((query: string) => ({
    matches: false,
    media: query,
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
  })) as unknown as typeof window.matchMedia;
}

afterEach(() => { cleanup(); vi.restoreAllMocks(); });

function renderShell() {
  return render(
    <Shell
      topBar={<div>TOPBAR</div>}
      jobs={<div data-testid="stub-jobs">JOBS</div>}
      center={<div data-testid="stub-center">CENTER</div>}
      right={<div data-testid="stub-right">RIGHT</div>}
      logs={<div data-testid="stub-logs">LOGS</div>}
    />,
  );
}

describe("Shell (mobile)", () => {
  it("sizes the app to the dynamic viewport (iOS Safari toolbar)", () => {
    forceMobile();
    const { container } = renderShell();
    // h-screen (100vh) is the iOS trap: the bottom tab bar sits under Safari's
    // toolbar. dvh tracks the real visible viewport (discipline V1).
    expect((container.firstElementChild as HTMLElement).className).toContain(
      "supports-[height:100dvh]:h-dvh",
    );
  });

  it("renders the bottom tab bar and defaults to the center region", () => {
    forceMobile();
    renderShell();
    expect(screen.getByTestId("mobile-tabbar")).toBeTruthy();
    expect(screen.getByTestId("stub-center")).toBeTruthy();
    // Secondary regions are not mounted until their tab is selected.
    expect(screen.queryByTestId("stub-jobs")).toBeNull();
  });

  it("switches the active region when a tab is tapped", () => {
    forceMobile();
    renderShell();
    fireEvent.click(screen.getByTestId("tab-mobile-jobs"));
    expect(screen.getByTestId("stub-jobs")).toBeTruthy();
    expect(screen.queryByTestId("stub-center")).toBeNull();

    fireEvent.click(screen.getByTestId("tab-mobile-diag"));
    expect(screen.getByTestId("stub-right")).toBeTruthy();

    fireEvent.click(screen.getByTestId("tab-mobile-logs"));
    expect(screen.getByTestId("stub-logs")).toBeTruthy();
  });
});
