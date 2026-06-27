import { describe, it, expect, vi, afterEach } from "vitest";
import { renderHook, act } from "@testing-library/react";
import { useMediaQuery } from "./useMediaQuery";

// Controllable matchMedia mock: returns a media query list whose `matches`
// reflects a closure variable, and captures the registered change handler.
function installMatchMedia(initial: boolean) {
  let current = initial;
  let handler: ((e: unknown) => void) | null = null;
  window.matchMedia = vi.fn().mockImplementation((query: string) => ({
    get matches() { return current; },
    media: query,
    addEventListener: (_type: string, cb: (e: unknown) => void) => { handler = cb; },
    removeEventListener: vi.fn(),
  })) as unknown as typeof window.matchMedia;
  return {
    flip(v: boolean) { current = v; handler?.({}); },
  };
}

afterEach(() => { vi.restoreAllMocks(); });

describe("useMediaQuery", () => {
  it("returns the initial match state", () => {
    installMatchMedia(true);
    const { result } = renderHook(() => useMediaQuery("(min-width: 1024px)"));
    expect(result.current).toBe(true);
  });

  it("updates when the media query changes", () => {
    const ctl = installMatchMedia(false);
    const { result } = renderHook(() => useMediaQuery("(min-width: 1024px)"));
    expect(result.current).toBe(false);
    act(() => ctl.flip(true));
    expect(result.current).toBe(true);
  });
});
