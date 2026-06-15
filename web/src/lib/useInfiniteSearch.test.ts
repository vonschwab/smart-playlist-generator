import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { renderHook, act } from "@testing-library/react";
import { useInfiniteSearch } from "./useInfiniteSearch";
import type { Page } from "./types";

function deferred<T>() {
  let resolve!: (v: T) => void;
  const promise = new Promise<T>((r) => { resolve = r; });
  return { promise, resolve };
}

beforeEach(() => vi.useFakeTimers());
afterEach(() => vi.useRealTimers());

describe("useInfiniteSearch", () => {
  it("ignores a stale out-of-order response from a previous query", async () => {
    const d1 = deferred<Page<string>>();
    const d2 = deferred<Page<string>>();
    const fetchPage = vi.fn<(q: string, o: number, l: number) => Promise<Page<string>>>()
      .mockReturnValueOnce(d1.promise)
      .mockReturnValueOnce(d2.promise);
    const { result } = renderHook(() =>
      useInfiniteSearch<string>({ fetchPage, firstDebounceMs: 100, prefetchDelayMs: 999999, pageSize: 10 }));

    act(() => result.current.setQuery("aa"));
    act(() => { vi.advanceTimersByTime(100); }); // debounce -> fetch1 (query "aa")
    act(() => result.current.setQuery("ab"));
    act(() => { vi.advanceTimersByTime(100); }); // debounce -> fetch2 (query "ab")

    await act(async () => { d2.resolve({ items: ["B1", "B2"], has_more: false }); });
    await act(async () => { d1.resolve({ items: ["A1"], has_more: false }); }); // late + stale

    expect(result.current.items).toEqual(["B1", "B2"]);
  });

  it("loadMore is a no-op while loading and when there is no more", async () => {
    const d1 = deferred<Page<string>>();
    const fetchPage = vi.fn<(q: string, o: number, l: number) => Promise<Page<string>>>()
      .mockReturnValueOnce(d1.promise);
    const { result } = renderHook(() =>
      useInfiniteSearch<string>({ fetchPage, firstDebounceMs: 0, prefetchDelayMs: 999999, pageSize: 10 }));

    act(() => result.current.setQuery("aa"));
    act(() => { vi.advanceTimersByTime(0); });
    act(() => result.current.loadMore());          // loading -> ignored
    expect(fetchPage).toHaveBeenCalledTimes(1);

    await act(async () => { d1.resolve({ items: ["A1"], has_more: false }); });
    act(() => result.current.loadMore());          // hasMore false -> ignored
    expect(fetchPage).toHaveBeenCalledTimes(1);
  });

  it("auto-prefetches exactly one extra page after a pause, then stops", async () => {
    const fetchPage = vi.fn<(q: string, o: number, l: number) => Promise<Page<string>>>()
      .mockResolvedValueOnce({ items: ["A1", "A2"], has_more: true })
      .mockResolvedValueOnce({ items: ["A3", "A4"], has_more: true });
    const { result } = renderHook(() =>
      useInfiniteSearch<string>({ fetchPage, firstDebounceMs: 100, prefetchDelayMs: 500, pageSize: 2 }));

    act(() => result.current.setQuery("aa"));
    await act(async () => { vi.advanceTimersByTime(100); });
    expect(result.current.items).toEqual(["A1", "A2"]);

    await act(async () => { vi.advanceTimersByTime(500); }); // pause-prefetch one page
    expect(result.current.items).toEqual(["A1", "A2", "A3", "A4"]);
    expect(fetchPage).toHaveBeenCalledTimes(2);
    expect(fetchPage).toHaveBeenNthCalledWith(2, "aa", 2, 2); // offset advances by pageSize

    await act(async () => { vi.advanceTimersByTime(2000); }); // no further auto-fetch
    expect(fetchPage).toHaveBeenCalledTimes(2);
  });

  it("clears items immediately when the query drops below minChars", async () => {
    const fetchPage = vi.fn<(q: string, o: number, l: number) => Promise<Page<string>>>()
      .mockResolvedValueOnce({ items: ["A1", "A2"], has_more: false });
    const { result } = renderHook(() =>
      useInfiniteSearch<string>({ fetchPage, firstDebounceMs: 0, prefetchDelayMs: 999999, pageSize: 10 }));

    act(() => result.current.setQuery("abc"));
    await act(async () => { vi.advanceTimersByTime(0); });
    expect(result.current.items).toEqual(["A1", "A2"]);

    act(() => result.current.setQuery("a")); // below minChars (2)
    expect(result.current.items).toEqual([]);
  });

  it("reset discards an in-flight response and clears state", async () => {
    const d1 = deferred<Page<string>>();
    const fetchPage = vi.fn<(q: string, o: number, l: number) => Promise<Page<string>>>()
      .mockReturnValueOnce(d1.promise);
    const { result } = renderHook(() =>
      useInfiniteSearch<string>({ fetchPage, firstDebounceMs: 0, prefetchDelayMs: 999999, pageSize: 10 }));

    act(() => result.current.setQuery("aa"));
    act(() => { vi.advanceTimersByTime(0); });
    act(() => result.current.reset());
    await act(async () => { d1.resolve({ items: ["A1"], has_more: true }); });

    expect(result.current.items).toEqual([]);
    expect(result.current.query).toBe("");
  });
});
