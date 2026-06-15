import { useCallback, useEffect, useRef, useState } from "react";
import type { Page } from "./types";

export interface UseInfiniteSearchOptions<T> {
  fetchPage: (q: string, offset: number, limit: number) => Promise<Page<T>>;
  minChars?: number;
  firstDebounceMs?: number;
  prefetchDelayMs?: number;
  pageSize?: number;
  maxItems?: number;
}

export interface UseInfiniteSearchResult<T> {
  query: string;
  setQuery: (q: string) => void;
  items: T[];
  loading: boolean;
  hasMore: boolean;
  loadMore: () => void;
  reset: () => void;
}

export function useInfiniteSearch<T>(opts: UseInfiniteSearchOptions<T>): UseInfiniteSearchResult<T> {
  const {
    fetchPage,
    minChars = 2,
    firstDebounceMs = 200,
    prefetchDelayMs = 500,
    pageSize = 25,
    maxItems,
  } = opts;

  const [query, setQueryState] = useState("");
  const [items, setItems] = useState<T[]>([]);
  const [loading, setLoading] = useState(false);
  const [hasMore, setHasMore] = useState(false);

  // Refs mirror state so async callbacks read current values without re-subscribing.
  const reqId = useRef(0);            // monotonic; only the latest fetch may apply
  const itemsRef = useRef<T[]>([]);
  const queryRef = useRef("");
  const hasMoreRef = useRef(false);
  const loadingRef = useRef(false);
  const debounceTimer = useRef<number | undefined>(undefined);
  const prefetchTimer = useRef<number | undefined>(undefined);

  const clearTimers = () => {
    window.clearTimeout(debounceTimer.current);
    window.clearTimeout(prefetchTimer.current);
  };

  const apply = (next: T[], more: boolean) => {
    itemsRef.current = next;
    hasMoreRef.current = more;
    setItems(next);
    setHasMore(more);
  };

  const runFetch = useCallback((q: string, offset: number, isFirst: boolean) => {
    const myId = ++reqId.current;
    loadingRef.current = true;
    setLoading(true);
    fetchPage(q, offset, pageSize)
      .then((page) => {
        if (myId !== reqId.current || q !== queryRef.current) return; // stale
        const merged = isFirst ? page.items : itemsRef.current.concat(page.items);
        apply(merged, page.has_more);
        loadingRef.current = false;
        setLoading(false);
        if (isFirst && page.has_more && (maxItems === undefined || merged.length < maxItems)) {
          window.clearTimeout(prefetchTimer.current);
          prefetchTimer.current = window.setTimeout(() => {
            if (q === queryRef.current && hasMoreRef.current && !loadingRef.current) {
              runFetch(q, itemsRef.current.length, false);
            }
          }, prefetchDelayMs);
        }
      })
      .catch(() => {
        if (myId !== reqId.current) return;
        loadingRef.current = false;
        setLoading(false); // keep existing items; stop paging silently
      });
  }, [fetchPage, pageSize, prefetchDelayMs, maxItems]);

  const setQuery = useCallback((q: string) => {
    queryRef.current = q;
    setQueryState(q);
    clearTimers();
    reqId.current += 1; // invalidate any in-flight fetch for the old query
    if (q.length < minChars) {
      itemsRef.current = [];
      hasMoreRef.current = false;
      loadingRef.current = false;
      setItems([]);
      setHasMore(false);
      setLoading(false);
      return;
    }
    debounceTimer.current = window.setTimeout(() => runFetch(q, 0, true), firstDebounceMs);
  }, [minChars, firstDebounceMs, runFetch]);

  const loadMore = useCallback(() => {
    if (loadingRef.current || !hasMoreRef.current) return;
    if (maxItems !== undefined && itemsRef.current.length >= maxItems) return;
    if (queryRef.current.length < minChars) return;
    runFetch(queryRef.current, itemsRef.current.length, false);
  }, [maxItems, minChars, runFetch]);

  const reset = useCallback(() => {
    reqId.current += 1;
    clearTimers();
    queryRef.current = "";
    itemsRef.current = [];
    hasMoreRef.current = false;
    loadingRef.current = false;
    setQueryState("");
    setItems([]);
    setHasMore(false);
    setLoading(false);
  }, []);

  useEffect(() => () => clearTimers(), []);

  return { query, setQuery, items, loading, hasMore, loadMore, reset };
}
