import { useState } from "react";

export function useLocalStorage<T>(key: string, defaultValue: T): [T, (v: T) => void] {
  const [value, setValue] = useState<T>(() => {
    try {
      const stored = localStorage.getItem(key);
      return stored !== null ? (JSON.parse(stored) as T) : defaultValue;
    } catch {
      return defaultValue;
    }
  });

  function set(v: T) {
    setValue(v);
    try { localStorage.setItem(key, JSON.stringify(v)); } catch {}
  }

  return [value, set];
}
