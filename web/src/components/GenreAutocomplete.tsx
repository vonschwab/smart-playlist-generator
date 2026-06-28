import { useEffect, useState } from "react";
import { api } from "../lib/api";
import type { CanonicalGenre } from "../lib/types";

interface GenreAutocompleteProps {
  value: string;
  onChange: (v: string) => void;
  /** Called when the user picks a suggestion (defaults to onChange). */
  onPick?: (name: string) => void;
  placeholder?: string;
  className?: string;
  autoFocus?: boolean;
  limit?: number;
}

/**
 * Single-pick autocomplete over the canonical genre vocabulary — the same
 * `/api/genres/search` endpoint the Edit-genres dialog uses (150ms debounce).
 * Picking a suggestion guarantees a real canonical name (so e.g. a taxonomy
 * alias target validates instead of dangling).
 */
export function GenreAutocomplete({
  value, onChange, onPick, placeholder, className, autoFocus, limit = 8,
}: GenreAutocompleteProps) {
  const [suggestions, setSuggestions] = useState<CanonicalGenre[]>([]);

  useEffect(() => {
    const q = value.trim();
    if (!q) { setSuggestions([]); return; }
    const h = setTimeout(() => {
      api.genresSearch(q, limit).then((r) => setSuggestions(r.items)).catch(() => setSuggestions([]));
    }, 150);
    return () => clearTimeout(h);
  }, [value, limit]);

  const pick = (name: string) => {
    (onPick ?? onChange)(name);
    setSuggestions([]);
  };

  return (
    <div className="relative">
      <input
        autoFocus={autoFocus}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && suggestions.length) { e.preventDefault(); pick(suggestions[0].name); }
          if (e.key === "Escape") setSuggestions([]);
        }}
        placeholder={placeholder}
        className={className}
      />
      {suggestions.length > 0 && (
        <div className="absolute z-20 mt-0.5 bg-panel2 border border-border rounded-md max-h-40 overflow-auto min-w-[200px] shadow-lg">
          {suggestions.map((s) => (
            <div key={s.genre_id} onClick={() => pick(s.name)}
                 className="px-2.5 py-1 text-[10px] text-text hover:bg-border cursor-pointer">
              {s.name}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
