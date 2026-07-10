import { useEffect, useState } from "react";
import { api } from "../lib/api";

interface ArtistAutocompleteProps {
  /** Called when the user picks a suggestion (click or Enter) with the raw artist name. */
  onPick: (name: string) => void;
  placeholder?: string;
  className?: string;
  limit?: number;
}

/**
 * Single-pick typeahead over distinct library artist names, backed by
 * `/api/artists/search` (150ms debounce) — structural copy of
 * GenreAutocomplete. Unlike GenreAutocomplete this is self-contained: it
 * owns its own input text and clears itself after a pick, since callers
 * (Artist Links) only want the picked name, not a controlled text field.
 */
export function ArtistAutocomplete({
  onPick, placeholder, className, limit = 8,
}: ArtistAutocompleteProps) {
  const [query, setQuery] = useState("");
  const [suggestions, setSuggestions] = useState<string[]>([]);

  useEffect(() => {
    const q = query.trim();
    if (!q) { setSuggestions([]); return; }
    const h = setTimeout(() => {
      api.artistsSearch(q, limit).then((r) => setSuggestions(r.items)).catch(() => setSuggestions([]));
    }, 150);
    return () => clearTimeout(h);
  }, [query, limit]);

  const pick = (name: string) => {
    onPick(name);
    setQuery("");
    setSuggestions([]);
  };

  return (
    <div className="relative">
      <input
        data-testid="artist-autocomplete-input"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && suggestions.length) { e.preventDefault(); pick(suggestions[0]); }
          if (e.key === "Escape") setSuggestions([]);
        }}
        placeholder={placeholder}
        className={
          className ??
          "w-full min-h-[44px] px-3 py-2 text-base bg-panel2 border border-border rounded-md text-text placeholder:text-muted"
        }
      />
      {suggestions.length > 0 && (
        <div className="absolute z-20 mt-0.5 bg-panel2 border border-border rounded-md max-h-56 overflow-auto min-w-[200px] shadow-lg">
          {suggestions.map((name) => (
            <div
              key={name}
              data-testid="artist-suggestion"
              onClick={() => pick(name)}
              className="min-h-[44px] flex items-center px-3 py-2 text-base text-text hover:bg-border cursor-pointer"
            >
              {name}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
