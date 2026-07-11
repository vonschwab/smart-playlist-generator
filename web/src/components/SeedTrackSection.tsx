import { useEffect, useRef, useState } from "react";
import { api } from "../lib/api";
import { usePlayer } from "../contexts/PlayerContext";
import type { SeedTrack, TrackOut } from "../lib/types";
import { GenreChips } from "./GenreChips";
import { useInfiniteSearch } from "../lib/useInfiniteSearch";
import { chip as CHIP } from "../lib/ui";

function seedToTrackOut(seed: SeedTrack, index: number): TrackOut {
  return {
    position: index + 1,
    rating_key: seed.track_id,
    title: seed.title,
    artist: seed.artist,
    album: seed.album,
    duration_ms: seed.duration_ms,
    file_path: seed.file_path,
    genres: seed.genres,
  };
}

export function SeedTrackSection({
  tracks,
  onAdd,
  onRemove,
  onClear,
}: {
  tracks: SeedTrack[];
  onAdd: (track: SeedTrack) => void;
  onRemove: (id: string) => void;
  onClear: () => void;
}) {
  const search = useInfiniteSearch<SeedTrack>({
    fetchPage: api.searchTracks,
    pageSize: 25,
    maxItems: 400,
  });
  const listRef = useRef<HTMLUListElement>(null);
  const dropdownRef = useRef<HTMLDivElement>(null);
  const player = usePlayer();
  const trackOuts = tracks.map(seedToTrackOut);

  // Graph-canonical genres for staged seeds, fetched when the seed SET changes
  // (not per keystroke — the search dropdown keeps its metadata genres).
  // Until the fetch lands (or if it fails), t.genres is the placeholder.
  const [canonGenres, setCanonGenres] = useState<Record<string, string[]>>({});
  const idsKey = tracks.map((t) => t.track_id).join("|");
  useEffect(() => {
    if (tracks.length === 0) {
      setCanonGenres({});
      return;
    }
    let cancelled = false;
    api
      .trackGenres(tracks.map((t) => t.track_id))
      .then((m) => {
        if (!cancelled) setCanonGenres(m);
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [idsKey]);

  useEffect(() => {
    function onOutside(e: MouseEvent) {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target as Node)) {
        search.reset();
      }
    }
    document.addEventListener("mousedown", onOutside);
    return () => document.removeEventListener("mousedown", onOutside);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function addTrack(track: SeedTrack) {
    if (tracks.some((t) => t.track_id === track.track_id)) return;
    onAdd(track);
    search.reset();
  }

  return (
    <div className="border-b border-border">
      {/* Search bar */}
      <div className="flex items-center border-b border-hairline bg-panel">
        <div className="flex items-center gap-1.5 px-3 py-[7px] border-r border-hairline">
          <span className="text-2xs uppercase tracking-[.08em] text-faint font-medium whitespace-nowrap">seeds</span>
        </div>
        <div ref={dropdownRef} className="relative flex-1 px-3 py-[5px]">
          <input
            data-testid="seed-search-input"
            value={search.query}
            onChange={(e) => search.setQuery(e.target.value)}
            placeholder="Search by title, artist, or album…"
            className="w-full bg-well border border-border rounded text-xs text-text px-2.5 py-1"
          />
          {search.items.length > 0 && (
            <ul
              ref={listRef}
              onScroll={() => {
                const el = listRef.current;
                if (el && el.scrollHeight - el.scrollTop - el.clientHeight < 48) search.loadMore();
              }}
              className="absolute z-20 left-3 right-3 mt-1 bg-panel border border-border rounded shadow-xl max-h-60 overflow-auto"
            >
              {search.items.map((r) => {
                const already = tracks.some((t) => t.track_id === r.track_id);
                return (
                  <li
                    key={r.track_id}
                    onClick={() => !already && addTrack(r)}
                    className={`flex items-center gap-2 px-3 py-2 text-xs border-b border-hairline last:border-b-0 ${
                      already ? "opacity-40 cursor-default" : "hover:bg-rowsel cursor-pointer"
                    }`}
                  >
                    <span className="text-text font-medium truncate min-w-0 flex-1">{r.title}</span>
                    <span className="text-faint truncate shrink-0">{r.artist}</span>
                    {r.genres.slice(0, 2).map((g) => (
                      <span key={g} className={CHIP}>{g}</span>
                    ))}
                  </li>
                );
              })}
              {search.items.length >= 400 && search.hasMore ? (
                <li className="px-3 py-2 text-2xs text-faint">Showing first 400 — refine your search</li>
              ) : (search.loading || search.hasMore) ? (
                <li className="px-3 py-2 text-2xs text-faint">{search.loading ? "Loading…" : "Scroll for more"}</li>
              ) : null}
            </ul>
          )}
        </div>
        {tracks.length > 0 && (
          <div className="flex items-center px-3 border-l border-hairline">
            <button
              onClick={onClear}
              className="text-2xs text-faint hover:text-danger"
            >
              clear all
            </button>
          </div>
        )}
      </div>

      {/* Track table */}
      {tracks.length === 0 ? (
        <div className="text-center text-faint text-xs py-5 bg-bg">
          Search above to add seed tracks
        </div>
      ) : (
        <table className="w-full text-xs bg-bg">
          <tbody>
            {tracks.map((t, i) => {
              const isCurrent = player.current?.rating_key === t.track_id;
              const isPlaying = isCurrent && player.playing;
              return (
                <tr
                  key={t.track_id}
                  className={`group border-b border-hairline ${isCurrent ? "bg-rowsel" : "hover:bg-panel2"}`}
                >
                  {/* Play */}
                  <td className="pl-3 pr-1 py-2 w-7 text-center">
                    <button
                      onClick={() => {
                        if (isCurrent && isPlaying) player.setPlaying(false);
                        else if (isCurrent) player.setPlaying(true);
                        else player.load(trackOuts, i);
                      }}
                      className={`text-sm leading-none ${isCurrent ? "text-accent" : "text-faint opacity-30 group-hover:opacity-80"}`}
                      title={isPlaying ? "Pause" : "Play"}
                    >
                      {isPlaying ? "❚❚" : "▶"}
                    </button>
                  </td>
                  {/* Index */}
                  <td className={`pr-3 py-2 w-7 text-right font-mono text-2xs ${isCurrent ? "text-accent" : "text-faint"}`}>
                    {i + 1}
                  </td>
                  {/* Title + genres */}
                  <td className="py-2 pr-4">
                    <div className="flex items-center gap-1.5 flex-wrap">
                      <span className={`font-medium ${isCurrent ? "text-accent" : "text-text"}`}>{t.title}</span>
                      <GenreChips genres={canonGenres[t.track_id] ?? t.genres} chipClass={CHIP} />
                    </div>
                  </td>
                  {/* Artist / album */}
                  <td className="py-2 pr-4 text-faint whitespace-nowrap">
                    {t.artist}{t.album ? ` — ${t.album}` : ""}
                  </td>
                  {/* Remove */}
                  <td className="py-2 pr-3 w-7 text-center">
                    <button
                      onClick={() => onRemove(t.track_id)}
                      className="text-faint opacity-0 group-hover:opacity-60 pointer-coarse:opacity-60 hover:!opacity-100 hover:text-danger text-base leading-none"
                      title="Remove seed"
                    >
                      ×
                    </button>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      )}
    </div>
  );
}
