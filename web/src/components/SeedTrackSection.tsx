import { useEffect, useRef, useState } from "react";
import { api } from "../lib/api";
import { usePlayer } from "../contexts/PlayerContext";
import type { SeedTrack, TrackOut } from "../lib/types";
import { GenreChips } from "./GenreChips";
import { useInfiniteSearch } from "../lib/useInfiniteSearch";

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

const CHIP = "text-[10px] bg-[#1d2937] text-[#7dd3fc] px-1.5 py-0.5 rounded-full shrink-0";

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
    <div className="border-b border-[#23262d]">
      {/* Search bar */}
      <div className="flex items-center border-b border-[#1e2128] bg-[#16181d]">
        <div className="flex items-center gap-1.5 px-3 py-[7px] border-r border-[#1e2128]">
          <span className="text-[9px] uppercase tracking-[.08em] text-[#5b6470] font-medium whitespace-nowrap">seeds</span>
        </div>
        <div ref={dropdownRef} className="relative flex-1 px-3 py-[5px]">
          <input
            data-testid="seed-search-input"
            value={search.query}
            onChange={(e) => search.setQuery(e.target.value)}
            placeholder="Search by title, artist, or album…"
            className="w-full bg-[#0c0e12] border border-[#23262d] rounded text-xs text-[#e6e9ec] px-2.5 py-1"
          />
          {search.items.length > 0 && (
            <ul
              ref={listRef}
              onScroll={() => {
                const el = listRef.current;
                if (el && el.scrollHeight - el.scrollTop - el.clientHeight < 48) search.loadMore();
              }}
              className="absolute z-20 left-3 right-3 mt-1 bg-[#16181d] border border-[#23262d] rounded shadow-xl max-h-60 overflow-auto"
            >
              {search.items.map((r) => {
                const already = tracks.some((t) => t.track_id === r.track_id);
                return (
                  <li
                    key={r.track_id}
                    onClick={() => !already && addTrack(r)}
                    className={`flex items-center gap-2 px-3 py-2 text-xs border-b border-[#1e2128] last:border-b-0 ${
                      already ? "opacity-40 cursor-default" : "hover:bg-[#1e2229] cursor-pointer"
                    }`}
                  >
                    <span className="text-[#e6e9ec] font-medium truncate min-w-0 flex-1">{r.title}</span>
                    <span className="text-[#5b6470] truncate shrink-0">{r.artist}</span>
                    {r.genres.slice(0, 2).map((g) => (
                      <span key={g} className={CHIP}>{g}</span>
                    ))}
                  </li>
                );
              })}
              {search.items.length >= 400 && search.hasMore ? (
                <li className="px-3 py-2 text-[10px] text-[#5b6470]">Showing first 400 — refine your search</li>
              ) : (search.loading || search.hasMore) ? (
                <li className="px-3 py-2 text-[10px] text-[#5b6470]">{search.loading ? "Loading…" : "Scroll for more"}</li>
              ) : null}
            </ul>
          )}
        </div>
        {tracks.length > 0 && (
          <div className="flex items-center px-3 border-l border-[#1e2128]">
            <button
              onClick={onClear}
              className="text-[10px] text-[#5b6470] hover:text-[#ef4444]"
            >
              clear all
            </button>
          </div>
        )}
      </div>

      {/* Track table */}
      {tracks.length === 0 ? (
        <div className="text-center text-[#3a3f4b] text-xs py-5 bg-[#0f1115]">
          Search above to add seed tracks
        </div>
      ) : (
        <table className="w-full text-xs bg-[#0f1115]">
          <tbody>
            {tracks.map((t, i) => {
              const isCurrent = player.current?.rating_key === t.track_id;
              const isPlaying = isCurrent && player.playing;
              return (
                <tr
                  key={t.track_id}
                  className={`group border-b border-[#1a1c21] ${isCurrent ? "bg-[#15202b]" : "hover:bg-[#13151a]"}`}
                >
                  {/* Play */}
                  <td className="pl-3 pr-1 py-2 w-7 text-center">
                    <button
                      onClick={() => {
                        if (isCurrent && isPlaying) player.setPlaying(false);
                        else if (isCurrent) player.setPlaying(true);
                        else player.load(trackOuts, i);
                      }}
                      className={`text-sm leading-none ${isCurrent ? "text-[#5eead4]" : "text-[#5b6470] opacity-30 group-hover:opacity-80"}`}
                      title={isPlaying ? "Pause" : "Play"}
                    >
                      {isPlaying ? "❚❚" : "▶"}
                    </button>
                  </td>
                  {/* Index */}
                  <td className={`pr-3 py-2 w-7 text-right font-mono text-[10px] ${isCurrent ? "text-[#5eead4]" : "text-[#3a3f4b]"}`}>
                    {i + 1}
                  </td>
                  {/* Title + genres */}
                  <td className="py-2 pr-4">
                    <div className="flex items-center gap-1.5 flex-wrap">
                      <span className={`font-medium ${isCurrent ? "text-[#5eead4]" : "text-[#e6e9ec]"}`}>{t.title}</span>
                      <GenreChips genres={canonGenres[t.track_id] ?? t.genres} chipClass={CHIP} />
                    </div>
                  </td>
                  {/* Artist / album */}
                  <td className="py-2 pr-4 text-[#5b6470] whitespace-nowrap">
                    {t.artist}{t.album ? ` — ${t.album}` : ""}
                  </td>
                  {/* Remove */}
                  <td className="py-2 pr-3 w-7 text-center">
                    <button
                      onClick={() => onRemove(t.track_id)}
                      className="text-[#5b6470] opacity-0 group-hover:opacity-60 hover:!opacity-100 hover:text-[#ef4444] text-base leading-none"
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
