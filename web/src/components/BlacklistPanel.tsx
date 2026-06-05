import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "../lib/api";
import type { BlacklistEntry, BlacklistFetchResponse } from "../lib/types";

const DOT: Record<string, string> = { artist: "#ef4444", album: "#f97316", track: "#a855f7" };

export function BlacklistPanel() {
  const [data, setData] = useState<BlacklistFetchResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [q, setQ] = useState("");
  const [suggestions, setSuggestions] = useState<string[]>([]);
  const timer = useRef<number | undefined>(undefined);
  const dropdownRef = useRef<HTMLDivElement>(null);

  const refresh = useCallback(async () => {
    setBusy(true);
    try {
      setData(await api.getBlacklist());
      setError(null);
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }, []);

  useEffect(() => { refresh(); }, [refresh]);

  useEffect(() => {
    if (q.length < 2) { setSuggestions([]); return; }
    window.clearTimeout(timer.current);
    timer.current = window.setTimeout(async () => {
      setSuggestions(await api.autocomplete(q).catch(() => []));
    }, 180);
  }, [q]);

  useEffect(() => {
    function onOutside(e: MouseEvent) {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target as Node)) setSuggestions([]);
    }
    document.addEventListener("mousedown", onOutside);
    return () => document.removeEventListener("mousedown", onOutside);
  }, []);

  async function addArtist() {
    const name = q.trim();
    if (!name) return;
    setBusy(true);
    try {
      await api.blacklistArtist(name);
      setQ("");
      setSuggestions([]);
      await refresh();
    } catch (e) {
      setError(String(e));
      setBusy(false);
    }
  }

  async function remove(entry: BlacklistEntry) {
    setBusy(true);
    try {
      if (entry.scope === "artist") {
        await api.blacklist({ scope: "artist", value: entry.artist ?? "", enabled: false });
      } else if (entry.scope === "album") {
        await api.blacklist({ scope: "album", value: entry.album ?? "", artist: entry.artist ?? "", enabled: false });
      } else {
        await api.blacklist({ track_ids: [entry.track_id ?? ""], enabled: false });
      }
      await refresh();
    } catch (e) {
      setError(String(e));
      setBusy(false);
    }
  }

  const section = (title: string, entries: BlacklistEntry[]) => (
    <div>
      <div className="flex justify-between text-[8px] uppercase tracking-[.08em] text-[#5b6470] mt-2 mb-1">
        <span>{title}</span><span className="text-[#3a3f4b]">{entries.length}</span>
      </div>
      {entries.map((e, i) => (
        <div key={`${e.scope}-${i}`} className="flex items-center gap-1.5 py-0.5 border-b border-[#1a1c21]">
          <span className="w-1.5 h-1.5 rounded-full shrink-0" style={{ background: DOT[e.scope] }} />
          <span className="text-[10px] text-[#8b939d] flex-1 truncate">{e.display_name}</span>
          <button onClick={() => remove(e)} className="text-[#3a3f4b] hover:text-[#ef4444] text-sm leading-none">×</button>
        </div>
      ))}
    </div>
  );

  const empty = data && data.total === 0;

  return (
    <div className="p-3 overflow-y-auto text-xs" data-testid="blacklist-panel">
      <div ref={dropdownRef} className="relative flex gap-1 mb-2">
        <input
          data-testid="blacklist-search"
          value={q}
          onChange={(e) => setQ(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && addArtist()}
          placeholder="Artist to blacklist…"
          className="flex-1 bg-[#0c0e12] border border-[#23262d] rounded text-[10px] text-[#e6e9ec] px-2 py-1"
        />
        <button data-testid="blacklist-add" onClick={addArtist} disabled={busy}
          className="text-[10px] bg-[#1d3a35] text-[#5eead4] rounded px-2 disabled:opacity-50">+ Add</button>
        {suggestions.length > 0 && (
          <ul className="absolute z-20 top-8 left-0 right-0 bg-[#16181d] border border-[#23262d] rounded shadow-xl max-h-40 overflow-auto">
            {suggestions.map((s) => (
              <li key={s} onClick={() => { setQ(s); setSuggestions([]); }}
                className="px-2 py-1 text-[10px] text-[#e6e9ec] hover:bg-[#1e2229] cursor-pointer">{s}</li>
            ))}
          </ul>
        )}
      </div>

      {error && <div className="text-[#ef4444] text-[10px] mb-2">{error}</div>}
      {busy && <div className="text-[#3a3f4b] text-[9px] mb-1">Refreshing…</div>}

      {empty ? (
        <div className="text-[#3a3f4b] text-[10px]">Nothing blacklisted yet. Use the track table context menu or search above.</div>
      ) : data ? (
        <>
          {section("Artists", data.artists)}
          {section("Albums", data.albums)}
          {section("Tracks", data.tracks)}
        </>
      ) : null}
    </div>
  );
}
