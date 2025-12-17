import { useEffect, useMemo, useState } from "react";

type TrackResult = {
  track_id: string;
  title: string;
  artist: string;
  album: string;
};

type ArtistResult = {
  artist_name: string;
  track_count: number;
};

export type Seed =
  | { type: "track"; track_id: string; title: string; artist: string; album: string }
  | { type: "artist"; artist_name: string };

type Tab = "tracks" | "artists";

type Props = {
  open: boolean;
  onClose: () => void;
  onSelect: (seed: Seed) => void;
  apiBase: string;
};

const DEBOUNCE_MS = 250;

export function SeedPickerModal({ open, onClose, onSelect, apiBase }: Props) {
  const [tab, setTab] = useState<Tab>("tracks");
  const [query, setQuery] = useState("a");
  const [isLoading, setIsLoading] = useState(false);
  const [trackResults, setTrackResults] = useState<TrackResult[]>([]);
  const [artistResults, setArtistResults] = useState<ArtistResult[]>([]);

  useEffect(() => {
    if (!open) return;
    setQuery("a");
    setTab("tracks");
  }, [open]);

  useEffect(() => {
    if (!open) return;
    if (!query.trim()) {
      setTrackResults([]);
      setArtistResults([]);
      return;
    }

    const handle = setTimeout(async () => {
      setIsLoading(true);
      try {
        const endpoint = tab === "tracks" ? "tracks" : "artists";
        const res = await fetch(`${apiBase}/api/search/${endpoint}?q=${encodeURIComponent(query)}&limit=20`);
        if (!res.ok) {
          throw new Error(`Search failed (${res.status})`);
        }
        const data = (await res.json()) as { results: unknown[] };
        if (tab === "tracks") {
          setTrackResults(data.results as TrackResult[]);
        } else {
          setArtistResults(data.results as ArtistResult[]);
        }
      } catch {
        setTrackResults([]);
        setArtistResults([]);
      } finally {
        setIsLoading(false);
      }
    }, DEBOUNCE_MS);

    return () => clearTimeout(handle);
  }, [apiBase, open, query, tab]);

  const content = useMemo(() => {
    if (isLoading) return <p className="muted">Searching…</p>;
    if (tab === "tracks" && trackResults.length === 0)
      return <p className="muted">No tracks found.</p>;
    if (tab === "artists" && artistResults.length === 0)
      return <p className="muted">No artists found.</p>;

    if (tab === "tracks") {
      return (
        <div className="results">
          {trackResults.map((t) => (
            <button
              key={t.track_id}
              className="result-row"
              onClick={() =>
                onSelect({
                  type: "track",
                  track_id: t.track_id,
                  title: t.title,
                  artist: t.artist,
                  album: t.album,
                })
              }
            >
              <div>
                <p className="track-title">
                  {t.title} <span className="muted">— {t.artist}</span>
                </p>
                <p className="muted tiny">{t.album || "Unknown album"}</p>
              </div>
              <span className="pill">Select</span>
            </button>
          ))}
        </div>
      );
    }

    return (
      <div className="results">
        {artistResults.map((a) => (
          <button
            key={a.artist_name}
            className="result-row"
            onClick={() =>
              onSelect({
                type: "artist",
                artist_name: a.artist_name,
              })
            }
          >
            <div>
              <p className="track-title">{a.artist_name}</p>
              <p className="muted tiny">{a.track_count} tracks</p>
            </div>
            <span className="pill">Select</span>
          </button>
        ))}
      </div>
    );
  }, [artistResults, isLoading, onSelect, tab, trackResults]);

  if (!open) return null;

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-head">
          <div>
            <p className="eyebrow">Seed Picker</p>
            <h3>Tracks or Artists</h3>
          </div>
          <button className="ghost-btn small" onClick={onClose}>
            Close
          </button>
        </div>

        <div className="tabs">
          {(["tracks", "artists"] as const).map((t) => (
            <button
              key={t}
              className={tab === t ? "tab active" : "tab"}
              onClick={() => setTab(t)}
              type="button"
            >
              {t === "tracks" ? "Tracks" : "Artists"}
            </button>
          ))}
        </div>

        <input
          className="search"
          placeholder={tab === "tracks" ? "Search tracks, artists, albums" : "Search artists"}
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />

        <div className="results-container">{content}</div>
      </div>
    </div>
  );
}
