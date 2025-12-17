import { useEffect, useMemo, useState } from "react";
import "./App.css";
import { Seed, SeedPickerModal } from "./components/SeedPickerModal";
import { SettingsPage } from "./SettingsPage";

type LibraryStatus = {
  total_tracks: number;
  sonic_analyzed_tracks: number;
  sonic_coverage_pct: number;
  genre_covered_tracks: number;
  genre_coverage_pct: number;
};

type PlaylistTrack = {
  track_id: string;
  title: string;
  artist: string;
  album?: string | null;
  duration_ms?: number | null;
  score?: number | null;
};

type ResolvedSeed = {
  primary_seed_track_id?: string | null;
  additional_seed_track_ids?: string[];
  source?: string;
};

type PlaylistResponse = {
  playlist_id: string;
  seed: { type: "track"; track_id: string } | { type: "artist"; artist_name: string };
  mode: string;
  length: number;
  random_seed: number | null;
  tracks: PlaylistTrack[];
  resolved_seed?: ResolvedSeed | null;
};

const API_BASE = "http://127.0.0.1:8000";

function App() {
  const [page, setPage] = useState<"main" | "settings">("main");
  const [status, setStatus] = useState<LibraryStatus | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [statusError, setStatusError] = useState<string | null>(null);

  const [seed, setSeed] = useState<Seed | null>(null);
  const [mode, setMode] = useState<"narrow" | "dynamic" | "discover">(
    "narrow",
  );
  const [length, setLength] = useState(12);
  const [randomSeed, setRandomSeed] = useState(0);
  const [playlist, setPlaylist] = useState<PlaylistResponse | null>(null);
  const [generateError, setGenerateError] = useState<string | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [isPickerOpen, setPickerOpen] = useState(false);
  const [defaultOutputDir, setDefaultOutputDir] = useState("");
  const [outputDir, setOutputDir] = useState("");
  const [fileName, setFileName] = useState("Auto - Playlist.m3u");
  const [exportMsg, setExportMsg] = useState<string | null>(null);
  const [exportError, setExportError] = useState<string | null>(null);
  const [isExporting, setIsExporting] = useState(false);

  useEffect(() => {
    const loadStatus = async () => {
      try {
        const res = await fetch(`${API_BASE}/api/library/status`);
        if (!res.ok) {
          throw new Error(`Request failed (${res.status})`);
        }
        const data = (await res.json()) as LibraryStatus;
        setStatus(data);
      } catch (err) {
        setStatusError(err instanceof Error ? err.message : "Unknown error");
      } finally {
        setIsLoading(false);
      }
    };

    loadStatus();
  }, []);

  useEffect(() => {
    const loadSettings = async () => {
      try {
        const res = await fetch(`${API_BASE}/api/settings`);
        if (!res.ok) return;
        const data = (await res.json()) as any;
        const gen = data.generation || {};
        const exp = data.export || {};
        setMode(gen.default_mode || "narrow");
        setLength(gen.default_length || 12);
        if (exp.default_output_dir) {
          setDefaultOutputDir(exp.default_output_dir);
          setOutputDir(exp.default_output_dir);
        }
      } catch {
        // ignore
      }
    };
    loadSettings();
  }, []);

  const playlistTitle = useMemo(() => {
    if (!playlist) return null;
    const first = playlist.tracks[0];
    if (!first) return "Generated playlist";
    return `${first.artist} — ${first.title}`;
  }, [playlist]);

  useEffect(() => {
    if (!playlist || !seed) return;
    const artistName =
      seed.type === "track"
        ? seed.artist || playlist.tracks[0]?.artist || "Library"
        : seed.artist_name;
    setFileName(`Auto - ${artistName}.m3u`);
  }, [playlist, seed]);

  const handleGenerate = async (seedValue?: number) => {
    if (!seed) {
      setGenerateError("Pick a seed to generate.");
      return;
    }
    setGenerateError(null);
    setIsGenerating(true);
    const seedToUse =
      seedValue !== undefined ? seedValue : randomSeed ?? Math.random();
    try {
      const res = await fetch(`${API_BASE}/api/playlist/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          seed:
            seed.type === "track"
              ? { type: "track", track_id: seed.track_id }
              : { type: "artist", artist_name: seed.artist_name },
          mode,
          length,
          random_seed: seedToUse,
        }),
      });
      if (!res.ok) {
        const text = await res.text();
        throw new Error(
          `Generate failed (${res.status})${text ? `: ${text}` : ""}`,
        );
      }
      const data = (await res.json()) as PlaylistResponse;
      setPlaylist(data);
      setRandomSeed(seedToUse);
    } catch (err) {
      setGenerateError(err instanceof Error ? err.message : "Unknown error");
      setPlaylist(null);
    } finally {
      setIsGenerating(false);
    }
  };

  const handleTryAgain = () => {
    const nextSeed = (randomSeed ?? 0) + 1;
    setRandomSeed(nextSeed);
    handleGenerate(nextSeed);
  };

  const ensureM3UExt = (name: string) =>
    name.toLowerCase().endsWith(".m3u") ? name : `${name}.m3u`;

  const handleExport = async () => {
    if (!playlist) return;
    setExportMsg(null);
    setExportError(null);
    setIsExporting(true);
    try {
      const res = await fetch(`${API_BASE}/api/playlist/export`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          track_ids: playlist.tracks.map((t) => t.track_id),
          output_dir: outputDir || defaultOutputDir,
          filename: ensureM3UExt(fileName.trim()),
        }),
      });
      const data = await res.json();
      if (!res.ok || !data.ok) {
        throw new Error(data.error || `Export failed (${res.status})`);
      }
      setExportMsg(`Saved to ${data.path}`);
    } catch (err) {
      setExportError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setIsExporting(false);
    }
  };

  const handleBrowse = async () => {
    setExportError(null);
    try {
      // @ts-expect-error optional API
      if (window.showDirectoryPicker) {
        // @ts-expect-error experimental
        const dirHandle = await window.showDirectoryPicker();
        // @ts-expect-error experimental
        const path = dirHandle.name;
        setOutputDir(path);
      } else {
        setExportError("Directory picker not supported in this browser.");
      }
    } catch (err) {
        setExportError(
          err instanceof Error ? err.message : "Unable to pick directory",
        );
    }
  };

  return (
    <div className="page">
      {page === "settings" ? (
        <SettingsPage
          apiBase={API_BASE}
          onBack={() => setPage("main")}
          onSaved={(s) => {
            if (s.export?.default_output_dir) {
              setDefaultOutputDir(s.export.default_output_dir);
              setOutputDir(s.export.default_output_dir);
            }
            if (s.generation?.default_mode) {
              setMode(s.generation.default_mode);
            }
            if (s.generation?.default_length) {
              setLength(s.generation.default_length);
            }
          }}
        />
      ) : (
        <>
      <header className="header">
        <div className="pill">MVP</div>
        <div>
          <p className="eyebrow">Playlist Generator</p>
          <h1>Track seed flow</h1>
          <p className="subhead">
            Generate a playlist from a track ID using the local FastAPI backend.
          </p>
        </div>
      </header>

      <section className="panel">
        <div className="panel-head">
          <div>
            <p className="eyebrow">Generate</p>
            <h2>Seed a playlist</h2>
          </div>
        </div>

        <div className="form-grid">
          <label className="field">
            <span>Seed</span>
            <div className="seed-display">
              {seed ? (
                seed.type === "track" ? (
                  <>
                    <strong>Track:</strong> {seed.title}{" "}
                    <span className="muted">— {seed.artist}</span>
                  </>
                ) : (
                  <>
                    <strong>Artist:</strong> {seed.artist_name}
                  </>
                )
              ) : (
                <span className="muted">Pick a track or artist</span>
              )}
            </div>
            {playlist?.resolved_seed && (
              <p className="muted tiny">
                Using primary:{" "}
                <code className="mono">
                  {playlist.resolved_seed.primary_seed_track_id}
                </code>{" "}
                · Additional:{" "}
                {playlist.resolved_seed.additional_seed_track_ids?.length
                  ? playlist.resolved_seed.additional_seed_track_ids.join(", ")
                  : "none"}{" "}
                · Source: {playlist.resolved_seed.source || "unknown"}
              </p>
            )}
          </label>
          <div className="field pick-btn">
            <span>&nbsp;</span>
            <button className="primary-btn" onClick={() => setPickerOpen(true)}>
              Pick seed
            </button>
          </div>

          <label className="field">
            <span>Length</span>
            <input
              type="number"
              min={1}
              max={200}
              value={length}
              onChange={(e) => setLength(Number(e.target.value))}
            />
          </label>

          <div className="field">
            <span>Mode</span>
            <div className="segmented">
              {(["narrow", "dynamic", "discover"] as const).map((m) => (
                <button
                  key={m}
                  className={mode === m ? "seg active" : "seg"}
                  onClick={() => setMode(m)}
                  type="button"
                >
                  {m}
                </button>
              ))}
            </div>
          </div>
        </div>

        <div className="actions">
          <button
            className="primary-btn"
            onClick={() => handleGenerate()}
            disabled={isGenerating || !seed}
          >
            {isGenerating ? "Generating…" : "Generate"}
          </button>
          <button
            className="ghost-btn"
            onClick={handleTryAgain}
            disabled={isGenerating || !playlist}
          >
            Try Again
          </button>
          <div className="muted seed">
            Random seed: <strong>{randomSeed}</strong>
          </div>
        </div>

        {generateError && <p className="error">{generateError}</p>}

        {playlist && (
          <div className="playlist">
            <div className="playlist-head">
              <div>
                <p className="eyebrow">Playlist</p>
                <h3>{playlistTitle}</h3>
                <p className="muted">
                  {playlist.tracks.length} tracks · mode {playlist.mode} · seed{" "}
                  {playlist.random_seed}
                </p>
              </div>
              <code className="pill mono">{playlist.playlist_id}</code>
            </div>

            <ol>
              {playlist.tracks.map((t, idx) => (
                <li key={`${t.track_id}-${idx}`} className="track-row">
                  <div>
                    <p className="track-title">
                      {t.title}{" "}
                      <span className="muted">— {t.artist || "Unknown"}</span>
                    </p>
                    <p className="muted tiny">
                      {t.album || "Unknown album"} · {t.track_id}
                    </p>
                  </div>
                  <div className="score">
                    {t.score !== undefined && t.score !== null
                      ? t.score.toFixed(3)
                      : "—"}
                  </div>
                </li>
              ))}
            </ol>
          </div>
        )}
      </section>

      <section className="panel">
        <div className="panel-head">
          <div>
            <p className="eyebrow">Export</p>
            <h2>Save M3U</h2>
          </div>
        </div>
        <div className="form-grid">
          <label className="field">
            <span>Folder</span>
            <input
              value={outputDir}
              onChange={(e) => setOutputDir(e.target.value)}
              placeholder="E:\\PLAYLISTS"
            />
            <div className="inline-actions">
              <button
                className="ghost-btn small"
                onClick={() => setOutputDir(defaultOutputDir)}
                type="button"
              >
                Use default
              </button>
              <button
                className="ghost-btn small"
                onClick={handleBrowse}
                type="button"
              >
                Browse
              </button>
            </div>
          </label>
          <label className="field">
            <span>Filename</span>
            <input
              value={fileName}
              onChange={(e) => setFileName(ensureM3UExt(e.target.value))}
              placeholder="Auto - Band.m3u"
            />
          </label>
        </div>
        <div className="actions">
          <button
            className="primary-btn"
            onClick={handleExport}
            disabled={!playlist || isExporting}
          >
            {isExporting ? "Exporting…" : "Export M3U"}
          </button>
          {exportMsg && <p className="muted">{exportMsg}</p>}
          {exportError && <p className="error">{exportError}</p>}
        </div>
      </section>

      <section className="panel">
        <div className="panel-head">
          <div>
            <p className="eyebrow">Status</p>
            <h2>Library coverage</h2>
          </div>
          <div className="dot-group">
            <span className="dot active" />
            <span className="dot" />
            <span className="dot" />
          </div>
        </div>

        {isLoading && <p className="muted">Loading library status…</p>}
        {statusError && (
          <p className="error">
            Unable to reach API: <span>{statusError}</span>
          </p>
        )}

        {status && !statusError && (
          <div className="grid">
            <div className="card primary">
              <p className="eyebrow">Total tracks</p>
              <p className="metric">{status.total_tracks.toLocaleString()}</p>
            </div>
            <div className="card">
              <p className="eyebrow">Sonic analyzed</p>
              <p className="metric">
                {status.sonic_analyzed_tracks.toLocaleString()}
              </p>
              <p className="muted">
                Coverage: {status.sonic_coverage_pct.toFixed(2)}%
              </p>
            </div>
            <div className="card">
              <p className="eyebrow">Genre coverage</p>
              <p className="metric">
                {status.genre_covered_tracks.toLocaleString()}
              </p>
              <p className="muted">
                Coverage: {status.genre_coverage_pct.toFixed(2)}%
              </p>
            </div>
          </div>
        )}
      </section>

      <SeedPickerModal
        open={isPickerOpen}
        onClose={() => setPickerOpen(false)}
        onSelect={(s) => {
          setSeed(s);
          setPickerOpen(false);
        }}
        apiBase={API_BASE}
      />
        </>
      )}
      <button className="ghost-btn floating" onClick={() => setPage(page === "main" ? "settings" : "main")}>
        {page === "main" ? "Advanced Settings" : "Back"}
      </button>
    </div>
  );
}

export default App;
