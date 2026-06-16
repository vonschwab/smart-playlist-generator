import { useCallback, useEffect, useState } from "react";
import { Shell } from "./components/Shell";
import { AdvancedPanel } from "./components/AdvancedPanel";
import { GenerateControls } from "./components/GenerateControls";
import { SeedTrackSection } from "./components/SeedTrackSection";
import { TrackTable } from "./components/TrackTable";
import { QualityStats } from "./components/QualityStats";
import { LogPanel } from "./components/LogPanel";
import { JobsPanel } from "./components/JobsPanel";
import { ToolsPanel } from "./components/ToolsPanel";
import { PlayerProvider } from "./contexts/PlayerContext";
import { MiniPlayer } from "./components/MiniPlayer";
import { api } from "./lib/api";
import { useWorkerEvents } from "./lib/ws";
import { useLocalStorage } from "./lib/useLocalStorage";
import type { CandidateOut, GenerateRequestBody, JobOut, Mode, PlaylistOut, SeedTrack, TrackOut, WsEvent } from "./lib/types";
import { TrackContextMenu, type MenuTarget } from "./components/TrackContextMenu";
import { ReplaceDialog } from "./components/ReplaceDialog";
import { EditGenresDialog } from "./components/EditGenresDialog";
import { ExportPlexDialog } from "./components/ExportPlexDialog";
import { downloadM3U8 } from "./lib/m3u";

export default function App() {
  const [mode, setMode] = useLocalStorage<Mode>("pg_mode", "artist");
  const [seedTracks, setSeedTracks] = useLocalStorage<SeedTrack[]>("pg_seed_tracks", []);
  const addSeed = useCallback((t: SeedTrack) => {
    setSeedTracks(seedTracks.some((s) => s.track_id === t.track_id) ? seedTracks : [...seedTracks, t]);
  }, [seedTracks, setSeedTracks]);
  const removeSeed = useCallback((id: string) => {
    setSeedTracks(seedTracks.filter((t) => t.track_id !== id));
  }, [seedTracks, setSeedTracks]);
  const clearSeeds = useCallback(() => setSeedTracks([]), [setSeedTracks]);

  const [tab, setTab] = useState<"generate" | "tools">("generate");
  const [busy, setBusy] = useState(false);
  const [rerunValues, setRerunValues] = useState<GenerateRequestBody | null>(null);
  const [playlist, setPlaylist] = useState<PlaylistOut | null>(null);
  const [logs, setLogs] = useState<string[]>([]);
  const [jobs, setJobs] = useState<JobOut[]>(() => {
    try {
      const raw = localStorage.getItem("pg_jobs_history");
      if (!raw) return [];
      return (JSON.parse(raw) as Array<Omit<JobOut, "playlist">>).map((j) => ({ ...j, playlist: null }));
    } catch { return []; }
  });
  const [error, setError] = useState<string | null>(null);
  const [menuOpen, setMenuOpen] = useState(false);
  const [menuTarget, setMenuTarget] = useState<MenuTarget | null>(null);
  const [menuPos, setMenuPos] = useState({ x: 0, y: 0 });
  const [blacklisted, setBlacklisted] = useState<Set<string>>(new Set());
  const [jobId, setJobId] = useState<string>("");
  const [replaceOpen, setReplaceOpen] = useState(false);
  const [replacePos, setReplacePos] = useState(0);
  const [editGenresOpen, setEditGenresOpen] = useState(false);
  const [editTarget, setEditTarget] = useState<{ artist: string; album: string; genres: string[] }>({ artist: "", album: "", genres: [] });
  const [plexOpen, setPlexOpen] = useState(false);

  // Persist slim job history (no playlist payload) across server restarts.
  useEffect(() => {
    try {
      const slim = jobs.map(({ playlist: _, ...j }) => j);
      localStorage.setItem("pg_jobs_history", JSON.stringify(slim.slice(0, 30)));
    } catch {}
  }, [jobs]);

  // Merge server jobs into local history; server wins for the same job_id.
  const refreshJobs = useCallback(() => {
    api.jobs().then((serverJobs) => {
      setJobs((prev) => {
        const byId = new Map(prev.map((j) => [j.job_id, j]));
        for (const sj of serverJobs) byId.set(sj.job_id, sj);
        return [...byId.values()]
          .sort((a, b) => (b.created_at ?? 0) - (a.created_at ?? 0))
          .slice(0, 30);
      });
    }).catch(() => {});
  }, []);

  const handleCancel = useCallback(async (j: JobOut) => {
    try { await api.cancelJob(j.job_id); refreshJobs(); }
    catch (e) { setError(String(e)); }
  }, [refreshJobs]);

  const handleRerun = useCallback((params: GenerateRequestBody) => {
    setMode((params.mode as Mode) ?? "artist");
    setRerunValues({ ...params });
  }, [setMode]);

  const openMenu = useCallback((track: TrackOut, index: number, x: number, y: number) => {
    const last = (playlist?.tracks.length ?? 0) - 1;
    setMenuTarget({ track, index, isPier: index === 0 || index === last });
    setMenuPos({ x, y });
    setMenuOpen(true);
  }, [playlist]);

  const markBlacklisted = useCallback((ids: string[]) => {
    setBlacklisted((prev) => { const n = new Set(prev); ids.forEach((i) => n.add(i)); return n; });
  }, []);

  const handleBlacklistTrack = useCallback(async (t: MenuTarget) => {
    setMenuOpen(false);
    try {
      await api.blacklist({ track_ids: [t.track.rating_key ?? ""], enabled: true });
      markBlacklisted([t.track.rating_key ?? ""]);
    } catch (e) { setError(String(e)); }
  }, [markBlacklisted]);

  const handleBlacklistAlbum = useCallback(async (t: MenuTarget) => {
    setMenuOpen(false);
    try {
      await api.blacklist({ scope: "album", value: t.track.album, artist: t.track.artist, enabled: true });
    } catch (e) { setError(String(e)); }
  }, []);

  const handleBlacklistArtist = useCallback(async (t: MenuTarget) => {
    setMenuOpen(false);
    try {
      await api.blacklist({ scope: "artist", value: t.track.artist, enabled: true });
    } catch (e) { setError(String(e)); }
  }, []);

  const handleReplace = useCallback((t: MenuTarget) => {
    setMenuOpen(false);
    setReplacePos(t.index);
    setReplaceOpen(true);
  }, []);

  const applyReplacement = useCallback((position: number, cand: CandidateOut) => {
    setPlaylist((pl) => {
      if (!pl) return pl;
      const tracks = [...pl.tracks];
      const old = tracks[position];
      tracks[position] = {
        ...old,
        rating_key: cand.track_id,
        // file_path is the identity the Plex/M3U exporters resolve on — without it
        // the export keeps the OLD track's path and the replacement is lost.
        file_path: cand.file_path,
        duration_ms: cand.duration_ms,
        title: cand.title,
        artist: cand.artist,
        album: cand.album,
        genres: cand.genres,
        sonic_similarity: cand.fit_score,
      };
      return { ...pl, tracks };
    });
  }, []);

  const handleEditGenres = useCallback((t: MenuTarget) => {
    setMenuOpen(false);
    setEditTarget({ artist: t.track.artist, album: t.track.album, genres: t.track.genres });
    setEditGenresOpen(true);
  }, []);

  const applyGenreEdit = useCallback((artist: string, album: string, genres: string[]) => {
    setPlaylist((pl) => {
      if (!pl) return pl;
      const tracks = pl.tracks.map((tr) =>
        (tr.album === album && tr.artist === artist) ? { ...tr, genres } : tr
      );
      return { ...pl, tracks };
    });
  }, []);

  const defaultPlexName = useCallback(() => {
    const date = new Date().toISOString().slice(0, 10);
    const seed = playlist?.tracks[0]?.artist ?? "Playlist";
    return `${seed} — ${date}`;
  }, [playlist]);

  useWorkerEvents(useCallback((e: WsEvent) => {
    if (e.type === "log") setLogs((l) => [...l, `${e["level"] ?? "INFO"}: ${e["msg"] ?? ""}`].slice(-500));
    if (e.type === "error") setError(String(e["message"] ?? "error"));
    if (e.type === "done") {
      setBusy(false);
      const jid = e.job_id;
      if (jid) api.job(jid).then((j) => { if (j.playlist) setPlaylist(j.playlist); }).catch(() => {});
      refreshJobs();
    }
  }, [refreshJobs]));

  async function submit(body: GenerateRequestBody) {
    setError(null); setBusy(true); setLogs([]); setPlaylist(null);
    try {
      const { job_id } = await api.generate(body);
      setJobs((prev) => {
        const pending: JobOut = {
          job_id,
          status: "pending",
          stage: "queued",
          created_at: Date.now() / 1000,
          request_params: body as unknown as Record<string, unknown>,
          playlist: null,
        };
        const byId = new Map(prev.map((j) => [j.job_id, j]));
        byId.set(job_id, pending);
        return [...byId.values()]
          .sort((a, b) => (b.created_at ?? 0) - (a.created_at ?? 0))
          .slice(0, 30);
      });
      setJobId(job_id);
      refreshJobs();
    } catch (err) {
      setError(String(err)); setBusy(false);
    }
  }

  return (
    <PlayerProvider>
      <Shell
        topBar={
          <>
            <div className="font-bold text-sm"><span className="text-accent">◆</span> Playlist Generator</div>
            <div className="flex items-center gap-1 ml-4">
              {(["generate", "tools"] as const).map((t) => (
                <button
                  key={t}
                  onClick={() => setTab(t)}
                  className={[
                    "px-3 py-1 text-[11px] rounded transition-colors capitalize",
                    tab === t
                      ? "bg-[#23262d] text-[#c9d1d9]"
                      : "text-[#5b6470] hover:text-[#8b939d]",
                  ].join(" ")}
                >
                  {t}
                </button>
              ))}
            </div>
            {error && <div className="text-danger text-xs ml-auto">{error}</div>}
          </>
        }
        jobs={<JobsPanel jobs={jobs} onSelect={(j) => setPlaylist(j.playlist ?? null)} onCancel={handleCancel} onRerun={handleRerun} />}
        center={
          tab === "tools" ? (
            <ToolsPanel externalBusy={busy} refreshJobs={refreshJobs} />
          ) : (
            <div className="h-full flex flex-col overflow-hidden">
              <GenerateControls
                mode={mode}
                onModeChange={setMode}
                seedTrackIds={seedTracks.map((t) => t.track_id)}
                seedDisplays={seedTracks.map((t) => `${t.title} - ${t.artist}`)}
                onSubmit={submit}
                busy={busy}
                initialValues={rerunValues ?? undefined}
              />
              {mode === "seeds" && (
                <SeedTrackSection
                  tracks={seedTracks}
                  onAdd={addSeed}
                  onRemove={removeSeed}
                  onClear={clearSeeds}
                />
              )}
              <QualityStats
                metrics={playlist?.metrics}
                count={playlist?.track_count ?? 0}
                tracks={playlist?.tracks ?? []}
                onExportM3U8={() => playlist && downloadM3U8(playlist.tracks)}
                onExportPlex={() => setPlexOpen(true)}
              />
              <div className="flex-1 overflow-auto">
                <TrackTable
                  tracks={playlist?.tracks ?? []}
                  blacklisted={blacklisted}
                  onContextAction={openMenu}
                />
              </div>
            </div>
          )
        }
        right={<AdvancedPanel playlist={playlist} />}
        logs={<LogPanel lines={logs} />}
      />
      <MiniPlayer />
      <TrackContextMenu
        open={menuOpen}
        onOpenChange={setMenuOpen}
        target={menuTarget}
        pos={menuPos}
        onReplace={handleReplace}
        onBlacklistTrack={handleBlacklistTrack}
        onBlacklistAlbum={handleBlacklistAlbum}
        onBlacklistArtist={handleBlacklistArtist}
        onEditGenres={handleEditGenres}
      />
      <ReplaceDialog
        open={replaceOpen}
        onOpenChange={setReplaceOpen}
        jobId={jobId}
        position={replacePos}
        prevTitle={playlist?.tracks[replacePos - 1]?.title ?? ""}
        nextTitle={playlist?.tracks[replacePos + 1]?.title ?? ""}
        onConfirm={applyReplacement}
      />
      <EditGenresDialog
        open={editGenresOpen}
        onOpenChange={setEditGenresOpen}
        artist={editTarget.artist}
        album={editTarget.album}
        initialGenres={editTarget.genres}
        onSaved={applyGenreEdit}
      />
      <ExportPlexDialog
        open={plexOpen}
        onOpenChange={setPlexOpen}
        tracks={playlist?.tracks ?? []}
        defaultName={defaultPlexName()}
      />
    </PlayerProvider>
  );
}
