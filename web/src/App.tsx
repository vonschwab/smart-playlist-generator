import { useCallback, useState } from "react";
import { Shell } from "./components/Shell";
import { AdvancedPanel } from "./components/AdvancedPanel";
import { GenerateControls } from "./components/GenerateControls";
import { TrackTable } from "./components/TrackTable";
import { QualityStats } from "./components/QualityStats";
import { LogPanel } from "./components/LogPanel";
import { JobsPanel } from "./components/JobsPanel";
import { PlayerProvider } from "./contexts/PlayerContext";
import { MiniPlayer } from "./components/MiniPlayer";
import { api } from "./lib/api";
import { useWorkerEvents } from "./lib/ws";
import type { CandidateOut, GenerateRequestBody, JobOut, PlaylistOut, TrackOut, WsEvent } from "./lib/types";
import { TrackContextMenu, type MenuTarget } from "./components/TrackContextMenu";
import { ReplaceDialog } from "./components/ReplaceDialog";
import { EditGenresDialog } from "./components/EditGenresDialog";

export default function App() {
  const [busy, setBusy] = useState(false);
  const [playlist, setPlaylist] = useState<PlaylistOut | null>(null);
  const [logs, setLogs] = useState<string[]>([]);
  const [jobs, setJobs] = useState<JobOut[]>([]);
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

  const refreshJobs = useCallback(() => { api.jobs().then(setJobs).catch(() => {}); }, []);

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

  const applyGenreEdit = useCallback((album: string, genres: string[]) => {
    setPlaylist((pl) => {
      if (!pl) return pl;
      const tracks = pl.tracks.map((tr) => (tr.album === album ? { ...tr, genres } : tr));
      return { ...pl, tracks };
    });
  }, []);

  useWorkerEvents(useCallback((e: WsEvent) => {
    if (e.type === "log") setLogs((l) => [...l, `${(e as any).level ?? "INFO"}: ${(e as any).msg ?? ""}`].slice(-500));
    if (e.type === "error") setError(String((e as any).message ?? "error"));
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
            {error && <div className="text-danger text-xs">{error}</div>}
          </>
        }
        jobs={<JobsPanel jobs={jobs} onSelect={(j) => setPlaylist(j.playlist ?? null)} />}
        center={
          <div className="h-full flex flex-col overflow-hidden">
            <GenerateControls onSubmit={submit} busy={busy} />
            <QualityStats metrics={playlist?.metrics} count={playlist?.track_count ?? 0} />
            <div className="flex-1 overflow-auto">
              <TrackTable
                tracks={playlist?.tracks ?? []}
                blacklisted={blacklisted}
                onContextAction={openMenu}
              />
            </div>
          </div>
        }
        right={<AdvancedPanel />}
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
    </PlayerProvider>
  );
}
