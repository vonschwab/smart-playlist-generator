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
import type { GenerateRequestBody, JobOut, PlaylistOut, WsEvent } from "./lib/types";

export default function App() {
  const [busy, setBusy] = useState(false);
  const [playlist, setPlaylist] = useState<PlaylistOut | null>(null);
  const [logs, setLogs] = useState<string[]>([]);
  const [jobs, setJobs] = useState<JobOut[]>([]);
  const [error, setError] = useState<string | null>(null);

  const refreshJobs = useCallback(() => { api.jobs().then(setJobs).catch(() => {}); }, []);

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
      void job_id; // job_id tracked via WS events
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
              <TrackTable tracks={playlist?.tracks ?? []} />
            </div>
          </div>
        }
        right={<AdvancedPanel />}
        logs={<LogPanel lines={logs} />}
      />
      <MiniPlayer />
    </PlayerProvider>
  );
}
