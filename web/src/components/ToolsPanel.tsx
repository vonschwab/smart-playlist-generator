import { useCallback, useState } from "react";
import { api } from "../lib/api";
import { useJobReconcile } from "../lib/useJobReconcile";
import { useWorkerEvents } from "../lib/ws";
import type { AnalyzeToolRequest, EnrichToolRequest, WsEvent } from "../lib/types";

// Mirror of ANALYZE_LIBRARY_STAGE_ORDER (src/playlist/request_models.py) — keep in sync.
// The album-grain adjudicate/apply/popularity stages are in the default run; the legacy
// tag-grain `enrich` stage is opt-in CLI-only (`--stages enrich`), never the default.
const ALL_STAGES = [
  "scan", "genres", "discogs", "lastfm", "sonic", "muq",
  "adjudicate", "apply", "publish", "genre-sim", "artifacts", "energy",
  "popularity", "genre-embedding", "verify",
] as const;

type AnalyzeStageName = (typeof ALL_STAGES)[number];

interface StageResult {
  name: string;
  decision: string;
  duration_ms: number;
  errors: number;
}

function Card({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="flex flex-col gap-3 p-4 bg-[#0f1117] border border-[#23262d] rounded-md min-w-0">
      <div className="text-[11px] uppercase tracking-[.1em] text-[#5b6470] font-semibold">
        {title}
      </div>
      {children}
    </div>
  );
}

function Row({ children }: { children: React.ReactNode }) {
  return <div className="flex items-center gap-2">{children}</div>;
}

function Lbl({ children }: { children: React.ReactNode }) {
  return (
    <span className="text-[10px] uppercase tracking-[.08em] text-[#5b6470] font-medium select-none">
      {children}
    </span>
  );
}

function RunBtn({
  disabled,
  onClick,
  children,
}: {
  disabled?: boolean;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className="px-3 py-1 text-[11px] font-medium rounded bg-accent text-black
                 disabled:opacity-30 disabled:cursor-default hover:opacity-90 transition-opacity"
    >
      {children}
    </button>
  );
}

function CancelBtn({ onClick }: { onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className="px-3 py-1 text-[11px] font-medium rounded border border-[#3a3f4b]
                 text-[#8b939d] hover:text-[#c9d1d9] transition-colors"
    >
      Cancel
    </button>
  );
}

function ProgressBar({ label }: { label: string }) {
  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 h-1 bg-[#1e2128] rounded overflow-hidden">
        <div className="h-full bg-accent animate-pulse w-full" />
      </div>
      <span className="text-[10px] text-[#5b6470] truncate max-w-[160px]">{label}</span>
    </div>
  );
}

function StageTable({ stages }: { stages: StageResult[] }) {
  return (
    <table className="w-full text-[10px] text-[#5b6470]">
      <tbody>
        {stages.map((s) => (
          <tr key={s.name}>
            <td className="pr-2 text-[#8b939d] font-mono">{s.name}</td>
            <td className={`pr-2 ${s.decision === "ran" ? "text-accent" : "text-[#5b6470]"}`}>
              {s.decision}
            </td>
            <td className="pr-2">{s.duration_ms}ms</td>
            {s.errors > 0 && <td className="text-danger">{s.errors} err</td>}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function ToolsPanel({
  externalBusy,
  refreshJobs,
}: {
  externalBusy: boolean;
  refreshJobs: () => void;
}) {
  // ── Analyze Library state ───────────────────────────────────────────────
  const [selectedStages, setSelectedStages] = useState<AnalyzeStageName[]>(
    [...ALL_STAGES]
  );
  const [force, setForce] = useState(false);
  const [dryRun, setDryRun] = useState(false);
  const [analyzeJobId, setAnalyzeJobId] = useState<string | null>(null);
  const [analyzeProgress, setAnalyzeProgress] = useState("");
  const [lastAnalyzeStages, setLastAnalyzeStages] = useState<StageResult[] | null>(null);
  const [analyzeError, setAnalyzeError] = useState<string | null>(null);

  // ── Enrich Genres state ─────────────────────────────────────────────────
  const [scope, setScope] = useState<EnrichToolRequest["scope"]>("all_unenriched");
  const [enrichArtist, setEnrichArtist] = useState("");
  const [enrichAlbum, setEnrichAlbum] = useState("");
  const [enrichJobId, setEnrichJobId] = useState<string | null>(null);
  const [enrichProgress, setEnrichProgress] = useState("");
  const [lastEnrichSummary, setLastEnrichSummary] = useState<string | null>(null);
  const [enrichError, setEnrichError] = useState<string | null>(null);

  const anyRunning = analyzeJobId !== null || enrichJobId !== null;
  const runDisabled = externalBusy || anyRunning;

  // Completion handlers, shared by the WS `done` path and the reconcile backstop
  // (useJobReconcile) so a fast job whose `done` raced ahead of setAnalyzeJobId/
  // setEnrichJobId still clears the running state. Idempotent: clearing the job
  // id stops the poll and makes the WS guard a no-op if both fire.
  const finishAnalyze = useCallback(
    (jid: string) => {
      setAnalyzeJobId(null);
      setAnalyzeProgress("");
      api
        .job(jid)
        .then((j) => {
          const tr = j.tool_result;
          if (tr && Array.isArray(tr["stages"]))
            setLastAnalyzeStages(tr["stages"] as StageResult[]);
          if (j.error) setAnalyzeError(j.error ?? null);
        })
        .catch(() => {});
      refreshJobs();
    },
    [refreshJobs]
  );

  const finishEnrich = useCallback(
    (jid: string) => {
      setEnrichJobId(null);
      setEnrichProgress("");
      api
        .job(jid)
        .then((j) => {
          const tr = j.tool_result;
          if (tr)
            setLastEnrichSummary(
              `${tr["releases"] ?? 0} releases enriched, ${tr["genres_applied"] ?? 0} genres applied`
            );
          if (j.error) setEnrichError(j.error ?? null);
        })
        .catch(() => {});
      refreshJobs();
    },
    [refreshJobs]
  );

  useWorkerEvents(
    useCallback(
      (e: WsEvent) => {
        if (e.type === "progress") {
          const detail = e["detail"] as string | undefined;
          if (e.job_id === analyzeJobId) setAnalyzeProgress(detail ?? "");
          if (e.job_id === enrichJobId) setEnrichProgress(detail ?? "");
        }
        if (e.type === "done") {
          if (e.job_id === analyzeJobId) finishAnalyze(e.job_id as string);
          if (e.job_id === enrichJobId) finishEnrich(e.job_id as string);
        }
      },
      [analyzeJobId, enrichJobId, finishAnalyze, finishEnrich]
    )
  );

  useJobReconcile(analyzeJobId, finishAnalyze);
  useJobReconcile(enrichJobId, finishEnrich);

  function toggleStage(s: AnalyzeStageName) {
    setSelectedStages((prev) =>
      prev.includes(s)
        ? prev.filter((x) => x !== s)
        : [...ALL_STAGES.filter((a) => prev.includes(a) || a === s)]
    );
  }

  async function runAnalyze() {
    setAnalyzeError(null);
    const req: AnalyzeToolRequest = {
      stages: selectedStages.length < ALL_STAGES.length ? [...selectedStages] : [],
      force,
      dry_run: dryRun,
    };
    try {
      const { job_id } = await api.analyzeLibrary(req);
      setAnalyzeJobId(job_id);
      refreshJobs();
    } catch (e) {
      setAnalyzeError(String(e));
    }
  }

  async function runEnrich() {
    setEnrichError(null);
    const req: EnrichToolRequest = {
      scope,
      artist: enrichArtist.trim() || undefined,
      album: enrichAlbum.trim() || undefined,
    };
    try {
      const { job_id } = await api.enrich(req);
      setEnrichJobId(job_id);
      refreshJobs();
    } catch (e) {
      setEnrichError(String(e));
    }
  }

  async function cancelAnalyze() {
    if (!analyzeJobId) return;
    try { await api.cancelJob(analyzeJobId); } catch { /* ignore */ }
  }

  async function cancelEnrich() {
    if (!enrichJobId) return;
    try { await api.cancelJob(enrichJobId); } catch { /* ignore */ }
  }

  return (
    <div className="h-full overflow-auto p-4">
      <div className="grid grid-cols-2 gap-4 max-w-4xl">
        {/* ── Analyze Library ───────────────────────────────────────────── */}
        <Card title="Analyze Library">
          {/* Stage checkboxes */}
          <div className="flex flex-wrap gap-x-3 gap-y-1">
            {ALL_STAGES.map((s) => (
              <label key={s} className="flex items-center gap-1 cursor-pointer">
                <input
                  type="checkbox"
                  checked={selectedStages.includes(s)}
                  onChange={() => toggleStage(s)}
                  className="accent-accent"
                />
                <span className="text-[10px] text-[#8b939d] font-mono">{s}</span>
              </label>
            ))}
          </div>

          {/* Options row */}
          <Row>
            <label className="flex items-center gap-1 cursor-pointer">
              <input
                type="checkbox"
                checked={force}
                onChange={(e) => setForce(e.target.checked)}
                className="accent-accent"
              />
              <Lbl>Force</Lbl>
            </label>
            <label className="flex items-center gap-1 cursor-pointer">
              <input
                type="checkbox"
                checked={dryRun}
                onChange={(e) => setDryRun(e.target.checked)}
                className="accent-accent"
              />
              <Lbl>Dry run</Lbl>
            </label>
          </Row>

          {/* Run / Cancel */}
          <Row>
            {analyzeJobId ? (
              <CancelBtn onClick={cancelAnalyze} />
            ) : (
              <RunBtn disabled={runDisabled} onClick={runAnalyze}>
                Run
              </RunBtn>
            )}
          </Row>

          {/* Progress */}
          {analyzeJobId && analyzeProgress && (
            <ProgressBar label={analyzeProgress} />
          )}

          {/* Error */}
          {analyzeError && (
            <div className="text-[10px] text-danger">{analyzeError}</div>
          )}

          {/* Last-run summary */}
          {lastAnalyzeStages && !analyzeJobId && (
            <details className="text-[10px] text-[#5b6470]">
              <summary className="cursor-pointer select-none">
                Last run — {lastAnalyzeStages.length} stages
              </summary>
              <div className="mt-1">
                <StageTable stages={lastAnalyzeStages} />
              </div>
            </details>
          )}
        </Card>

        {/* ── Enrich Genres ─────────────────────────────────────────────── */}
        <Card title="Enrich Genres">
          {/* Scope select */}
          <Row>
            <Lbl>Scope</Lbl>
            <select
              value={scope}
              onChange={(e) =>
                setScope(e.target.value as EnrichToolRequest["scope"])
              }
              className="bg-[#0c0e12] border border-[#23262d] rounded text-[11px]
                         text-[#8b939d] py-[3px] px-2 cursor-pointer"
            >
              <option value="all_unenriched">All unenriched</option>
              <option value="artist">Artist</option>
              <option value="release">Release</option>
            </select>
          </Row>

          {/* Artist field */}
          {(scope === "artist" || scope === "release") && (
            <Row>
              <Lbl>Artist</Lbl>
              <input
                type="text"
                value={enrichArtist}
                onChange={(e) => setEnrichArtist(e.target.value)}
                placeholder="artist name"
                className="flex-1 bg-[#0c0e12] border border-[#23262d] rounded
                           text-[11px] text-[#c9d1d9] py-[3px] px-2
                           placeholder:text-[#3a3f4b] outline-none
                           focus:border-[#3a3f4b]"
              />
            </Row>
          )}

          {/* Album field */}
          {scope === "release" && (
            <Row>
              <Lbl>Album</Lbl>
              <input
                type="text"
                value={enrichAlbum}
                onChange={(e) => setEnrichAlbum(e.target.value)}
                placeholder="album title (optional)"
                className="flex-1 bg-[#0c0e12] border border-[#23262d] rounded
                           text-[11px] text-[#c9d1d9] py-[3px] px-2
                           placeholder:text-[#3a3f4b] outline-none
                           focus:border-[#3a3f4b]"
              />
            </Row>
          )}

          {/* Run / Cancel */}
          <Row>
            {enrichJobId ? (
              <CancelBtn onClick={cancelEnrich} />
            ) : (
              <RunBtn disabled={runDisabled} onClick={runEnrich}>
                {scope === "all_unenriched" ? "Enrich all pending" : "Enrich"}
              </RunBtn>
            )}
          </Row>

          {/* Progress */}
          {enrichJobId && enrichProgress && (
            <ProgressBar label={enrichProgress} />
          )}

          {/* Error */}
          {enrichError && (
            <div className="text-[10px] text-danger">{enrichError}</div>
          )}

          {/* Last-run summary */}
          {lastEnrichSummary && !enrichJobId && (
            <div className="text-[10px] text-[#5b6470]">{lastEnrichSummary}</div>
          )}
        </Card>
      </div>
    </div>
  );
}
