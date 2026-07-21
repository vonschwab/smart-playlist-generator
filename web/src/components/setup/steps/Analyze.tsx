import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "../../../lib/api";
import { friendlyError } from "../../../lib/errors";
import { useJobReconcile } from "../../../lib/useJobReconcile";
import { useWorkerEvents } from "../../../lib/ws";
import type { WsEvent } from "../../../lib/types";

// Step 7/7 — the wizard's terminal step. Review already wrote config.yaml;
// this step fires the first library analysis the instant it mounts and hands
// off to a lightweight progress readout, reusing the same WS + reconcile
// pattern ToolsPanel uses for its long-running jobs (just without the stage
// picker — the wizard always requests the full default stage set). There's
// no in-wizard "finish": analysis is long-running (can be hours on a big
// library), so this is a handoff, not a wait. The app's own reload picks up
// `state: "ready"` once the job's `publish`/`artifacts` stages land.
export function Analyze() {
  const [jobId, setJobId] = useState<string | null>(null);
  const [progress, setProgress] = useState("");
  const [done, setDone] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [failed, setFailed] = useState<string | null>(null);
  const started = useRef(false);

  useEffect(() => {
    // Guard against a double-invoke (React StrictMode / a stray re-mount) —
    // this must fire the analyze job at most once per wizard visit.
    if (started.current) return;
    started.current = true;
    api
      .analyzeLibrary({ stages: [], force: false, dry_run: false })
      .then(({ job_id }) => setJobId(job_id))
      .catch((e) => setError(friendlyError(e)));
  }, []);

  // A WS/reconcile "terminal" signal only means the job stopped running — it
  // does NOT mean it succeeded (a "done" event fires for a failed job too).
  // Mirrors ToolsPanel's finishAnalyze: fetch the canonical job record and
  // branch on its `error` field, exactly like finishAnalyze does. Unlike
  // finishAnalyze, a fetch failure here has no prior "last known good" state
  // to fall back to, so it surfaces as a failure too rather than being
  // swallowed — swallowing it would strand the user on "Working…" forever.
  const finish = useCallback((jid: string) => {
    setProgress("");
    api
      .job(jid)
      .then((j) => {
        if (j.error) setFailed(j.error);
        else setDone(true);
      })
      .catch((e) => setFailed(friendlyError(e)));
  }, []);

  useWorkerEvents((e: WsEvent) => {
    if (!jobId || e.job_id !== jobId) return;
    if (e.type === "progress") setProgress((e["detail"] as string | undefined) ?? "");
    if (e.type === "done") finish(jobId);
  });

  // Backstop for the WS-done race (see useJobReconcile) — also catches the
  // case where this step's own subscription missed the event entirely.
  useJobReconcile(jobId, finish);

  return (
    <div data-testid="step-analyze" className="flex max-w-xl flex-col gap-3">
      <h2 className="text-lg font-semibold text-text">Analyze your library</h2>

      {error ? (
        <p role="alert" className="text-sm text-danger">
          Couldn't start analysis: {error}
        </p>
      ) : failed ? (
        <p role="alert" className="text-sm text-danger">
          Analysis failed: {failed}. Check the logs for details, or retry from Tools &gt;
          Analyze Library after reloading MixArc.
        </p>
      ) : done ? (
        <p className="text-sm text-text">
          Analysis finished. Reload MixArc to start generating playlists.
        </p>
      ) : (
        <>
          <p className="text-sm text-text">
            Analysis started — this can take a while. MixArc is scanning your library and
            building the similarity model it needs to generate playlists. You can leave this
            page open, or check back later from Tools.
          </p>
          <div className="flex items-center gap-2 rounded border border-border bg-panel p-3">
            <div className="h-2 w-2 shrink-0 animate-pulse rounded-full bg-accent" aria-hidden="true" />
            <span className="text-xs text-faint">{progress || "Working…"}</span>
          </div>
        </>
      )}
    </div>
  );
}
