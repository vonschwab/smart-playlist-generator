import { useEffect, useRef } from "react";
import { api } from "./api";

const TERMINAL = new Set(["success", "failed", "cancelled"]);

/**
 * Backstop for the WS-done race.
 *
 * Panels track a running job by storing its `job_id` in state after the POST
 * that starts it resolves, then clearing it when the matching WS `done` event
 * arrives. But a fast job can emit `done` BEFORE React has stored the id, so the
 * WS handler's `e.job_id === jobId` guard drops the event and the panel waits
 * forever (the scan button never re-enables).
 *
 * While `jobId` is set, this polls the authoritative job registry and fires
 * `onTerminal` once the job reaches a terminal status — or can't be found (e.g.
 * the server restarted). The WS path stays the low-latency norm; this guarantees
 * recovery when it's missed. `onTerminal` must be idempotent (clearing `jobId`
 * stops the poll) since the WS handler may also fire.
 */
export function useJobReconcile(
  jobId: string | null,
  onTerminal: (jobId: string) => void,
  intervalMs = 700,
) {
  const cb = useRef(onTerminal);
  cb.current = onTerminal;
  useEffect(() => {
    if (!jobId) return;
    let cancelled = false;
    const tick = async () => {
      try {
        const job = await api.job(jobId);
        if (!cancelled && TERMINAL.has(job.status)) cb.current(jobId);
      } catch {
        // Job not found (server restarted) — stop waiting so the UI recovers.
        if (!cancelled) cb.current(jobId);
      }
    };
    const id = setInterval(tick, intervalMs);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [jobId, intervalMs]);
}
