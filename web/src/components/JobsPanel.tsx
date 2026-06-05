import type { GenerateRequestBody, JobOut } from "../lib/types";

const dot: Record<string, string> = {
  success: "text-accent", running: "text-warn", failed: "text-danger",
  cancelled: "text-muted", pending: "text-muted",
};

function clock(ts?: number | null): string {
  if (!ts) return "";
  const d = new Date(ts * 1000);
  return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

export function JobsPanel({
  jobs,
  onSelect,
  onCancel,
  onRerun,
}: {
  jobs: JobOut[];
  onSelect: (j: JobOut) => void;
  onCancel: (j: JobOut) => void;
  onRerun: (params: GenerateRequestBody) => void;
}) {
  return (
    <div className="h-full overflow-auto" data-testid="jobs-panel">
      <div className="px-3 py-2 text-[10px] uppercase tracking-wide text-faint border-b border-border">Jobs</div>
      {jobs.map((j) => {
        const meanT = (j.playlist?.metrics?.mean_transition);
        const tracks = j.playlist?.track_count;
        return (
          <div key={j.job_id} className="px-3 py-2 border-b border-[#181b21]">
            <div className="flex items-center justify-between gap-2">
              <div className="text-[11px] text-text truncate">{j.playlist?.name ?? j.stage ?? "Playlist"}</div>
              <span className={`text-[9px] ${dot[j.status] ?? "text-muted"}`}>{j.status}</span>
            </div>
            <div className="text-[9px] text-faint mt-0.5">
              {tracks != null ? `${tracks} tracks` : "—"}
              {typeof meanT === "number" ? ` · T̄ ${meanT.toFixed(2)}` : ""}
              {clock(j.created_at) ? ` · ${clock(j.created_at)}` : ""}
            </div>
            <div className="flex gap-1.5 mt-1.5">
              {j.status === "running" && (
                <button data-testid="job-cancel" onClick={() => onCancel(j)}
                  className="text-[9px] px-1.5 py-0.5 rounded border border-[#3a1a1a] text-danger">✕ cancel</button>
              )}
              {j.status !== "running" && j.request_params && (
                <button data-testid="job-rerun" onClick={() => onRerun(j.request_params as unknown as GenerateRequestBody)}
                  className="text-[9px] px-1.5 py-0.5 rounded border border-[#1d3a35] text-accent">↺ re-run</button>
              )}
              {j.playlist && (
                <button onClick={() => onSelect(j)}
                  className="text-[9px] px-1.5 py-0.5 rounded border border-border text-muted">restore</button>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}
