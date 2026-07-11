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
  onClear,
}: {
  jobs: JobOut[];
  onSelect: (j: JobOut) => void;
  onCancel: (j: JobOut) => void;
  onRerun: (params: GenerateRequestBody) => void;
  onClear: () => void;
}) {
  const clearable = jobs.some((j) => j.status !== "running" && j.status !== "pending");
  return (
    <div className="h-full overflow-auto" data-testid="jobs-panel">
      <div className="flex items-center justify-between px-3 py-2 border-b border-border">
        <span className="text-2xs uppercase tracking-wide text-faint">Jobs</span>
        {clearable && (
          <button
            data-testid="jobs-clear"
            onClick={onClear}
            title="Clear finished jobs"
            className="text-2xs uppercase tracking-wide text-faint hover:text-danger"
          >
            Clear
          </button>
        )}
      </div>
      {jobs.length === 0 && (
        <div className="px-4 py-8 text-center text-faint text-xs">
          No jobs yet — playlists you generate appear here.
        </div>
      )}
      {jobs.map((j) => {
        const meanT = (j.playlist?.metrics?.mean_transition);
        const tracks = j.playlist?.track_count;
        return (
          <div key={j.job_id} className="px-3 py-2 border-b border-hairline">
            <div className="flex items-center justify-between gap-2">
              <div className="text-xs text-text truncate">{j.playlist?.name ?? j.stage ?? "Playlist"}</div>
              <span className={`text-2xs ${dot[j.status] ?? "text-muted"}`}>{j.status}</span>
            </div>
            <div className="text-2xs text-faint mt-0.5">
              {tracks != null ? `${tracks} tracks` : "—"}
              {typeof meanT === "number" ? ` · T̄ ${meanT.toFixed(2)}` : ""}
              {clock(j.created_at) ? ` · ${clock(j.created_at)}` : ""}
            </div>
            <div className="flex gap-1.5 mt-1.5">
              {j.status === "running" && (
                <button data-testid="job-cancel" onClick={() => onCancel(j)}
                  className="text-2xs px-1.5 py-0.5 rounded border border-danger/30 text-danger">✕ cancel</button>
              )}
              {j.status !== "running" && j.request_params && (
                <button data-testid="job-rerun" onClick={() => onRerun(j.request_params as unknown as GenerateRequestBody)}
                  className="text-2xs px-1.5 py-0.5 rounded border border-accent/30 text-accent">↺ re-run</button>
              )}
              {j.playlist && (
                <button onClick={() => onSelect(j)}
                  className="text-2xs px-1.5 py-0.5 rounded border border-border text-muted">restore</button>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}
