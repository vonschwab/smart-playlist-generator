import type { JobOut } from "../lib/types";

const dot: Record<string, string> = {
  success: "text-accent", running: "text-warn", failed: "text-danger", cancelled: "text-muted", pending: "text-muted",
};

export function JobsPanel({ jobs, onSelect }: { jobs: JobOut[]; onSelect: (j: JobOut) => void }) {
  return (
    <div className="h-full overflow-auto">
      <div className="px-3 py-2 text-[10px] uppercase tracking-wide text-faint border-b border-border">Jobs</div>
      {jobs.map((j) => (
        <button key={j.job_id} onClick={() => onSelect(j)}
          className="w-full text-left px-3 py-2 border-b border-[#181b21] hover:bg-border">
          <div className="text-[11px] text-text truncate">{j.playlist?.name ?? j.stage ?? "Playlist"}</div>
          <div className={`text-[9px] ${dot[j.status] ?? "text-muted"}`}>{j.status}</div>
        </button>
      ))}
    </div>
  );
}
