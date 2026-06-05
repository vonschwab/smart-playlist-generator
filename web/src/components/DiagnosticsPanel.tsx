import type { PlaylistOut } from "../lib/types";

function barColor(t: number): string {
  if (t >= 0.6) return "#5eead4";
  if (t >= 0.4) return "#f97316";
  return "#ef4444";
}

function fmt(n: number | null | undefined): string {
  return typeof n === "number" ? n.toFixed(2) : "—";
}

export function DiagnosticsPanel({ playlist }: { playlist: PlaylistOut | null }) {
  if (!playlist || playlist.tracks.length === 0) {
    return (
      <div className="p-4 text-xs text-[#3a3f4b]" data-testid="diagnostics-empty">
        Generate a playlist to see diagnostics.
      </div>
    );
  }

  const m = playlist.metrics;
  // Edges = every track that has a transition_score (last track is null).
  const edges = playlist.tracks
    .map((t, i) => ({ i, score: t.transition_score }))
    .filter((e): e is { i: number; score: number } => typeof e.score === "number");

  const weakest = edges.length
    ? edges.reduce((min, e) => (e.score < min.score ? e : min), edges[0])
    : null;

  const stat = (label: string, value: string) => (
    <div className="flex justify-between items-baseline py-1 border-b border-[#1a1c21]">
      <span className="text-[9px] uppercase tracking-[.06em] text-[#5b6470]">{label}</span>
      <span className="text-[11px] font-bold font-mono text-[#5eead4]">{value}</span>
    </div>
  );

  return (
    <div className="p-3 overflow-y-auto text-xs" data-testid="diagnostics-content">
      <div className="text-[9px] uppercase tracking-[.08em] text-[#3a3f4b] mb-1">Summary</div>
      <div className="grid grid-cols-2 gap-x-3">
        {stat("Mean", fmt(m?.mean_transition))}
        {stat("Min", fmt(m?.min_transition))}
        {stat("P10", fmt(m?.p10_transition))}
        {stat("P90", fmt(m?.p90_transition))}
        {stat("Artists", m?.distinct_artists != null ? String(m.distinct_artists) : "—")}
        {stat("Tracks", String(playlist.track_count))}
      </div>

      {weakest && (
        <div className="mt-3 bg-[#1a1015] border border-[#3a1a1a] rounded p-2" data-testid="weakest-edge">
          <div className="text-[8px] uppercase tracking-[.06em] text-[#ef4444] mb-0.5">⚠ Weakest edge</div>
          <div className="text-[10px] text-[#e6e9ec]">track {weakest.i + 1} → {weakest.i + 2}</div>
          <div className="text-[8px] font-mono text-[#ef4444]">T = {weakest.score.toFixed(2)}</div>
        </div>
      )}

      <div className="text-[8px] uppercase tracking-[.08em] text-[#3a3f4b] mt-3 mb-1">Transitions</div>
      <div className="flex flex-col gap-0.5" data-testid="transition-bars">
        {edges.map((e) => (
          <div key={e.i} className="flex items-center gap-1.5">
            <span className="text-[8px] text-[#3a3f4b] w-7 text-right shrink-0">{e.i + 1}→{e.i + 2}</span>
            <div className="flex-1 h-1.5 bg-[#1a1c21] rounded-sm overflow-hidden">
              <div className="h-full rounded-sm" style={{ width: `${e.score * 100}%`, background: barColor(e.score) }} />
            </div>
            <span className="text-[8px] font-mono text-[#5b6470] w-8 text-right shrink-0">{e.score.toFixed(2)}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
