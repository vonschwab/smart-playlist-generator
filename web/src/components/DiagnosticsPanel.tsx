import type { PlaylistOut } from "../lib/types";

function barColor(t: number): string {
  if (t >= 0.6) return "var(--color-accent)";
  if (t >= 0.4) return "var(--color-warn)";
  return "var(--color-danger)";
}

function fmt(n: number | null | undefined): string {
  return typeof n === "number" ? n.toFixed(2) : "—";
}

export function DiagnosticsPanel({ playlist }: { playlist: PlaylistOut | null }) {
  if (!playlist || playlist.tracks.length === 0) {
    return (
      <div className="p-4 text-xs text-faint" data-testid="diagnostics-empty">
        Generate a playlist to see diagnostics.
      </div>
    );
  }

  const m = playlist.metrics;
  // Edges = every track that has a transition_score (the FIRST track is null —
  // no incoming transition). track[i].transition_score is the edge INTO track i,
  // i.e. the 1-based transition "i → i+1".
  const edges = playlist.tracks
    .map((t, i) => ({ i, score: t.transition_score }))
    .filter((e): e is { i: number; score: number } => typeof e.score === "number");

  const weakest = edges.length
    ? edges.reduce((min, e) => (e.score < min.score ? e : min), edges[0])
    : null;

  const stat = (label: string, value: string) => (
    <div className="flex justify-between items-baseline py-1 border-b border-hairline">
      <span className="text-2xs uppercase tracking-[.06em] text-faint">{label}</span>
      <span className="text-xs font-bold font-mono text-accent">{value}</span>
    </div>
  );

  return (
    <div className="h-full overflow-y-auto p-3 text-xs" data-testid="diagnostics-content">
      <div className="text-2xs uppercase tracking-[.08em] text-faint mb-1">Summary</div>
      <div className="grid grid-cols-2 gap-x-3">
        {stat("Mean", fmt(m?.mean_transition))}
        {stat("Min", fmt(m?.min_transition))}
        {stat("P10", fmt(m?.p10_transition))}
        {stat("P90", fmt(m?.p90_transition))}
        {stat("Artists", m?.distinct_artists != null ? String(m.distinct_artists) : "—")}
        {stat("Tracks", String(playlist.track_count))}
      </div>

      {weakest && (
        <div className="mt-3 bg-danger/10 border border-danger/30 rounded p-2" data-testid="weakest-edge">
          <div className="text-2xs uppercase tracking-[.06em] text-danger mb-0.5">⚠ Weakest edge</div>
          <div className="text-2xs text-text">track {weakest.i} → {weakest.i + 1}</div>
          <div className="text-2xs font-mono text-danger">T = {weakest.score.toFixed(2)}</div>
        </div>
      )}

      <div className="text-2xs uppercase tracking-[.08em] text-faint mt-3 mb-1">Transitions</div>
      <div className="flex flex-col gap-0.5" data-testid="transition-bars">
        {edges.map((e) => (
          <div key={e.i} className="flex items-center gap-1.5">
            <span className="text-2xs text-faint w-7 text-right shrink-0">{e.i}→{e.i + 1}</span>
            <div className="flex-1 h-1.5 bg-hairline rounded-sm overflow-hidden">
              <div className="h-full rounded-sm" style={{ width: `${e.score * 100}%`, background: barColor(e.score) }} />
            </div>
            <span className="text-2xs font-mono text-faint w-8 text-right shrink-0">{e.score.toFixed(2)}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
