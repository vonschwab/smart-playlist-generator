import type { MetricsOut, TrackOut } from "../lib/types";

const fmt = (n?: number | null) => (n == null ? "—" : n.toFixed(2));

export interface QualityStatsProps {
  metrics?: MetricsOut;
  count: number;
  tracks: TrackOut[];
  onExportM3U8: () => void;
  onExportPlex: () => void;
}

export function QualityStats({ metrics, count, tracks, onExportM3U8, onExportPlex }: QualityStatsProps) {
  if (!metrics || count === 0) return null;
  const disabled = tracks.length === 0;

  const stat = (label: string, value: string) => (
    <div className="flex flex-col">
      <span className="text-[9px] uppercase tracking-wide text-faint">{label}</span>
      <span className="font-mono text-accent text-xs">{value}</span>
    </div>
  );

  return (
    <div className="flex items-center gap-5 px-3 py-2 border-b border-border bg-panel2">
      {stat("tracks", String(count))}
      {stat("mean T", fmt(metrics.mean_transition))}
      {stat("min T", fmt(metrics.min_transition))}
      {metrics.p10_transition != null && stat("p10", fmt(metrics.p10_transition))}
      {metrics.p90_transition != null && stat("p90", fmt(metrics.p90_transition))}
      {stat("distinct artists", String(metrics.distinct_artists ?? "—"))}
      <div className="ml-auto flex gap-2">
        <button
          data-testid="export-m3u8"
          onClick={onExportM3U8}
          disabled={disabled}
          className="border border-border text-muted hover:text-text text-[11px] px-2.5 py-1 rounded disabled:opacity-40"
        >
          ↓ M3U8
        </button>
        <button
          data-testid="export-plex"
          onClick={onExportPlex}
          disabled={disabled}
          className="border border-border text-muted hover:text-text text-[11px] px-2.5 py-1 rounded disabled:opacity-40"
        >
          → Plex
        </button>
      </div>
    </div>
  );
}
