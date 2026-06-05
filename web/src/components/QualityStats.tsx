import type { MetricsOut } from "../lib/types";

const fmt = (n?: number | null) => (n == null ? "—" : n.toFixed(2));

export function QualityStats({ metrics, count }: { metrics?: MetricsOut; count: number }) {
  if (!metrics || count === 0) return null;

  const stat = (label: string, value: string) => (
    <div className="flex flex-col">
      <span className="text-[9px] uppercase tracking-wide text-faint">{label}</span>
      <span className="font-mono text-accent text-xs">{value}</span>
    </div>
  );

  return (
    <div className="flex gap-5 px-3 py-2 border-b border-border bg-panel2">
      {stat("tracks", String(count))}
      {stat("mean T", fmt(metrics.mean_transition))}
      {stat("min T", fmt(metrics.min_transition))}
      {metrics.p10_transition != null && stat("p10", fmt(metrics.p10_transition))}
      {metrics.p90_transition != null && stat("p90", fmt(metrics.p90_transition))}
      {stat("distinct artists", String(metrics.distinct_artists ?? "—"))}
    </div>
  );
}
