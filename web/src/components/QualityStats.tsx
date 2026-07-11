import { useState } from "react";
import type { MetricsOut, Receipt, TrackOut } from "../lib/types";

const fmt = (n?: number | null) => (n == null ? "—" : n.toFixed(2));

export interface QualityStatsProps {
  metrics?: MetricsOut;
  receipt?: Receipt | null;
  count: number;
  tracks: TrackOut[];
  onExportM3U8: () => void;
  onExportPlex: () => void;
}

function ReceiptLine({ receipt }: { receipt: Receipt | null | undefined }) {
  const [open, setOpen] = useState(false);
  if (!receipt) return null;
  const bits: string[] = [];
  if (receipt.range.pool != null) bits.push(`Range: ${receipt.range.pool} tracks in reach`);
  if (receipt.flow.worst != null)
    bits.push(`Flow: roughest seam ${receipt.flow.worst.toFixed(2)}` +
      (receipt.flow.mean != null ? ` (avg ${receipt.flow.mean.toFixed(2)})` : ""));
  if (receipt.pace.bpm_std != null) bits.push(`Pace: ±${Math.round(receipt.pace.bpm_std)} BPM`);
  if (!bits.length && !receipt.notes.length) return null;
  return (
    <div className="text-xs text-faint mt-1">
      <span>⚙ {bits.join(" · ")}</span>
      {receipt.notes.length > 0 && (
        <button type="button" className="ml-2 text-muted" onClick={() => setOpen((v) => !v)}>
          ⚠ {receipt.notes.length} note{receipt.notes.length > 1 ? "s" : ""} {open ? "▴" : "▾"}
        </button>
      )}
      {open && (
        <ul className="mt-1 list-disc pl-5">
          {receipt.notes.map((n, i) => <li key={i}>{n}</li>)}
        </ul>
      )}
    </div>
  );
}

export function QualityStats({ metrics, receipt, count, tracks, onExportM3U8, onExportPlex }: QualityStatsProps) {
  if (!metrics || count === 0) return null;
  const disabled = tracks.length === 0;

  const stat = (label: string, value: string) => (
    <div className="flex flex-col">
      <span className="text-2xs uppercase tracking-wide text-faint">{label}</span>
      <span className="font-mono text-accent text-xs">{value}</span>
    </div>
  );

  return (
    <div className="flex flex-col px-3 py-2 border-b border-border bg-panel2">
      <div className="flex flex-wrap items-center gap-x-5 gap-y-2">
        {stat("tracks", String(count))}
        {stat("mean T", fmt(metrics.mean_transition))}
        {stat("min T", fmt(metrics.min_transition))}
        {metrics.p10_transition != null && stat("p10", fmt(metrics.p10_transition))}
        {metrics.p90_transition != null && stat("p90", fmt(metrics.p90_transition))}
        {stat("distinct artists", String(metrics.distinct_artists ?? "—"))}
        <div className="ml-auto flex gap-2 shrink-0">
          <button
            data-testid="export-m3u8"
            onClick={onExportM3U8}
            disabled={disabled}
            className="border border-border text-muted hover:text-text text-xs px-2.5 py-1 rounded disabled:opacity-40"
          >
            ↓ M3U8
          </button>
          <button
            data-testid="export-plex"
            onClick={onExportPlex}
            disabled={disabled}
            className="border border-border text-muted hover:text-text text-xs px-2.5 py-1 rounded disabled:opacity-40"
          >
            → Plex
          </button>
        </div>
      </div>
      <ReceiptLine receipt={receipt} />
    </div>
  );
}
