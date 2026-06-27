import { useState } from "react";
import type { PlaylistOut } from "../lib/types";
import { BlacklistPanel } from "./BlacklistPanel";
import { DiagnosticsPanel } from "./DiagnosticsPanel";
import { GenreReviewPanel } from "./GenreReviewPanel";
import { TaxonomyReviewPanel } from "./TaxonomyReviewPanel";

type Tab = "diagnostics" | "blacklist" | "review" | "taxonomy";

export function AdvancedPanel({ playlist }: { playlist: PlaylistOut | null }) {
  const [tab, setTab] = useState<Tab>("diagnostics");

  const tabBtn = (t: Tab, label: string) => (
    <button
      key={t}
      data-testid={`tab-${t}`}
      onClick={() => setTab(t)}
      className={`text-[11px] px-2.5 py-1.5 rounded-t ${tab === t ? "text-accent bg-bg" : "text-muted"}`}
    >
      {label}
    </button>
  );

  return (
    <div className="h-full flex flex-col">
      <div className="flex gap-1 px-2 pt-2 bg-panel2">
        {tabBtn("diagnostics", "Diagnostics")}
        {tabBtn("blacklist", "Blacklist")}
        {tabBtn("review", "Genre Review")}
        {tabBtn("taxonomy", "Taxonomy")}
      </div>
      <div className="flex-1 overflow-hidden">
        {tab === "diagnostics" && <DiagnosticsPanel playlist={playlist} />}
        {tab === "blacklist" && <BlacklistPanel />}
        {tab === "review" && <GenreReviewPanel />}
        {tab === "taxonomy" && <TaxonomyReviewPanel />}
      </div>
    </div>
  );
}
