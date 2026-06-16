import { useState } from "react";
import type { RelaxationEntry } from "../lib/types";

interface RelaxationNoticeProps {
  relaxations: RelaxationEntry[];
}

export function RelaxationNotice({ relaxations }: RelaxationNoticeProps) {
  const [dismissed, setDismissed] = useState(false);

  if (!relaxations || relaxations.length === 0 || dismissed) return null;

  const hasInvariant = relaxations.some((r) => r.severity === "invariant");

  return (
    <div
      className={`flex items-start gap-2 px-3 py-2 border-b border-border text-xs ${
        hasInvariant
          ? "bg-yellow-900/20 text-yellow-300"
          : "bg-blue-900/20 text-blue-300"
      }`}
    >
      <span className="mt-0.5 shrink-0">{hasInvariant ? "⚠" : "ℹ"}</span>
      <div className="flex-1">
        {relaxations.map((r, i) => (
          <div key={i}>
            <span className="font-semibold">Relaxed to fit:</span>{" "}
            {r.bridge} — dropped {r.relaxed.join(", ")}
            {r.severity === "invariant" && (
              <span className="ml-1 text-yellow-400 font-semibold">[invariant]</span>
            )}
          </div>
        ))}
      </div>
      <button
        onClick={() => setDismissed(true)}
        aria-label="Dismiss"
        className="shrink-0 text-muted hover:text-text leading-none"
      >
        ✕
      </button>
    </div>
  );
}
