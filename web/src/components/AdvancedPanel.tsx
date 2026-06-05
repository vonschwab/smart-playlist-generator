import { useState } from "react";

export function AdvancedPanel() {
  const [tab, setTab] = useState<"advanced" | "review">("advanced");
  return (
    <div className="h-full flex flex-col">
      <div className="flex gap-1 px-2 pt-2 bg-panel2">
        {(["advanced", "review"] as const).map((t) => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={`text-[11px] px-2.5 py-1.5 rounded-t ${
              tab === t ? "text-accent bg-bg" : "text-muted"
            }`}
          >
            {t === "advanced" ? "Advanced" : "Genre Review"}
          </button>
        ))}
      </div>
      <div className="p-3 text-xs text-muted">
        {tab === "advanced"
          ? "Advanced settings land in a later phase."
          : "Genre review lands in a later phase."}
      </div>
    </div>
  );
}
