import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";

type Tag = { name: string; release_count: number; confidence: number };

/**
 * Unified "Style" popover for artist mode: a fixed-width trigger that opens a
 * floating card holding the Style Spread selector + the genre-lean chips.
 *
 * The card is PORTALED to document.body and positioned `fixed` under the
 * trigger, so it escapes the toolbar's `overflow-hidden` ancestors (the
 * GenerateControls root and the App content wrapper) instead of being clipped.
 * Presentational only — all steering/spread state is owned by the parent
 * (GenerateControls), which threads it into the generate request.
 */
export function StylePopover({
  artistVariety,
  onVarietyChange,
  artistTags,
  steeringTags,
  onToggleTag,
  tagsFetched,
}: {
  artistVariety: string;
  onVarietyChange: (v: string) => void;
  artistTags: Tag[];
  steeringTags: string[];
  onToggleTag: (name: string) => void;
  tagsFetched: boolean;
}) {
  const [open, setOpen] = useState(false);
  const [pos, setPos] = useState<{ top: number; left: number } | null>(null);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const cardRef = useRef<HTMLDivElement>(null);

  // Position the card under the trigger (viewport-fixed) while open; keep it
  // pinned to the trigger through resize / ancestor scroll.
  useLayoutEffect(() => {
    if (!open) return;
    function place() {
      const r = triggerRef.current?.getBoundingClientRect();
      if (r) setPos({ top: r.bottom + 4, left: r.left });
    }
    place();
    window.addEventListener("resize", place);
    window.addEventListener("scroll", place, true); // capture: catch ancestor scrolls
    return () => {
      window.removeEventListener("resize", place);
      window.removeEventListener("scroll", place, true);
    };
  }, [open]);

  // Close on outside-click (trigger and portaled card are separate trees) / Escape.
  useEffect(() => {
    if (!open) return;
    function onOutside(e: MouseEvent) {
      const t = e.target as Node;
      if (triggerRef.current?.contains(t) || cardRef.current?.contains(t)) return;
      setOpen(false);
    }
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") setOpen(false);
    }
    document.addEventListener("mousedown", onOutside);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onOutside);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  const count = steeringTags.length;

  const card =
    open && pos
      ? createPortal(
          <div
            ref={cardRef}
            id="style-popover-card"
            style={{ position: "fixed", top: pos.top, left: pos.left }}
            className="z-50 w-[280px] bg-[#16181d] border border-[#23262d] rounded shadow-xl p-3 flex flex-col gap-3"
          >
            {/* Spread */}
            <div className="flex items-center gap-2">
              <label
                className="uppercase tracking-wide text-[10px] text-[#8b939d] w-14 shrink-0"
                title="How much of the seed artist's stylistic range to draw from. Some artists span many styles across their catalog — Focused draws from one corner, Sprawling draws from across the full range."
              >
                spread
              </label>
              <select
                value={artistVariety}
                onChange={(e) => onVarietyChange(e.target.value)}
                title="How much of the seed artist's stylistic range to draw from."
                className="flex-1 bg-[#0c0e12] border border-[#23262d] rounded text-[11px] text-[#e6e9ec] px-2 py-[3px]"
              >
                <option value="focused">focused</option>
                <option value="balanced">balanced</option>
                <option value="sprawling">sprawling</option>
              </select>
            </div>

            {/* Genre lean */}
            <div className="flex flex-col gap-1.5">
              <span className="uppercase tracking-wide text-[10px] text-[#8b939d]">
                lean toward genres
              </span>
              {artistTags.length > 0 ? (
                <div className="flex flex-wrap items-center gap-1.5" data-testid="steering-chips">
                  {artistTags.map((t) => {
                    const on = steeringTags.includes(t.name);
                    const capped = !on && steeringTags.length >= 3;
                    return (
                      <button
                        key={t.name}
                        type="button"
                        disabled={capped}
                        onClick={() => onToggleTag(t.name)}
                        title={`${t.release_count} release${t.release_count === 1 ? "" : "s"}`}
                        className={
                          "rounded-full border px-2 py-0.5 text-[11px] transition-colors " +
                          (on
                            ? "border-[#5eead4] bg-[#5eead4]/15 text-[#5eead4]"
                            : capped
                              ? "border-[#23262d] text-[#8b939d] opacity-40"
                              : "border-[#23262d] text-[#8b939d] hover:bg-[#1e2229]")
                        }
                      >
                        {t.name}
                      </button>
                    );
                  })}
                </div>
              ) : tagsFetched ? (
                <p className="text-[11px] text-[#5b6470]">
                  No published genres for this artist — run enrichment publish to enable tag steering.
                </p>
              ) : (
                <p className="text-[11px] text-[#5b6470]">Loading…</p>
              )}
            </div>
          </div>,
          document.body,
        )
      : null;

  return (
    <>
      <button
        ref={triggerRef}
        type="button"
        aria-expanded={open}
        aria-controls="style-popover-card"
        onClick={() => setOpen((v) => !v)}
        title="Style: how much of the artist's stylistic range to draw from, plus which genres to lean toward."
        className="flex items-center gap-1.5 bg-[#0c0e12] border border-[#23262d] rounded text-[11px] text-[#e6e9ec] px-2.5 py-[3px]"
      >
        <span className="uppercase tracking-wide text-[10px] text-[#8b939d]">style</span>
        {count > 0 && (
          <span className="inline-flex items-center justify-center min-w-[16px] h-[16px] px-1 rounded-full bg-[#5eead4] text-[#0f1115] text-[10px] font-bold leading-none">
            {count}
          </span>
        )}
        <span className="text-[#5b6470]">▾</span>
      </button>
      {card}
    </>
  );
}
