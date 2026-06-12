import { useCallback, useEffect, useState } from "react";
import { api } from "../lib/api";
import { useWorkerEvents } from "../lib/ws";
import type {
  ReviewQueueResponse,
  ReviewReleaseOut,
  ReviewTermOut,
  WsEvent,
} from "../lib/types";

const BASIS_LABEL: Record<string, string> = {
  layered_taxonomy: "unknown term",
  hybrid_provisional: "provisional",
  hybrid_fusion: "uncertain",
};

function TermRow({
  term,
  onDecide,
  focused,
}: {
  term: ReviewTermOut;
  onDecide: (decision: "accept" | "reject") => void;
  focused: boolean;
}) {
  return (
    <div
      className={[
        "px-2 py-1.5 rounded border",
        focused ? "border-accent/50 bg-panel2" : "border-border",
      ].join(" ")}
    >
      <div className="flex items-center gap-2">
        <span className="text-text text-xs font-medium flex-1 truncate">{term.term}</span>
        <span className="text-faint text-[9px] uppercase tracking-wide">
          {BASIS_LABEL[term.basis] ?? term.basis}
        </span>
        {term.confidence != null && (
          <span className="text-faint text-[10px]">{term.confidence.toFixed(2)}</span>
        )}
      </div>
      {term.reason && <div className="text-muted text-[10px] mt-0.5">{term.reason}</div>}
      {term.sources.length > 0 && (
        <div className="text-faint text-[9px] mt-0.5">{term.sources.join(" · ")}</div>
      )}
      <div className="flex gap-1.5 mt-1.5">
        <button
          onClick={() => onDecide("accept")}
          className="text-[10px] px-2 py-0.5 rounded bg-accent text-bg font-semibold"
        >
          Accept (A)
        </button>
        <button
          onClick={() => onDecide("reject")}
          className="text-[10px] px-2 py-0.5 rounded border border-border text-muted hover:text-text"
        >
          Reject (R)
        </button>
      </div>
    </div>
  );
}

export function GenreReviewPanel() {
  const [data, setData] = useState<ReviewQueueResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [search, setSearch] = useState("");
  const [selectedKey, setSelectedKey] = useState<string | null>(null);
  const [scanJobId, setScanJobId] = useState<string | null>(null);
  const [scanProgress, setScanProgress] = useState("");

  const load = useCallback(async (q: string) => {
    try {
      const page = await api.reviewQueue(q);
      setData(page);
      setError(null);
    } catch (e) {
      setError(String(e));
    }
  }, []);

  useEffect(() => {
    load(search);
  }, [load, search]);

  useWorkerEvents(
    useCallback(
      (e: WsEvent) => {
        if (!scanJobId || e.job_id !== scanJobId) return;
        if (e.type === "progress") {
          const cur = (e as Record<string, unknown>)["current"] as number | undefined;
          const total = (e as Record<string, unknown>)["total"] as number | undefined;
          const detail = ((e as Record<string, unknown>)["detail"] as string) ?? "";
          setScanProgress(`${cur ?? "?"}/${total ?? "?"} ${detail}`);
        }
        if (e.type === "done") {
          setScanJobId(null);
          setScanProgress("");
          load(search);
        }
      },
      [scanJobId, load, search]
    )
  );

  async function startScan() {
    setError(null);
    try {
      const { job_id } = await api.reviewScan();
      setScanJobId(job_id);
      setScanProgress("starting…");
    } catch (e) {
      setError(String(e));
    }
  }

  const decide = useCallback(
    async (release: ReviewReleaseOut, term: ReviewTermOut, decision: "accept" | "reject" | "revert") => {
      // Optimistic update; reload on error.
      setData((prev) => {
        if (!prev) return prev;
        const releases = prev.releases
          .map((r) => {
            if (r.release_key !== release.release_key) return r;
            if (decision === "revert") {
              const t = r.decided.find((x) => x.term === term.term);
              if (!t) return r;
              return {
                ...r,
                decided: r.decided.filter((x) => x.term !== term.term),
                pending: [...r.pending, { ...t, status: "pending" as const }],
              };
            }
            const t = r.pending.find((x) => x.term === term.term);
            if (!t) return r;
            const status = decision === "accept" ? ("accepted" as const) : ("rejected" as const);
            return {
              ...r,
              pending: r.pending.filter((x) => x.term !== term.term),
              decided: [...r.decided, { ...t, status }],
            };
          })
          .filter((r) => r.pending.length > 0 || r.release_key === release.release_key);
        const delta = decision === "revert" ? 1 : -1;
        return { ...prev, releases, pending_terms: prev.pending_terms + delta };
      });
      try {
        await api.reviewDecision({
          release_key: release.release_key,
          term: term.term,
          decision,
        });
      } catch (e) {
        setError(String(e));
        load(search);
      }
    },
    [load, search]
  );

  const releases = data?.releases ?? [];
  const selected =
    releases.find((r) => r.release_key === selectedKey) ?? releases[0] ?? null;

  function onKeyDown(e: React.KeyboardEvent) {
    if (!selected || selected.pending.length === 0) return;
    const key = e.key.toLowerCase();
    if (key === "a" || key === "r") {
      e.preventDefault();
      decide(selected, selected.pending[0], key === "a" ? "accept" : "reject");
    }
  }

  return (
    <div className="h-full flex flex-col p-3 gap-2 outline-none" tabIndex={0} onKeyDown={onKeyDown}>
      {/* Header */}
      <div className="flex items-center gap-2">
        <div className="text-muted text-xs flex-1">
          {data ? `${data.pending_releases} releases · ${data.pending_terms} terms` : "…"}
        </div>
        {scanJobId ? (
          <span className="text-faint text-[10px] truncate max-w-[160px]">{scanProgress}</span>
        ) : (
          <button
            onClick={startScan}
            className="text-[10px] px-2 py-1 rounded border border-border text-muted hover:text-text"
          >
            Scan
          </button>
        )}
      </div>
      <input
        value={search}
        onChange={(e) => setSearch(e.target.value)}
        placeholder="Filter artist / album…"
        className="bg-panel2 border border-border rounded text-[11px] text-text px-2 py-1
                   placeholder:text-faint outline-none"
      />
      {error && <div className="text-danger text-[10px]">{error}</div>}

      {/* Empty state */}
      {data && releases.length === 0 && !scanJobId && (
        <div className="text-faint text-xs p-3">
          Queue is empty. Run a scan to find genre terms needing review.
        </div>
      )}

      {/* Release list */}
      <div className="flex-1 overflow-auto flex flex-col gap-1">
        {releases.map((r) => (
          <div key={r.release_key}>
            <button
              onClick={() =>
                setSelectedKey(selected?.release_key === r.release_key ? null : r.release_key)
              }
              className={[
                "w-full text-left px-2 py-1 rounded flex items-center gap-2",
                selected?.release_key === r.release_key
                  ? "bg-panel2 text-text"
                  : "text-muted hover:text-text",
              ].join(" ")}
            >
              <span className="text-xs flex-1 truncate">
                {r.artist} – {r.album}
              </span>
              <span className="bg-chip text-chipText text-[10px] px-1.5 rounded-full">
                {r.pending.length}
              </span>
            </button>
            {selected?.release_key === r.release_key && (
              <div className="flex flex-col gap-1 mt-1 mb-2 ml-1">
                {r.pending.map((t, i) => (
                  <TermRow
                    key={t.term}
                    term={t}
                    focused={i === 0}
                    onDecide={(d) => decide(r, t, d)}
                  />
                ))}
                {r.pending.length > 1 && (
                  <div className="flex gap-1.5 mt-0.5">
                    <button
                      onClick={async () => {
                        for (const t of r.pending.slice()) {
                          await decide(r, t, "accept");
                        }
                      }}
                      className="text-[10px] px-2 py-0.5 rounded border border-border text-muted hover:text-text"
                    >
                      Accept all
                    </button>
                    <button
                      onClick={async () => {
                        for (const t of r.pending.slice()) {
                          await decide(r, t, "reject");
                        }
                      }}
                      className="text-[10px] px-2 py-0.5 rounded border border-border text-muted hover:text-text"
                    >
                      Reject all
                    </button>
                  </div>
                )}
                {r.decided.length > 0 && (
                  <details className="text-[10px] text-faint">
                    <summary className="cursor-pointer select-none">
                      {r.decided.length} decided
                    </summary>
                    <div className="flex flex-col gap-0.5 mt-1">
                      {r.decided.map((t) => (
                        <div key={t.term} className="flex items-center gap-2 px-2">
                          <span className="flex-1 truncate">{t.term}</span>
                          <span className={t.status === "accepted" ? "text-accent" : "text-danger"}>
                            {t.status}
                          </span>
                          <button
                            onClick={() => decide(r, t, "revert")}
                            className="underline hover:text-text"
                          >
                            revert
                          </button>
                        </div>
                      ))}
                    </div>
                  </details>
                )}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
