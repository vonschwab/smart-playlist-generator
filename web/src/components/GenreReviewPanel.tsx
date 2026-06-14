import { useCallback, useEffect, useState } from "react";
import { api } from "../lib/api";
import { useJobReconcile } from "../lib/useJobReconcile";
import { useWorkerEvents } from "../lib/ws";
import type {
  CompletedReviewResponse,
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

type View = "pending" | "completed";

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
  const [view, setView] = useState<View>("pending");
  const [data, setData] = useState<ReviewQueueResponse | null>(null);
  const [completed, setCompleted] = useState<CompletedReviewResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [search, setSearch] = useState("");
  const [selectedKey, setSelectedKey] = useState<string | null>(null);
  const [scanJobId, setScanJobId] = useState<string | null>(null);
  const [scanProgress, setScanProgress] = useState("");
  // Visible proof that decisions persist: a running session tally + a brief
  // per-decision confirmation. (Each accept/reject is already written to the
  // DB immediately — this just surfaces that so saved work doesn't feel lost.)
  const [sessionCount, setSessionCount] = useState(0);
  const [flash, setFlash] = useState<string | null>(null);

  const load = useCallback(async (q: string) => {
    try {
      const page = await api.reviewQueue(q);
      setData(page);
      setError(null);
    } catch (e) {
      setError(String(e));
    }
  }, []);

  const loadCompleted = useCallback(async (q: string) => {
    try {
      const page = await api.reviewCompleted(q);
      setCompleted(page);
      setError(null);
    } catch (e) {
      setError(String(e));
    }
  }, []);

  // Load whichever view is active (and refresh on search change).
  useEffect(() => {
    if (view === "pending") load(search);
    else loadCompleted(search);
  }, [view, search, load, loadCompleted]);

  // Auto-clear the "saved ✓" flash.
  useEffect(() => {
    if (!flash) return;
    const t = setTimeout(() => setFlash(null), 1600);
    return () => clearTimeout(t);
  }, [flash]);

  const refreshActive = useCallback(() => {
    if (view === "pending") load(search);
    else loadCompleted(search);
  }, [view, search, load, loadCompleted]);

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
          refreshActive();
        }
      },
      [scanJobId, refreshActive]
    )
  );

  // Backstop: if the scan's `done` event raced ahead of setScanJobId (fast scans),
  // the WS handler above misses it. Poll the registry so the button re-enables.
  useJobReconcile(
    scanJobId,
    useCallback(() => {
      setScanJobId(null);
      setScanProgress("");
      refreshActive();
    }, [refreshActive])
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

  // Persist one decision and surface the confirmation. Returns success so the
  // caller can roll back its optimistic update on failure.
  const applyDecision = useCallback(
    async (release: ReviewReleaseOut, term: ReviewTermOut, decision: "accept" | "reject" | "revert") => {
      try {
        await api.reviewDecision({
          release_key: release.release_key,
          term: term.term,
          decision,
        });
        if (decision === "revert") {
          setSessionCount((n) => Math.max(0, n - 1));
        } else {
          setSessionCount((n) => n + 1);
          setFlash(`saved ✓ ${term.term}`);
        }
        return true;
      } catch (e) {
        setError(String(e));
        return false;
      }
    },
    []
  );

  // Pending view: optimistic move of a term between pending/decided.
  const decide = useCallback(
    async (release: ReviewReleaseOut, term: ReviewTermOut, decision: "accept" | "reject" | "revert") => {
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
      const ok = await applyDecision(release, term, decision);
      if (!ok) load(search);
    },
    [applyDecision, load, search]
  );

  // Completed view: optimistic removal of a reverted term (it returns to Pending).
  const revertCompleted = useCallback(
    async (release: ReviewReleaseOut, term: ReviewTermOut) => {
      setCompleted((prev) => {
        if (!prev) return prev;
        let droppedRelease = false;
        const releases = prev.releases
          .map((r) => {
            if (r.release_key !== release.release_key) return r;
            const decided = r.decided.filter((x) => x.term !== term.term);
            if (decided.length === 0) droppedRelease = true;
            return { ...r, decided };
          })
          .filter((r) => r.decided.length > 0);
        return {
          ...prev,
          releases,
          decided_terms: Math.max(0, prev.decided_terms - 1),
          decided_releases: Math.max(0, prev.decided_releases - (droppedRelease ? 1 : 0)),
        };
      });
      const ok = await applyDecision(release, term, "revert");
      if (!ok) loadCompleted(search);
    },
    [applyDecision, loadCompleted, search]
  );

  const releases = (view === "pending" ? data?.releases : completed?.releases) ?? [];
  const selected =
    releases.find((r) => r.release_key === selectedKey) ?? releases[0] ?? null;

  function onKeyDown(e: React.KeyboardEvent) {
    if (view !== "pending" || !selected || selected.pending.length === 0) return;
    const key = e.key.toLowerCase();
    if (key === "a" || key === "r") {
      e.preventDefault();
      decide(selected, selected.pending[0], key === "a" ? "accept" : "reject");
    }
  }

  const decidedTotal = (view === "pending" ? data?.decided_terms : completed?.decided_terms) ?? 0;

  return (
    <div data-testid="review-panel" className="h-full flex flex-col p-3 gap-2 outline-none" tabIndex={0} onKeyDown={onKeyDown}>
      {/* View toggle */}
      <div className="flex items-center gap-1">
        {(["pending", "completed"] as View[]).map((v) => (
          <button
            key={v}
            onClick={() => { setView(v); setSelectedKey(null); }}
            className={[
              "text-[10px] px-2 py-1 rounded border capitalize",
              view === v ? "border-accent/60 bg-panel2 text-text" : "border-border text-muted hover:text-text",
            ].join(" ")}
          >
            {v}
          </button>
        ))}
        <div className="flex-1" />
        {sessionCount > 0 && (
          <span className="text-accent text-[10px]">✓ {sessionCount} this session</span>
        )}
        {scanJobId ? (
          <span className="text-faint text-[10px] truncate max-w-[140px]">{scanProgress}</span>
        ) : (
          <button
            onClick={startScan}
            className="text-[10px] px-2 py-1 rounded border border-border text-muted hover:text-text"
          >
            Scan
          </button>
        )}
      </div>

      {/* Header counts */}
      <div className="flex items-center gap-2 min-h-[16px]">
        <div className="text-muted text-xs flex-1">
          {view === "pending"
            ? data
              ? `${data.pending_releases} releases · ${data.pending_terms} terms · ${decidedTotal} decided`
              : "…"
            : completed
              ? `${completed.decided_releases} releases · ${completed.decided_terms} decided`
              : "…"}
        </div>
        {flash && <span className="text-accent text-[10px] truncate max-w-[160px]">{flash}</span>}
      </div>

      <input
        value={search}
        onChange={(e) => setSearch(e.target.value)}
        placeholder="Filter artist / album…"
        className="bg-panel2 border border-border rounded text-[11px] text-text px-2 py-1
                   placeholder:text-faint outline-none"
      />
      {error && <div className="text-danger text-[10px]">{error}</div>}

      {/* Empty states */}
      {view === "pending" && data && releases.length === 0 && !scanJobId && (
        <div className="text-faint text-xs p-3">
          Queue is empty. Run a scan to find genre terms needing review.
        </div>
      )}
      {view === "completed" && completed && releases.length === 0 && (
        <div className="text-faint text-xs p-3">
          No decisions yet. Accept or reject terms in the Pending view and they'll appear here.
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
                {view === "pending" ? r.pending.length : r.decided.length}
              </span>
            </button>
            {selected?.release_key === r.release_key && view === "pending" && (
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
            {selected?.release_key === r.release_key && view === "completed" && (
              <div className="flex flex-col gap-0.5 mt-1 mb-2 ml-1 text-[10px]">
                {r.decided.map((t) => (
                  <div key={t.term} className="flex items-center gap-2 px-2 py-0.5">
                    <span className="flex-1 truncate text-text">{t.term}</span>
                    <span className={t.status === "accepted" ? "text-accent" : "text-danger"}>
                      {t.status}
                    </span>
                    <button
                      onClick={() => revertCompleted(r, t)}
                      className="text-faint underline hover:text-text"
                    >
                      revert
                    </button>
                  </div>
                ))}
                {r.pending.length > 0 && (
                  <div className="text-faint px-2 mt-0.5">
                    {r.pending.length} still pending — finish in the Pending view.
                  </div>
                )}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
