import { useCallback, useEffect, useState } from "react";
import { api } from "../lib/api";
import { useJobReconcile } from "../lib/useJobReconcile";
import { useWorkerEvents } from "../lib/ws";
import type { EscalationOut, EscalationQueueResponse, WsEvent } from "../lib/types";

type View = "pending" | "completed";

function chips(items: string[], cls: string) {
  return items.map((t) => (
    <span key={t} className={`text-2xs px-1.5 py-0.5 rounded-full ${cls}`}>{t}</span>
  ));
}

function AlbumCard({
  esc, onDecide,
}: {
  esc: EscalationOut;
  onDecide: (decision: "accept" | "edit" | "reject", genres?: string[]) => void;
}) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(esc.proposed_genres.map((g) => g.term).join(", "));
  return (
    <div className="flex flex-col gap-1 mt-1 mb-2 ml-1 px-2 py-2 rounded border border-border">
      <div className="flex flex-wrap items-center gap-1">
        <span className="text-faint text-2xs uppercase tracking-wide">currently</span>
        {esc.prior_observed_leaf.length ? chips(esc.prior_observed_leaf, "bg-panel2 text-muted")
          : <span className="text-faint text-2xs">—</span>}
      </div>
      <div className="flex flex-wrap items-center gap-1">
        <span className="text-faint text-2xs uppercase tracking-wide">proposed</span>
        {esc.proposed_genres.length
          ? esc.proposed_genres.map((g) => (
              <span key={g.term} className="text-2xs px-1.5 py-0.5 rounded-full bg-accent/20 text-text">
                {g.term}{g.confidence != null ? ` ${g.confidence.toFixed(2)}` : ""}
              </span>))
          : <span className="text-faint text-2xs">(none resolved — use Edit)</span>}
      </div>
      {esc.escalate_reason && <div className="text-muted text-2xs">{esc.escalate_reason}</div>}
      {esc.dropped_file_tags.length > 0 && (
        <div className="text-danger text-2xs">⚠ would drop your file tag: {esc.dropped_file_tags.join(", ")}</div>
      )}
      {editing ? (
        <div className="flex flex-col gap-1 mt-1">
          <input
            autoFocus value={draft} onChange={(e) => setDraft(e.target.value)}
            placeholder="genre a, genre b, …"
            className="bg-panel2 border border-border rounded text-xs text-text px-2 py-1 outline-none"
          />
          <div className="flex gap-1.5">
            <button
              onClick={() => onDecide("edit", draft.split(",").map((s) => s.trim()).filter(Boolean))}
              className="text-2xs px-2 py-0.5 rounded bg-accent text-bg font-semibold"
            >Save edit</button>
            <button onClick={() => setEditing(false)}
              className="text-2xs px-2 py-0.5 rounded border border-border text-muted hover:text-text">Cancel</button>
          </div>
        </div>
      ) : (
        <div className="flex gap-1.5 mt-1">
          <button onClick={() => onDecide("accept")}
            className="text-2xs px-2 py-0.5 rounded bg-accent text-bg font-semibold">Accept (A)</button>
          <button onClick={() => setEditing(true)}
            className="text-2xs px-2 py-0.5 rounded border border-border text-muted hover:text-text">Edit</button>
          <button onClick={() => onDecide("reject")}
            className="text-2xs px-2 py-0.5 rounded border border-border text-muted hover:text-text">Reject (R)</button>
        </div>
      )}
    </div>
  );
}

export function GenreReviewPanel() {
  const [view, setView] = useState<View>("pending");
  const [data, setData] = useState<EscalationQueueResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [search, setSearch] = useState("");
  const [selected, setSelected] = useState<string | null>(null);
  const [publishJob, setPublishJob] = useState<string | null>(null);
  const [publishMsg, setPublishMsg] = useState("");
  const [sessionCount, setSessionCount] = useState(0);
  const [flash, setFlash] = useState<string | null>(null);

  const load = useCallback(async (q: string, v: View) => {
    try {
      const page = v === "pending" ? await api.reviewQueue(q) : await api.reviewCompleted(q);
      setData(page);
      setError(null);
    } catch (e) { setError(String(e)); }
  }, []);

  useEffect(() => { load(search, view); }, [search, view, load]);
  useEffect(() => { if (!flash) return; const t = setTimeout(() => setFlash(null), 1600); return () => clearTimeout(t); }, [flash]);

  useWorkerEvents(useCallback((e: WsEvent) => {
    if (!publishJob || e.job_id !== publishJob) return;
    if (e.type === "progress") {
      const d = ((e as Record<string, unknown>)["detail"] as string) ?? "";
      setPublishMsg(`publishing… ${d}`);
    }
    if (e.type === "done") { setPublishJob(null); setPublishMsg(""); load(search, view); }
  }, [publishJob, search, view, load]));

  useJobReconcile(publishJob, useCallback(() => {
    setPublishJob(null); setPublishMsg(""); load(search, view);
  }, [search, view, load]));

  const decide = useCallback(async (esc: EscalationOut, decision: "accept" | "edit" | "reject", genres?: string[]) => {
    // optimistic: drop from the pending list
    setData((prev) => prev && view === "pending"
      ? { ...prev, escalations: prev.escalations.filter((x) => x.album_id !== esc.album_id),
          pending_albums: Math.max(0, prev.pending_albums - 1), decided_albums: prev.decided_albums + 1 }
      : prev);
    try {
      await api.reviewDecision({ album_id: esc.album_id, decision, genres });
      setSessionCount((n) => n + 1);
      setFlash(`saved ✓ ${esc.artist} – ${esc.album}`);
    } catch (e) { setError(String(e)); load(search, view); }
  }, [view, search, load]);

  async function publishDecided() {
    setError(null);
    try { const { job_id } = await api.reviewPublish(); setPublishJob(job_id); setPublishMsg("starting…"); }
    catch (e) { setError(String(e)); }
  }

  const escalations = data?.escalations ?? [];
  const sel = escalations.find((x) => x.album_id === selected) ?? escalations[0] ?? null;

  function onKeyDown(e: React.KeyboardEvent) {
    if (view !== "pending" || !sel) return;
    // Don't hijack keystrokes meant for typing/editing. Keydown bubbles up
    // from the search box and the card's genre-edit input, so ignore events
    // originating in an editable element, and ignore any modifier combo
    // (e.g. Ctrl+A / Cmd+A to select text) — only a bare "a"/"r" is a shortcut.
    const t = e.target as HTMLElement;
    if (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.tagName === "SELECT" || t.isContentEditable) return;
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    const k = e.key.toLowerCase();
    if (k === "a") { e.preventDefault(); decide(sel, "accept"); }
    else if (k === "r") { e.preventDefault(); decide(sel, "reject"); }
    // Edit is via the card's Edit button (it owns the edit-input state).
  }

  const decidedK = data?.decided_albums ?? 0;

  return (
    <div data-testid="review-panel" className="h-full flex flex-col p-3 gap-2 outline-none" tabIndex={0} onKeyDown={onKeyDown}>
      <div className="flex items-center gap-1">
        {(["pending", "completed"] as View[]).map((v) => (
          <button key={v} onClick={() => { setView(v); setSelected(null); }}
            className={["text-2xs px-2 py-1 rounded border capitalize",
              view === v ? "border-accent/60 bg-panel2 text-text" : "border-border text-muted hover:text-text"].join(" ")}>
            {v}
          </button>
        ))}
        <div className="flex-1" />
        {sessionCount > 0 && <span className="text-accent text-2xs">✓ {sessionCount} this session</span>}
        {publishJob ? (
          <span className="text-faint text-2xs truncate max-w-[160px]">{publishMsg}</span>
        ) : decidedK > 0 ? (
          <button onClick={publishDecided}
            className="text-2xs px-2 py-1 rounded bg-accent text-bg font-semibold">
            Publish decided ({decidedK})
          </button>
        ) : null}
      </div>

      <div className="flex items-center gap-2 min-h-[16px]">
        <div className="text-muted text-xs flex-1">
          {data ? `${data.pending_albums} pending · ${data.decided_albums} decided` : "…"}
        </div>
        {flash && <span className="text-accent text-2xs truncate max-w-[160px]">{flash}</span>}
      </div>

      <input value={search} onChange={(e) => setSearch(e.target.value)} placeholder="Filter artist / album…"
        className="bg-panel2 border border-border rounded text-xs text-text px-2 py-1 placeholder:text-faint outline-none" />
      {error && <div className="text-danger text-2xs">{error}</div>}

      {data && escalations.length === 0 && (
        <div className="text-faint text-xs p-3">
          {view === "pending" ? "No escalations pending — all reviewed." : "No decisions yet."}
        </div>
      )}

      <div className="flex-1 overflow-auto flex flex-col gap-1">
        {escalations.map((esc) => (
          <div key={esc.album_id}>
            <div className={["w-full px-2 py-1 rounded flex items-center gap-2",
                sel?.album_id === esc.album_id ? "bg-panel2 text-text" : "text-muted hover:text-text"].join(" ")}>
              <button onClick={() => setSelected(sel?.album_id === esc.album_id ? null : esc.album_id)}
                className="text-left flex-1 min-w-0 flex items-center gap-2">
                <span className="text-xs flex-1 truncate select-text">{esc.artist} – {esc.album}</span>
                {esc.dropped_file_tags.length > 0 && <span className="text-danger text-2xs">⚠</span>}
              </button>
              <span className="text-faint text-2xs capitalize">{view === "completed" ? esc.status : ""}</span>
              <button
                title="Copy artist – album"
                onClick={(e) => {
                  e.stopPropagation();
                  const text = `${esc.artist} – ${esc.album}`;
                  navigator.clipboard?.writeText(text);
                  setFlash(`copied ✓ ${text}`);
                }}
                className="shrink-0 text-faint hover:text-text text-xs px-1 leading-none">⧉</button>
            </div>
            {sel?.album_id === esc.album_id && view === "pending" && (
              <AlbumCard esc={esc} onDecide={(d, g) => decide(esc, d, g)} />
            )}
            {sel?.album_id === esc.album_id && view === "completed" && (
              <div className="ml-1 mb-2 px-2 py-1 text-2xs text-muted flex items-center gap-2">
                <span className="flex-1">decided: {(esc.decision_genres ?? esc.proposed_genres.map((g) => g.term)).join(", ") || "—"}</span>
                <button onClick={async () => {
                  await api.reviewDecision({ album_id: esc.album_id, decision: "revert" });
                  load(search, view);
                }} className="underline hover:text-text">revert</button>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
