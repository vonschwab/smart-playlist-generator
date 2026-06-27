import { useCallback, useEffect, useState } from "react";
import { api } from "../lib/api";
import { useJobReconcile } from "../lib/useJobReconcile";
import { useWorkerEvents } from "../lib/ws";
import type {
  TaxonomyProposal, TaxonomyQueueItem, TaxonomyQueueResponse, TaxonomyVerdict, WsEvent,
} from "../lib/types";

type View = "untriaged" | "completed";

// Mirrors enums.reject_reason in data/layered_genre_taxonomy.yaml.
const REJECT_REASONS = [
  "label", "artist_name", "release_title", "place", "format", "era", "user_list",
  "malformed", "joke_tag", "negative_tag", "retail_bucket", "source_noise", "unknown_noise",
];

function chips(items: string[], cls: string) {
  return items.map((t) => (
    <span key={t} className={`text-[10px] px-1.5 py-0.5 rounded-full ${cls}`}>{t}</span>
  ));
}

function VerdictSummary({ v }: { v: TaxonomyVerdict }) {
  const p = v.proposal;
  if (v.verdict === "reject") {
    return (
      <div className="text-[10px] text-muted">
        <span className="text-danger font-semibold">reject</span>
        {p.reject_reason ? ` · ${p.reject_reason}` : ""}
        {p.rationale ? <div className="text-faint mt-0.5">{p.rationale}</div> : null}
      </div>
    );
  }
  if (v.verdict === "alias") {
    return (
      <div className="text-[10px] text-muted">
        <span className="text-accent font-semibold">alias</span> → {p.canonical_target}
        {p.rationale ? <div className="text-faint mt-0.5">{p.rationale}</div> : null}
      </div>
    );
  }
  return (
    <div className="text-[10px] text-muted flex flex-col gap-0.5">
      <div>
        <span className="text-accent font-semibold">add</span>
        {" · "}{p.kind}{p.status ? ` (${p.status})` : ""}
        {p.specificity_score != null ? ` · spec ${p.specificity_score.toFixed(2)}` : ""}
      </div>
      {(p.parent_edges ?? []).length > 0 && (
        <div className="flex flex-wrap items-center gap-1">
          <span className="text-faint">parents</span>
          {(p.parent_edges ?? []).map((e) => (
            <span key={e.target} className="text-[10px] px-1.5 py-0.5 rounded-full bg-panel2 text-text">
              {e.target} <span className="text-faint">{e.edge_type} {e.weight}</span>
            </span>
          ))}
        </div>
      )}
      {(p.alias_variants ?? []).length > 0 && (
        <div className="flex flex-wrap items-center gap-1">
          <span className="text-faint">aliases</span>{chips(p.alias_variants ?? [], "bg-panel2 text-muted")}
        </div>
      )}
      {p.rationale ? <div className="text-faint">{p.rationale}</div> : null}
    </div>
  );
}

function TermCard({
  item, onDecide,
}: {
  item: TaxonomyQueueItem;
  onDecide: (verdict: "add" | "alias" | "reject", proposal: TaxonomyProposal | null,
             claude: TaxonomyProposal | null, humanEdited: boolean) => void;
}) {
  const [verdict, setVerdict] = useState<TaxonomyVerdict | null>(null);
  const [asking, setAsking] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState("");
  const [rejecting, setRejecting] = useState(false);
  const [reason, setReason] = useState(REJECT_REASONS[0]);

  async function ask() {
    setAsking(true); setError(null);
    try {
      const v = await api.taxonomyAdjudicate(item.term);
      setVerdict(v);
      setDraft(JSON.stringify(v.proposal, null, 2));
    } catch (e) { setError(String(e)); }
    finally { setAsking(false); }
  }

  function accept() {
    if (!verdict) return;
    onDecide(verdict.verdict, verdict.proposal, verdict.proposal, false);
  }

  function saveEdit() {
    if (!verdict) return;
    let parsed: TaxonomyProposal;
    try { parsed = JSON.parse(draft); }
    catch (e) { setError(`Invalid JSON: ${String(e)}`); return; }
    onDecide(verdict.verdict, parsed, verdict.proposal, true);
  }

  function reject() {
    onDecide("reject", { reject_reason: reason, rationale: verdict?.proposal.rationale ?? "" },
             verdict?.proposal ?? null, verdict ? verdict.verdict !== "reject" : true);
  }

  return (
    <div className="flex flex-col gap-1 mt-1 mb-2 ml-1 px-2 py-2 rounded border border-border">
      <div className="flex flex-wrap items-center gap-1">
        <span className="text-faint text-[9px] uppercase tracking-wide">reach</span>
        <span className="text-[10px] text-muted">{item.album_frequency} albums</span>
        {item.variants.length > 0 && (
          <>
            <span className="text-faint text-[9px] uppercase tracking-wide ml-2">spellings</span>
            {chips(item.variants, "bg-panel2 text-muted")}
          </>
        )}
      </div>
      {item.cooccurring_tags.length > 0 && (
        <div className="flex flex-wrap items-center gap-1">
          <span className="text-faint text-[9px] uppercase tracking-wide">with</span>
          {chips(item.cooccurring_tags.slice(0, 8), "bg-panel2 text-muted")}
        </div>
      )}
      {item.examples.length > 0 && (
        <div className="text-faint text-[10px] truncate">{item.examples.join(" · ")}</div>
      )}
      {error && <div className="text-danger text-[10px]">{error}</div>}

      {!verdict ? (
        <div className="flex gap-1.5 mt-1">
          <button onClick={ask} disabled={asking}
            className="text-[10px] px-2 py-0.5 rounded bg-accent text-bg font-semibold disabled:opacity-50">
            {asking ? "asking Claude…" : "Ask Claude"}
          </button>
        </div>
      ) : (
        <div className="flex flex-col gap-1 mt-1">
          <VerdictSummary v={verdict} />
          {editing ? (
            <div className="flex flex-col gap-1">
              <textarea value={draft} onChange={(e) => setDraft(e.target.value)} rows={8}
                className="bg-panel2 border border-border rounded text-[10px] font-mono text-text px-2 py-1 outline-none" />
              <div className="flex gap-1.5">
                <button onClick={saveEdit}
                  className="text-[10px] px-2 py-0.5 rounded bg-accent text-bg font-semibold">Save edit</button>
                <button onClick={() => setEditing(false)}
                  className="text-[10px] px-2 py-0.5 rounded border border-border text-muted hover:text-text">Cancel</button>
              </div>
            </div>
          ) : rejecting ? (
            <div className="flex items-center gap-1.5">
              <select value={reason} onChange={(e) => setReason(e.target.value)}
                className="bg-panel2 border border-border rounded text-[10px] text-text px-1.5 py-0.5 outline-none">
                {REJECT_REASONS.map((r) => <option key={r} value={r}>{r}</option>)}
              </select>
              <button onClick={reject}
                className="text-[10px] px-2 py-0.5 rounded bg-danger/80 text-bg font-semibold">Confirm reject</button>
              <button onClick={() => setRejecting(false)}
                className="text-[10px] px-2 py-0.5 rounded border border-border text-muted hover:text-text">Cancel</button>
            </div>
          ) : (
            <div className="flex gap-1.5">
              <button onClick={accept}
                className="text-[10px] px-2 py-0.5 rounded bg-accent text-bg font-semibold">Accept</button>
              <button onClick={() => setEditing(true)}
                className="text-[10px] px-2 py-0.5 rounded border border-border text-muted hover:text-text">Edit</button>
              {verdict.verdict !== "reject" && (
                <button onClick={() => setRejecting(true)}
                  className="text-[10px] px-2 py-0.5 rounded border border-border text-muted hover:text-text">Reject</button>
              )}
              <button onClick={() => setVerdict(null)}
                className="text-[10px] px-2 py-0.5 rounded border border-border text-faint hover:text-text">Re-ask</button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

interface ApplyStats {
  ok: boolean;
  added?: number;
  aliased?: number;
  rejected?: number;
  applied_terms?: string[];
  deferred_edges?: { source: string; target: string; reason: string }[];
  backup?: string;
  new_version?: string;
  validation_failures?: [string, string[]][];
}

export function TaxonomyReviewPanel() {
  const [view, setView] = useState<View>("untriaged");
  const [data, setData] = useState<TaxonomyQueueResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [search, setSearch] = useState("");
  const [selected, setSelected] = useState<string | null>(null);
  const [applyJob, setApplyJob] = useState<string | null>(null);
  const [applyMsg, setApplyMsg] = useState("");
  const [applyStats, setApplyStats] = useState<ApplyStats | null>(null);
  const [sessionCount, setSessionCount] = useState(0);
  const [flash, setFlash] = useState<string | null>(null);

  const load = useCallback(async (q: string, v: View) => {
    try {
      const page = v === "untriaged" ? await api.taxonomyQueue(q) : await api.taxonomyCompleted(q);
      setData(page);
      setError(null);
    } catch (e) { setError(String(e)); }
  }, []);

  useEffect(() => { load(search, view); }, [search, view, load]);
  useEffect(() => { if (!flash) return; const t = setTimeout(() => setFlash(null), 1600); return () => clearTimeout(t); }, [flash]);

  useWorkerEvents(useCallback((e: WsEvent) => {
    if (!applyJob || e.job_id !== applyJob) return;
    if (e.type === "progress") setApplyMsg(`applying… ${((e as Record<string, unknown>)["detail"] as string) ?? ""}`);
    if (e.type === "result") setApplyStats(e as unknown as ApplyStats);
    if (e.type === "done") { setApplyJob(null); setApplyMsg(""); load(search, "completed"); }
  }, [applyJob, search, load]));

  useJobReconcile(applyJob, useCallback(() => {
    setApplyJob(null); setApplyMsg(""); load(search, view);
  }, [search, view, load]));

  const decide = useCallback(async (
    item: TaxonomyQueueItem, verdict: "add" | "alias" | "reject",
    proposal: TaxonomyProposal | null, claude: TaxonomyProposal | null, humanEdited: boolean,
  ) => {
    // optimistic: drop from the untriaged list
    setData((prev) => prev && view === "untriaged"
      ? { ...prev, terms: prev.terms.filter((x) => x.term !== item.term),
          untriaged_terms: Math.max(0, prev.untriaged_terms - 1), decided_terms: prev.decided_terms + 1 }
      : prev);
    try {
      await api.taxonomyDecision({
        term: item.term, raw_term: item.raw_term, verdict, proposal, claude, human_edited: humanEdited });
      setSessionCount((n) => n + 1);
      setFlash(`saved ✓ ${item.term} → ${verdict}`);
    } catch (e) { setError(String(e)); load(search, view); }
  }, [view, search, load]);

  async function revert(term: string) {
    try { await api.taxonomyDecision({ term, verdict: "revert" }); load(search, view); }
    catch (e) { setError(String(e)); }
  }

  async function applyDecisions() {
    setError(null); setApplyStats(null);
    try { const { job_id } = await api.taxonomyApply(); setApplyJob(job_id); setApplyMsg("starting…"); }
    catch (e) { setError(String(e)); }
  }

  const terms = data?.terms ?? [];
  const sel = terms.find((x) => x.term === selected) ?? null;
  const decidedK = data?.decided_terms ?? 0;

  // "M albums re-classify" estimate: sum reach of applied add/alias terms we have loaded.
  const reclassifyEstimate = applyStats?.applied_terms
    ? terms.filter((t) => applyStats.applied_terms!.includes(t.term))
        .reduce((acc, t) => acc + t.album_frequency, 0)
    : 0;

  return (
    <div data-testid="taxonomy-panel" className="h-full flex flex-col p-3 gap-2 outline-none">
      <div className="flex items-center gap-1">
        {(["untriaged", "completed"] as View[]).map((v) => (
          <button key={v} onClick={() => { setView(v); setSelected(null); }}
            className={["text-[10px] px-2 py-1 rounded border capitalize",
              view === v ? "border-accent/60 bg-panel2 text-text" : "border-border text-muted hover:text-text"].join(" ")}>
            {v}
          </button>
        ))}
        <div className="flex-1" />
        {sessionCount > 0 && <span className="text-accent text-[10px]">✓ {sessionCount} this session</span>}
        {applyJob ? (
          <span className="text-faint text-[10px] truncate max-w-[160px]">{applyMsg}</span>
        ) : decidedK > 0 ? (
          <button onClick={applyDecisions}
            className="text-[10px] px-2 py-1 rounded bg-accent text-bg font-semibold">
            Apply {decidedK} decision{decidedK === 1 ? "" : "s"}
          </button>
        ) : null}
      </div>

      <div className="flex items-center gap-2 min-h-[16px]">
        <div className="text-muted text-xs flex-1">
          {data ? `${data.untriaged_terms} untriaged · ${data.decided_terms} decided` : "…"}
        </div>
        {flash && <span className="text-accent text-[10px] truncate max-w-[180px]">{flash}</span>}
      </div>

      {applyStats && (
        <div className={["text-[10px] rounded px-2 py-1.5 border",
          applyStats.ok ? "border-accent/40 bg-panel2 text-muted" : "border-danger/50 bg-panel2 text-danger"].join(" ")}>
          {applyStats.ok ? (
            <>
              <div className="text-text">
                Applied: {applyStats.added ?? 0} added · {applyStats.aliased ?? 0} aliased · {applyStats.rejected ?? 0} rejected
                {applyStats.new_version ? ` · ${applyStats.new_version}` : ""}
              </div>
              {reclassifyEstimate > 0 && (
                <div>~{reclassifyEstimate} album(s) tagged with these terms will re-classify on the next publish.</div>
              )}
              {(applyStats.deferred_edges ?? []).length > 0 && (
                <div className="text-faint">deferred edges: {(applyStats.deferred_edges ?? [])
                  .map((d) => `${d.source}→${d.target}`).join(", ")}</div>
              )}
              {applyStats.backup && <div className="text-faint truncate">backup: {applyStats.backup}</div>}
            </>
          ) : (
            <>
              <div className="font-semibold">Validation failed — nothing written.</div>
              {(applyStats.validation_failures ?? []).map(([term, errs]) => (
                <div key={term}>{term}: {errs.join("; ")}</div>
              ))}
            </>
          )}
        </div>
      )}

      <input value={search} onChange={(e) => setSearch(e.target.value)} placeholder="Filter term…"
        className="bg-panel2 border border-border rounded text-[11px] text-text px-2 py-1 placeholder:text-faint outline-none" />
      {error && <div className="text-danger text-[10px]">{error}</div>}

      {data && terms.length === 0 && (
        <div className="text-faint text-xs p-3">
          {view === "untriaged" ? "No untriaged terms — vocabulary is fully covered." : "No decisions yet."}
        </div>
      )}

      <div className="flex-1 overflow-auto flex flex-col gap-1">
        {terms.map((item) => (
          <div key={item.term}>
            <div className={["w-full px-2 py-1 rounded flex items-center gap-2",
                sel?.term === item.term ? "bg-panel2 text-text" : "text-muted hover:text-text"].join(" ")}>
              <button onClick={() => setSelected(sel?.term === item.term ? null : item.term)}
                className="text-left flex-1 min-w-0 flex items-center gap-2">
                <span className="text-xs flex-1 truncate select-text">{item.raw_term}</span>
                <span className="text-faint text-[10px]">{item.album_frequency}</span>
              </button>
              {view === "completed" && (
                <span className="text-faint text-[10px] capitalize">{item.decision?.verdict ?? ""}</span>
              )}
            </div>
            {sel?.term === item.term && view === "untriaged" && (
              <TermCard item={item}
                onDecide={(v, p, c, h) => decide(item, v, p, c, h)} />
            )}
            {sel?.term === item.term && view === "completed" && (
              <div className="ml-1 mb-2 px-2 py-1 text-[10px] text-muted flex items-center gap-2">
                <span className="flex-1">decided: {item.decision?.verdict ?? "—"}
                  {item.decision?.status === "applied" ? " (applied)" : ""}</span>
                {item.decision?.status !== "applied" && (
                  <button onClick={() => revert(item.term)} className="underline hover:text-text">revert</button>
                )}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
