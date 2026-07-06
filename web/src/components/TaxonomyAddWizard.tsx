// web/src/components/TaxonomyAddWizard.tsx
import { useState } from "react";
import { api } from "../lib/api";
import { GenreAutocomplete } from "./GenreAutocomplete";
import type { TaxonomyParentEdge, TaxonomyProposal, TaxonomyQueueItem } from "../lib/types";

// Mirrors enums.facet_type in data/layered_genre_taxonomy.yaml.
const FACET_TYPES = [
  "mood", "texture", "instrumentation", "production", "era", "region",
  "function", "vocal", "scene", "format", "rhythm",
];

// Governed edge presets (taxonomy-growth skill edge-shape table). No free-form
// weights in v1 — ad-hoc weights are how hand-built graphs go inconsistent.
const EDGE_PRESETS = {
  is_a: { edge_type: "is_a", weight: 0.75, confidence: 0.85, label: "strong parent — X is a kind of it" },
  family_context: { edge_type: "family_context", weight: 0.55, confidence: 0.8, label: "family context — X belongs in its orbit" },
} as const;
type EdgePresetKey = keyof typeof EDGE_PRESETS;

type Kind = "genre" | "subgenre" | "facet";

// Specificity ladder (taxonomy-growth skill governance).
const SPECIFICITY: Record<Kind, { def: number; lo: number; hi: number }> = {
  genre: { def: 0.55, lo: 0.48, hi: 0.66 },
  subgenre: { def: 0.7, lo: 0.62, hi: 0.82 },
  facet: { def: 0.5, lo: 0, hi: 1 },
};

interface ParentPick { target: string; preset: EdgePresetKey }

const btn = "text-[10px] px-2 py-0.5 rounded";
const btnPrimary = `${btn} bg-accent text-bg font-semibold disabled:opacity-50`;
const btnGhost = `${btn} border border-border text-muted hover:text-text`;
const input = "bg-panel2 border border-border rounded text-[10px] text-text px-1.5 py-0.5 outline-none";

export function TaxonomyAddWizard({
  item, onStage, onCancel,
}: {
  item: TaxonomyQueueItem;
  onStage: (proposal: TaxonomyProposal) => void;
  onCancel: () => void;
}) {
  const [step, setStep] = useState(1);
  const [kind, setKind] = useState<Kind>("genre");
  const [name, setName] = useState(item.term);
  const [facetType, setFacetType] = useState("instrumentation");
  const [parents, setParents] = useState<ParentPick[]>([]);
  const [parentDraft, setParentDraft] = useState("");
  const [parentPreset, setParentPreset] = useState<EdgePresetKey>("is_a");
  const [similar, setSimilar] = useState<string[]>([]);
  const [similarDraft, setSimilarDraft] = useState("");
  const [specificity, setSpecificity] = useState(SPECIFICITY.genre.def);
  const [specTouched, setSpecTouched] = useState(false);
  const [aliases, setAliases] = useState<string[]>([]);
  const [aliasDraft, setAliasDraft] = useState("");
  const [rationale, setRationale] = useState("");
  const [errors, setErrors] = useState<string[] | null>(null);
  const [validating, setValidating] = useState(false);
  const [validateFailed, setValidateFailed] = useState<string | null>(null);

  const isFacet = kind === "facet";
  const spec = SPECIFICITY[kind];

  function pickKind(k: Kind) {
    setKind(k);
    if (!specTouched) setSpecificity(SPECIFICITY[k].def);
  }

  function buildProposal(): TaxonomyProposal {
    const cleanName = name.trim().toLowerCase();
    const aliasSet = new Set(aliases.map((a) => a.trim().toLowerCase()).filter(Boolean));
    // Rename governance: keep the raw queue spelling resolvable (ingest skips
    // a variant identical to the record name, so this is always safe).
    if (item.term.trim().toLowerCase() !== cleanName) aliasSet.add(item.term.trim().toLowerCase());
    return {
      name: cleanName,
      kind,
      status: "active",
      specificity_score: specificity,
      parent_edges: isFacet ? [] : parents.map((p): TaxonomyParentEdge => ({
        target: p.target,
        edge_type: EDGE_PRESETS[p.preset].edge_type,
        weight: EDGE_PRESETS[p.preset].weight,
        confidence: EDGE_PRESETS[p.preset].confidence,
      })),
      similar_to: isFacet ? [] : similar,
      alias_variants: [...aliasSet],
      term_kind_confirm: isFacet ? "facet" : "genre",
      facet_type: isFacet ? facetType : null,
      canonical_target: null,
      rationale: rationale.trim(),
    };
  }

  async function validate() {
    setValidating(true); setErrors(null); setValidateFailed(null);
    try {
      const r = await api.taxonomyValidate(buildProposal());
      setErrors(r.errors ?? []);
    } catch (e) {
      setValidateFailed(String(e)); // endpoint down ⇒ never "stage anyway"
    } finally { setValidating(false); }
  }

  function goto(n: number) {
    setStep(n);
    if (n === 4) void validate();
    else { setErrors(null); setValidateFailed(null); }
  }

  const nextDisabled =
    (step === 1 && !name.trim()) ||
    (step === 2 && !isFacet && parents.length === 0);

  const stageDisabled = validating || validateFailed !== null || errors === null || errors.length > 0;

  return (
    <div data-testid="taxonomy-add-wizard" className="flex flex-col gap-1.5 mt-1 px-2 py-2 rounded border border-border">
      <div className="text-faint text-[9px] uppercase tracking-wide">Add manually · step {step}/4</div>

      {step === 1 && (
        <div className="flex flex-col gap-1.5">
          <div className="flex items-center gap-1.5 flex-wrap">
            <span className="text-faint text-[10px]">this term is a</span>
            {(["genre", "subgenre", "facet"] as Kind[]).map((k) => (
              <button key={k} data-testid={`wizard-kind-${k}`} onClick={() => pickKind(k)}
                className={[btn, "border capitalize",
                  kind === k ? "border-accent/60 bg-panel2 text-text" : "border-border text-muted hover:text-text"].join(" ")}>
                {k}
              </button>
            ))}
          </div>
          <div className="text-faint text-[10px]">
            Instrument-led terms ("jazz piano", "jazz guitar") are usually <span className="text-text">facets</span> (instrumentation),
            not genres — unless there's a genuine scene/style tradition beyond the instrument.
          </div>
          <div className="flex items-center gap-1.5">
            <span className="text-faint text-[10px]">canonical name</span>
            <input data-testid="wizard-name" value={name} onChange={(e) => setName(e.target.value)}
              className={`${input} min-w-[200px]`} />
          </div>
          {name.trim().toLowerCase() !== item.term.trim().toLowerCase() && (
            <div className="text-faint text-[10px]">"{item.term}" will be kept as an alias of "{name.trim().toLowerCase()}".</div>
          )}
        </div>
      )}

      {step === 2 && !isFacet && (
        <div className="flex flex-col gap-1.5">
          <div className="flex items-center gap-1.5 flex-wrap">
            <span className="text-faint text-[10px]">parent</span>
            <GenreAutocomplete value={parentDraft} onChange={setParentDraft} onPick={setParentDraft}
              placeholder="existing canonical genre, e.g. indie rock"
              className={`${input} min-w-[200px]`} autoFocus data-testid="wizard-parent-input" />
            <select data-testid="wizard-parent-preset" value={parentPreset}
              onChange={(e) => setParentPreset(e.target.value as EdgePresetKey)} className={input}>
              {Object.entries(EDGE_PRESETS).map(([k, v]) => <option key={k} value={k}>{v.label}</option>)}
            </select>
            <button data-testid="wizard-parent-add" disabled={!parentDraft.trim()}
              onClick={() => {
                const t = parentDraft.trim().toLowerCase();
                if (t && !parents.some((p) => p.target === t)) setParents([...parents, { target: t, preset: parentPreset }]);
                setParentDraft("");
              }} className={btnPrimary}>Add parent</button>
          </div>
          {parents.length > 0 && (
            <div className="flex flex-wrap items-center gap-1">
              {parents.map((p) => (
                <span key={p.target} className="text-[10px] px-1.5 py-0.5 rounded-full bg-panel2 text-text">
                  {p.target} <span className="text-faint">{EDGE_PRESETS[p.preset].edge_type}</span>
                  <button onClick={() => setParents(parents.filter((x) => x.target !== p.target))}
                    className="ml-1 text-faint hover:text-danger">×</button>
                </span>
              ))}
            </div>
          )}
          <div className="flex items-center gap-1.5 flex-wrap">
            <span className="text-faint text-[10px]">similar to (optional)</span>
            <GenreAutocomplete value={similarDraft} onChange={setSimilarDraft} onPick={setSimilarDraft}
              placeholder="existing canonical genre" className={`${input} min-w-[180px]`}
              data-testid="wizard-similar-input" />
            <button data-testid="wizard-similar-add" disabled={!similarDraft.trim()}
              onClick={() => {
                const t = similarDraft.trim().toLowerCase();
                if (t && !similar.includes(t)) setSimilar([...similar, t]);
                setSimilarDraft("");
              }} className={btnGhost}>Add</button>
            {similar.map((s) => (
              <span key={s} className="text-[10px] px-1.5 py-0.5 rounded-full bg-panel2 text-muted">
                {s}<button onClick={() => setSimilar(similar.filter((x) => x !== s))}
                  className="ml-1 text-faint hover:text-danger">×</button>
              </span>
            ))}
          </div>
          <div className="text-faint text-[10px]">Genre = a recognized style; subgenre = a recognized style <em>within</em> a named parent.</div>
        </div>
      )}

      {step === 2 && isFacet && (
        <div className="flex items-center gap-1.5">
          <span className="text-faint text-[10px]">facet type</span>
          <select data-testid="wizard-facet-type" value={facetType}
            onChange={(e) => setFacetType(e.target.value)} className={input}>
            {FACET_TYPES.map((f) => <option key={f} value={f}>{f}</option>)}
          </select>
          <span className="text-faint text-[10px]">facets are modifiers — they have no parents.</span>
        </div>
      )}

      {step === 3 && (
        <div className="flex flex-col gap-1.5">
          <div className="flex items-center gap-1.5">
            <span className="text-faint text-[10px]">specificity</span>
            <input data-testid="wizard-specificity" type="number" min={0} max={1} step={0.01}
              value={specificity}
              onChange={(e) => { setSpecTouched(true); setSpecificity(Number(e.target.value)); }}
              className={`${input} w-[64px]`} />
            <span className="text-faint text-[10px]">
              ladder: genre 0.48–0.66 · subgenre 0.62–0.82
              {!isFacet && (specificity < spec.lo || specificity > spec.hi) ? " — outside the usual band" : ""}
            </span>
          </div>
          <div className="flex items-center gap-1.5 flex-wrap">
            <span className="text-faint text-[10px]">alias spellings (optional)</span>
            <input data-testid="wizard-alias-input" value={aliasDraft}
              onChange={(e) => setAliasDraft(e.target.value)} className={`${input} min-w-[160px]`} />
            <button data-testid="wizard-alias-add" disabled={!aliasDraft.trim()}
              onClick={() => {
                const a = aliasDraft.trim().toLowerCase();
                if (a && !aliases.includes(a)) setAliases([...aliases, a]);
                setAliasDraft("");
              }} className={btnGhost}>Add</button>
            {aliases.map((a) => (
              <span key={a} className="text-[10px] px-1.5 py-0.5 rounded-full bg-panel2 text-muted">
                {a}<button onClick={() => setAliases(aliases.filter((x) => x !== a))}
                  className="ml-1 text-faint hover:text-danger">×</button>
              </span>
            ))}
          </div>
        </div>
      )}

      {step === 4 && (
        <div className="flex flex-col gap-1.5">
          <div className="text-[10px] text-muted">
            <span className="text-accent font-semibold">add</span> · {name.trim().toLowerCase()} · {kind}
            {isFacet ? ` (${facetType})` : ""} · spec {specificity.toFixed(2)}
          </div>
          {!isFacet && parents.length > 0 && (
            <div className="flex flex-wrap items-center gap-1">
              <span className="text-faint text-[10px]">parents</span>
              {parents.map((p) => (
                <span key={p.target} className="text-[10px] px-1.5 py-0.5 rounded-full bg-panel2 text-text">
                  {p.target} <span className="text-faint">{EDGE_PRESETS[p.preset].edge_type} {EDGE_PRESETS[p.preset].weight}</span>
                </span>
              ))}
            </div>
          )}
          <input data-testid="wizard-rationale" value={rationale} placeholder="why this placement? (optional)"
            onChange={(e) => setRationale(e.target.value)} className={input} />
          {validating && <div className="text-faint text-[10px]">validating…</div>}
          {validateFailed !== null && (
            <div data-testid="wizard-validate-failed" className="text-danger text-[10px]">
              validation unavailable — {validateFailed}
            </div>
          )}
          {errors !== null && errors.length > 0 && (
            <div data-testid="wizard-errors" className="text-danger text-[10px]">
              {errors.map((e) => <div key={e}>{e}</div>)}
            </div>
          )}
          {errors !== null && errors.length === 0 && (
            <div className="text-accent text-[10px]">structurally valid ✓</div>
          )}
        </div>
      )}

      <div className="flex items-center gap-1.5 mt-0.5">
        {step > 1 && <button data-testid="wizard-back" onClick={() => goto(step - 1)} className={btnGhost}>Back</button>}
        {step < 4 && (
          <button data-testid="wizard-next" onClick={() => goto(step + 1)} disabled={nextDisabled} className={btnPrimary}>
            Next
          </button>
        )}
        {step === 4 && (
          <button data-testid="wizard-stage" onClick={() => onStage(buildProposal())} disabled={stageDisabled}
            className={btnPrimary}>Stage decision</button>
        )}
        <button data-testid="wizard-cancel" onClick={onCancel} className={btnGhost}>Cancel</button>
      </div>
    </div>
  );
}
