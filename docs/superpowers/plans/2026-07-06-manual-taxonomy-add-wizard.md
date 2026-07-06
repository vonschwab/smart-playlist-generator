# Manual Taxonomy ADD Wizard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A governed inline step-flow in the Taxonomy Review panel that lets a no-Claude user add a new genre/subgenre or facet to the canonical taxonomy — the manual equivalent of the "Ask Claude" ADD verdict — validated server-side by `graph_growth.validate_proposal` before it can be staged. Spec: `docs/superpowers/specs/2026-07-06-manual-taxonomy-add-wizard-design.md`.

**Architecture:** One new untracked worker command (`validate_taxonomy_proposal`) wrapping the existing `validate_proposal` rule engine, exposed via `POST /api/taxonomy/validate`. One new React component (`TaxonomyAddWizard`) that assembles the same `TaxonomyProposal` shape a Claude verdict produces and stages it through the **existing** `onDecide("add", proposal, null, true)` path — staging store and Apply pipeline untouched.

**Tech Stack:** Python (FastAPI + NDJSON worker), React + TypeScript + Tailwind (vitest + @testing-library/react), pytest.

## Global Constraints

- **Data safety:** tests NEVER touch `data/metadata.db`, `data/ai_genre_enrichment.db`, or the real `data/layered_genre_taxonomy.yaml`. The real taxonomy YAML may be **read** (via `load_default_layered_taxonomy()` — other tests already do); any write path runs against a tmp copy with `DEFAULT_TAXONOMY_PATH` monkeypatched.
- **Single rule authority:** every structural guardrail lives in `graph_growth.validate_proposal` (src/ai_genre_enrichment/graph_growth.py:389). The wizard's client-side gating is navigation convenience only — no structural rule is re-implemented in TypeScript.
- **Governed values (exact, from the spec):** edge presets `is_a` = weight 0.75 / confidence 0.85, `family_context` = weight 0.55 / confidence 0.80. Facet types verbatim: `mood, texture, instrumentation, production, era, region, function, vocal, scene, format, rhythm`. Specificity defaults: genre 0.55, subgenre 0.70, facet 0.50; ladder bands genre 0.48–0.66, subgenre 0.62–0.82. `status` fixed `"active"`. `term_kind_confirm` auto: `"facet"` for facets, else `"genre"`.
- **Rename governance:** if the final canonical name differs from the queue term, the original term is auto-added to `alias_variants`.
- **v1 scope:** kinds `genre | subgenre | facet` only. Ask Claude / Alias… / Reject flows, `taxonomy_decision_store`, and the Apply pipeline are NOT modified.
- **Shared checkout (concurrent sessions commit to master):** commits MUST use the form `git commit --only -m "..." -- <exact paths>` (a guard hook rejects bare commits). New files need `git add <path>` first. Verify `git diff --cached --name-only` shows only your files; leave all foreign in-flight files alone.
- **Test commands:** `python -m pytest -q <targets>` directly — never piped through head/tail. Web: `npm --prefix web run test` and `npm --prefix web run build` from the repo root.

---

### Task 1: Worker command `validate_taxonomy_proposal`

**Files:**
- Modify: `src/playlist_gui/worker.py` (new handler after `handle_record_taxonomy_decision`, ~line 2919; registration in `UNTRACKED_COMMAND_HANDLERS`, ~line 3120)
- Test: `tests/unit/test_worker_taxonomy_validate.py` (new)

**Interfaces:**
- Consumes: `graph_growth.GrowthProposal` (dataclass: `name, kind, status, specificity_score, parent_edges, similar_to, alias_variants, term_kind_confirm, rationale, facet_type, canonical_target`), `graph_growth.validate_proposal(taxonomy, proposal) -> list[str]`, `layered_taxonomy.load_default_layered_taxonomy()`.
- Produces: untracked command `{"cmd": "validate_taxonomy_proposal", "proposal": {…TaxonomyProposal dict…}}` → result event `{"type": "result", "result_type": "taxonomy_validate", "errors": [...]}`. Task 2's route submits this command.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_worker_taxonomy_validate.py` (pattern: `tests/unit/test_worker_review_queue.py` — call the handler directly, parse the NDJSON events from capsys). The real taxonomy is loaded read-only; names chosen to be stable: `indie rock` (canonical parent), `shoegaze` (existing canonical → duplicate), `xyzzy-*` nonsense (guaranteed-new names, per the taxonomy-growth skill's fixture convention).

```python
# tests/unit/test_worker_taxonomy_validate.py
"""Worker round-trip for the manual ADD-wizard's validation command."""
import json

from src.playlist_gui.worker import handle_validate_taxonomy_proposal


def _events(capsys):
    return [json.loads(line) for line in capsys.readouterr().out.strip().splitlines()]


def _validate(capsys, proposal):
    handle_validate_taxonomy_proposal({
        "cmd": "validate_taxonomy_proposal", "request_id": "r1", "proposal": proposal})
    events = _events(capsys)
    result = next(e for e in events if e["type"] == "result")
    done = next(e for e in events if e["type"] == "done")
    assert result["result_type"] == "taxonomy_validate"
    return result["errors"], done


def _genre(**over):
    p = {
        "name": "xyzzy wizard genre", "kind": "genre", "status": "active",
        "specificity_score": 0.55,
        "parent_edges": [{"target": "indie rock", "edge_type": "is_a",
                          "weight": 0.75, "confidence": 0.85}],
        "similar_to": [], "alias_variants": [], "term_kind_confirm": "genre",
        "facet_type": None, "canonical_target": None, "rationale": "",
    }
    p.update(over)
    return p


def test_valid_genre_proposal_returns_no_errors(capsys):
    errors, done = _validate(capsys, _genre())
    assert errors == []
    assert done["ok"] is True


def test_leaf_without_parent_is_rejected(capsys):
    errors, _ = _validate(capsys, _genre(parent_edges=[]))
    assert any("parent edge" in e for e in errors)


def test_nonexistent_parent_target_is_rejected(capsys):
    errors, _ = _validate(capsys, _genre(
        parent_edges=[{"target": "xyzzy nonexistent parent", "edge_type": "is_a",
                       "weight": 0.75, "confidence": 0.85}]))
    assert any("does not exist" in e for e in errors)


def test_duplicate_name_is_rejected(capsys):
    errors, _ = _validate(capsys, _genre(name="shoegaze"))
    assert any("already exists" in e for e in errors)


def test_facet_with_parent_edges_is_rejected(capsys):
    errors, _ = _validate(capsys, _genre(
        kind="facet", term_kind_confirm="facet", facet_type="instrumentation"))
    assert any("Facet proposals cannot have parent_edges" in e for e in errors)


def test_valid_facet_proposal_returns_no_errors(capsys):
    errors, done = _validate(capsys, _genre(
        name="xyzzy wizard facet", kind="facet", term_kind_confirm="facet",
        facet_type="instrumentation", parent_edges=[]))
    assert errors == []
    assert done["ok"] is True


def test_malformed_proposal_reports_error_not_crash(capsys):
    handle_validate_taxonomy_proposal({
        "cmd": "validate_taxonomy_proposal", "request_id": "r1",
        "proposal": {"specificity_score": "not-a-number"}})
    events = _events(capsys)
    done = next(e for e in events if e["type"] == "done")
    assert done["ok"] is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest -q tests/unit/test_worker_taxonomy_validate.py`
Expected: FAIL at import — `ImportError: cannot import name 'handle_validate_taxonomy_proposal'`.

- [ ] **Step 3: Implement the handler**

In `src/playlist_gui/worker.py`, directly after `handle_record_taxonomy_decision` (before `_next_taxonomy_gui_version`, ~line 2920):

```python
def handle_validate_taxonomy_proposal(cmd_data: Dict[str, Any]) -> None:
    """Structural pre-flight for a manual ADD-wizard proposal. UNTRACKED +
    read-only (reader thread): wraps graph_growth.validate_proposal — the
    single guardrail authority — so the GUI never re-implements placement
    rules. Emits {"errors": [...]}; empty list means safe to stage."""
    rid = cmd_data.get("request_id")
    try:
        from src.ai_genre_enrichment.graph_growth import GrowthProposal, validate_proposal
        from src.ai_genre_enrichment.layered_taxonomy import load_default_layered_taxonomy

        pj = cmd_data.get("proposal")
        pj = pj if isinstance(pj, dict) else {}
        gp_fields = set(GrowthProposal.__dataclass_fields__)  # type: ignore[attr-defined]
        required = {"name", "kind", "status", "specificity_score"}
        proposal = GrowthProposal(
            name=str(pj.get("name") or ""),
            kind=str(pj.get("kind") or ""),
            status=str(pj.get("status") or "active"),
            specificity_score=float(pj.get("specificity_score") or 0.0),
            **{k: v for k, v in pj.items() if k in gp_fields - required},
        )
        errors = validate_proposal(load_default_layered_taxonomy(), proposal)
        emit_event({"type": "result", "result_type": "taxonomy_validate",
                    "errors": errors, "request_id": rid, "job_id": None})
        emit_event({"type": "done", "cmd": "validate_taxonomy_proposal", "ok": True,
                    "detail": f"{proposal.name}: {len(errors)} error(s)",
                    "request_id": rid, "job_id": None})
    except Exception as e:
        emit_event({"type": "error", "message": str(e), "request_id": rid, "job_id": None})
        emit_event({"type": "done", "cmd": "validate_taxonomy_proposal", "ok": False,
                    "detail": str(e), "request_id": rid, "job_id": None})
```

Register it in `UNTRACKED_COMMAND_HANDLERS` (~line 3120), after `"record_taxonomy_decision"`:

```python
    "record_taxonomy_decision": handle_record_taxonomy_decision,
    "validate_taxonomy_proposal": handle_validate_taxonomy_proposal,
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest -q tests/unit/test_worker_taxonomy_validate.py`
Expected: 7 passed. Then adjacent sanity: `python -m pytest -q tests/unit/test_worker_review_queue.py tests/unit/test_taxonomy_review_queue.py` — expected: pass (no regression).

- [ ] **Step 5: Commit**

```bash
git add tests/unit/test_worker_taxonomy_validate.py
git commit --only -m "feat(taxonomy): untracked validate_taxonomy_proposal worker command (ADD-wizard pre-flight)" -- src/playlist_gui/worker.py tests/unit/test_worker_taxonomy_validate.py
```

---

### Task 2: Web route + schema + API client method

**Files:**
- Modify: `src/playlist_web/schemas.py` (~line 316, after `TaxonomyDecisionRequest`)
- Modify: `src/playlist_web/app.py` (schema import ~line 34; new route after `taxonomy_decision`, ~line 379)
- Modify: `web/src/lib/api.ts` (~line 171, after `taxonomyDecision`; add `TaxonomyProposal` to the type imports at the top of the file)

**Interfaces:**
- Consumes: Task 1's `validate_taxonomy_proposal` command.
- Produces: `POST /api/taxonomy/validate` with body `{"proposal": {...}}` → `{..., "errors": string[]}`; client method `api.taxonomyValidate(proposal: TaxonomyProposal): Promise<{ errors: string[] }>`. Task 3's wizard calls this.

- [ ] **Step 1: Add the request model** (`schemas.py`, after `TaxonomyDecisionRequest`):

```python
class TaxonomyValidateRequest(BaseModel):
    """Structural pre-flight for a manual ADD-wizard proposal (no staging)."""

    proposal: dict
```

- [ ] **Step 2: Add the route** (`app.py`). Add `TaxonomyValidateRequest` to the existing `schemas` import block (alongside `TaxonomyDecisionRequest`), then after the `taxonomy_decision` route:

```python
    @app.post("/api/taxonomy/validate")
    async def taxonomy_validate(body: TaxonomyValidateRequest) -> dict:
        # Untracked: validate_proposal is a fast read-only check against the
        # live taxonomy; the ADD wizard blocks Stage until errors == [].
        try:
            return await bridge.command({
                "cmd": "validate_taxonomy_proposal",
                "proposal": body.proposal}, untracked=True)
        except BridgeBusy:
            raise HTTPException(status_code=409, detail="Worker is busy — try again when the current job finishes.")
        except WorkerCommandError as exc:
            raise HTTPException(status_code=502, detail=str(exc))
```

- [ ] **Step 3: Add the client method** (`web/src/lib/api.ts`, after `taxonomyDecision`; add `TaxonomyProposal` to the `types` import):

```typescript
  async taxonomyValidate(proposal: TaxonomyProposal): Promise<{ errors: string[] }> {
    return jsonOrThrow(await fetch("/api/taxonomy/validate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ proposal }),
    }));
  },
```

- [ ] **Step 4: Verify wiring**

Run: `python -c "from src.playlist_web.schemas import TaxonomyValidateRequest; from src.playlist_web.app import create_app; print('ok')"`
Expected: `ok` (route registration is exercised at app construction). Then `python -m pytest -q tests/integration/test_web_review_api.py` — expected: pass (app still builds under the test harness). TypeScript checked by Task 4's build.

- [ ] **Step 5: Commit**

```bash
git commit --only -m "feat(taxonomy): POST /api/taxonomy/validate + api.taxonomyValidate (ADD-wizard pre-flight)" -- src/playlist_web/schemas.py src/playlist_web/app.py web/src/lib/api.ts
```

---

### Task 3: `TaxonomyAddWizard` component (TDD)

**Files:**
- Create: `web/src/components/TaxonomyAddWizard.tsx`
- Test: `web/src/components/TaxonomyAddWizard.test.tsx` (new; pattern: `GenerateControls.test.tsx` — vitest + @testing-library/react)

**Interfaces:**
- Consumes: `api.taxonomyValidate` (Task 2), `GenreAutocomplete` (props: `value, onChange, onPick?, placeholder?, className?, autoFocus?, limit?`; suggestions fetched via `api.genresSearch` with 150ms debounce), types `TaxonomyProposal`, `TaxonomyParentEdge`, `TaxonomyQueueItem`.
- Produces: `<TaxonomyAddWizard item={TaxonomyQueueItem} onStage={(p: TaxonomyProposal) => void} onCancel={() => void} />`. Task 4 mounts it in `TermCard`.

- [ ] **Step 1: Write the failing tests**

```tsx
// web/src/components/TaxonomyAddWizard.test.tsx
import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen, fireEvent, cleanup, waitFor } from "@testing-library/react";

vi.mock("../lib/api", () => ({
  api: {
    genresSearch: vi.fn(async () => ({ items: [{ genre_id: "g1", name: "indie rock" }] })),
    taxonomyValidate: vi.fn(async () => ({ errors: [] })),
  },
}));
import { api } from "../lib/api";
import { TaxonomyAddWizard } from "./TaxonomyAddWizard";
import type { TaxonomyQueueItem } from "../lib/types";

afterEach(() => { cleanup(); vi.clearAllMocks(); });

const ITEM: TaxonomyQueueItem = {
  term: "shoe gaze", raw_term: "Shoe Gaze", album_frequency: 3,
  cooccurring_tags: [], examples: [], variants: [], source: "", decision: null,
};

function renderWizard(onStage = vi.fn(), onCancel = vi.fn()) {
  render(<TaxonomyAddWizard item={ITEM} onStage={onStage} onCancel={onCancel} />);
  return { onStage, onCancel };
}

async function addParent(name: string) {
  fireEvent.change(screen.getByTestId("wizard-parent-input"), { target: { value: name.slice(0, 5) } });
  const suggestion = await waitFor(() => screen.getByText(name));
  // GenreAutocomplete commits picks on the suggestion's pointer handler —
  // if fireEvent.click doesn't register, use fireEvent.mouseDown (check the
  // component's suggestion-row handler).
  fireEvent.mouseDown(suggestion);
  fireEvent.click(screen.getByTestId("wizard-parent-add"));
}

describe("TaxonomyAddWizard genre path", () => {
  it("assembles a governed proposal, auto-aliases the renamed term, and stages it", async () => {
    const { onStage } = renderWizard();
    // Step 1: rename to the clean canonical form
    fireEvent.change(screen.getByTestId("wizard-name"), { target: { value: "shoegazer" } });
    fireEvent.click(screen.getByTestId("wizard-next"));
    // Step 2: one strong parent
    await addParent("indie rock");
    fireEvent.click(screen.getByTestId("wizard-next"));
    // Step 3: keep defaults
    fireEvent.click(screen.getByTestId("wizard-next"));
    // Step 4: validate runs automatically; Stage enables on empty errors
    await waitFor(() => expect(api.taxonomyValidate).toHaveBeenCalled());
    const stage = await waitFor(() => screen.getByTestId("wizard-stage"));
    await waitFor(() => expect(stage).not.toBeDisabled());
    fireEvent.click(stage);
    const proposal = onStage.mock.calls[0][0];
    expect(proposal.name).toBe("shoegazer");
    expect(proposal.kind).toBe("genre");
    expect(proposal.status).toBe("active");
    expect(proposal.term_kind_confirm).toBe("genre");
    expect(proposal.specificity_score).toBeCloseTo(0.55);
    expect(proposal.parent_edges).toEqual([
      { target: "indie rock", edge_type: "is_a", weight: 0.75, confidence: 0.85 },
    ]);
    expect(proposal.alias_variants).toContain("shoe gaze"); // rename governance
  });

  it("blocks advancing past placement without a parent", () => {
    renderWizard();
    fireEvent.click(screen.getByTestId("wizard-next")); // -> step 2
    expect(screen.getByTestId("wizard-next")).toBeDisabled();
  });
});

describe("TaxonomyAddWizard facet path", () => {
  it("emits facet_type, no parents, term_kind_confirm=facet", async () => {
    const { onStage } = renderWizard();
    fireEvent.click(screen.getByTestId("wizard-kind-facet"));
    fireEvent.click(screen.getByTestId("wizard-next"));
    // Step 2 (facet): no parent picker rendered
    expect(screen.queryByTestId("wizard-parent-input")).toBeNull();
    fireEvent.change(screen.getByTestId("wizard-facet-type"), { target: { value: "instrumentation" } });
    fireEvent.click(screen.getByTestId("wizard-next"));
    fireEvent.click(screen.getByTestId("wizard-next"));
    const stage = await waitFor(() => screen.getByTestId("wizard-stage"));
    await waitFor(() => expect(stage).not.toBeDisabled());
    fireEvent.click(stage);
    const proposal = onStage.mock.calls[0][0];
    expect(proposal.kind).toBe("facet");
    expect(proposal.facet_type).toBe("instrumentation");
    expect(proposal.parent_edges).toEqual([]);
    expect(proposal.term_kind_confirm).toBe("facet");
  });
});

describe("TaxonomyAddWizard validation gate", () => {
  it("shows server errors and keeps Stage disabled", async () => {
    vi.mocked(api.taxonomyValidate).mockResolvedValueOnce({
      errors: ["A taxonomy record named/sluged like 'shoe gaze' already exists."] });
    renderWizard();
    fireEvent.click(screen.getByTestId("wizard-next"));
    await addParent("indie rock");
    fireEvent.click(screen.getByTestId("wizard-next"));
    fireEvent.click(screen.getByTestId("wizard-next"));
    await waitFor(() => expect(screen.getByTestId("wizard-errors").textContent).toContain("already exists"));
    expect(screen.getByTestId("wizard-stage")).toBeDisabled();
  });

  it("keeps Stage disabled when the validate endpoint fails", async () => {
    vi.mocked(api.taxonomyValidate).mockRejectedValueOnce(new Error("worker busy"));
    renderWizard();
    fireEvent.click(screen.getByTestId("wizard-next"));
    await addParent("indie rock");
    fireEvent.click(screen.getByTestId("wizard-next"));
    fireEvent.click(screen.getByTestId("wizard-next"));
    await waitFor(() => expect(screen.getByTestId("wizard-validate-failed").textContent).toContain("worker busy"));
    expect(screen.getByTestId("wizard-stage")).toBeDisabled();
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `npm --prefix web run test -- TaxonomyAddWizard`
Expected: FAIL — cannot resolve `./TaxonomyAddWizard`.

- [ ] **Step 3: Implement the component**

```tsx
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

// Specificity ladder (taxonomy-growth skill governance).
const SPECIFICITY: Record<Kind, { def: number; lo: number; hi: number }> = {
  genre: { def: 0.55, lo: 0.48, hi: 0.66 },
  subgenre: { def: 0.7, lo: 0.62, hi: 0.82 },
  facet: { def: 0.5, lo: 0, hi: 1 },
};

type Kind = "genre" | "subgenre" | "facet";
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
              className={`${input} min-w-[200px]`} autoFocus />
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
              placeholder="existing canonical genre" className={`${input} min-w-[180px]`} />
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
```

Note: if TypeScript complains that `SPECIFICITY` references `Kind` before declaration, move the `type Kind` line above it.

- [ ] **Step 4: Run tests to verify they pass**

Run: `npm --prefix web run test -- TaxonomyAddWizard`
Expected: 6 passed. If the autocomplete pick doesn't register, check `GenreAutocomplete.tsx`'s suggestion-row event (click vs mouseDown) and match it in `addParent`.

- [ ] **Step 5: Commit**

```bash
git add web/src/components/TaxonomyAddWizard.tsx web/src/components/TaxonomyAddWizard.test.tsx
git commit --only -m "feat(taxonomy): TaxonomyAddWizard - governed manual ADD step flow" -- web/src/components/TaxonomyAddWizard.tsx web/src/components/TaxonomyAddWizard.test.tsx
```

---

### Task 4: Mount the wizard in `TermCard` + build

**Files:**
- Modify: `web/src/components/TaxonomyReviewPanel.tsx` (TermCard: import, one state flag, one button, one render branch)

**Interfaces:**
- Consumes: Task 3's `TaxonomyAddWizard`; TermCard's existing `onDecide(verdict, proposal, claude, humanEdited)`.
- Produces: the "Add manually…" button in the untriaged term card's initial action row; staging via `onDecide("add", proposal, null, true)` (identical shape to the manual Alias path).

- [ ] **Step 1: Wire it in.** In `TaxonomyReviewPanel.tsx`:

Add the import next to `GenreAutocomplete`:

```tsx
import { TaxonomyAddWizard } from "./TaxonomyAddWizard";
```

In `TermCard`, add state beside the existing flags (`aliasing`, `rejecting`):

```tsx
  const [adding, setAdding] = useState(false);
```

In the `!verdict` branch, extend the ternary — `adding` takes precedence, and the existing `aliasing ? … : (…)` chain stays untouched inside the false arm:

```tsx
      {!verdict ? (
        adding ? (
          <TaxonomyAddWizard
            item={item}
            onStage={(p) => onDecide("add", p, null, true)}
            onCancel={() => setAdding(false)}
          />
        ) : aliasing ? (
          /* existing alias block, unchanged */
```

And add the button in the initial action row, after the `Alias…` button:

```tsx
            <button onClick={() => setAdding(true)}
              className="text-[10px] px-2 py-0.5 rounded border border-border text-muted hover:text-text">
              Add manually…
            </button>
```

- [ ] **Step 2: Full web test suite + type-check + build**

Run: `npm --prefix web run test`
Expected: all pass (existing suites + the 6 wizard tests).
Run: `npm --prefix web run build`
Expected: `tsc -b` clean, vite build succeeds. (Also refreshes `web/dist` — the stale-dist trap: the live GUI serves `dist`, so the build is required before any click-through.)

- [ ] **Step 3: Commit**

```bash
git commit --only -m "feat(taxonomy): Add manually... button mounts the ADD wizard in the term card" -- web/src/components/TaxonomyReviewPanel.tsx
```

---

### Task 5: Backend E2E — wizard-shaped decision applies to an isolated taxonomy

**Files:**
- Test: `tests/unit/test_taxonomy_wizard_apply_e2e.py` (new)

**Interfaces:**
- Consumes: `handle_record_taxonomy_decision` + `handle_apply_taxonomy_decisions` (worker.py:2885 / :2930). The apply handler imports `DEFAULT_TAXONOMY_PATH` from `src.ai_genre_enrichment.layered_taxonomy` *inside the function*, so monkeypatching that module attribute isolates the write; `SIDECAR_DB_PATH` monkeypatched to a tmp sidecar as in `test_worker_review_queue.py`.
- Produces: proof that a proposal exactly as the wizard assembles it lands in the YAML (record + alias variants + version bump) through the untouched apply pipeline.

- [ ] **Step 1: Write the test**

```python
# tests/unit/test_taxonomy_wizard_apply_e2e.py
"""A wizard-shaped ADD decision flows through record -> apply into the YAML.

The real taxonomy is copied to tmp and DEFAULT_TAXONOMY_PATH monkeypatched —
the canonical data/layered_genre_taxonomy.yaml is never written.
"""
import json
import shutil

import yaml

import src.ai_genre_enrichment.layered_taxonomy as lt
from src.playlist_gui.worker import (
    handle_apply_taxonomy_decisions,
    handle_record_taxonomy_decision,
)

WIZARD_PROPOSAL = {  # exactly what TaxonomyAddWizard.buildProposal() stages
    "name": "xyzzy wizard genre", "kind": "genre", "status": "active",
    "specificity_score": 0.55,
    "parent_edges": [{"target": "indie rock", "edge_type": "is_a",
                      "weight": 0.75, "confidence": 0.85}],
    "similar_to": [], "alias_variants": ["xyzzy wizard variant"],
    "term_kind_confirm": "genre", "facet_type": None, "canonical_target": None,
    "rationale": "e2e",
}


def test_wizard_add_lands_in_isolated_taxonomy(tmp_path, monkeypatch, capsys):
    tmp_yaml = tmp_path / "taxonomy.yaml"
    shutil.copyfile(lt.DEFAULT_TAXONOMY_PATH, tmp_yaml)
    original_version = (yaml.safe_load(tmp_yaml.read_text(encoding="utf-8")) or {}).get("taxonomy_version")
    monkeypatch.setattr(lt, "DEFAULT_TAXONOMY_PATH", str(tmp_yaml))
    monkeypatch.setattr("src.playlist_gui.worker.SIDECAR_DB_PATH", str(tmp_path / "sidecar.db"))

    handle_record_taxonomy_decision({
        "cmd": "record_taxonomy_decision", "request_id": "r1",
        "term": "xyzzy wizard genre", "raw_term": "Xyzzy Wizard Genre",
        "verdict": "add", "proposal": WIZARD_PROPOSAL, "claude": None,
        "human_edited": True})
    handle_apply_taxonomy_decisions({
        "cmd": "apply_taxonomy_decisions", "request_id": "r2", "job_id": "j1"})

    events = [json.loads(line) for line in capsys.readouterr().out.strip().splitlines()]
    dones = [e for e in events if e["type"] == "done" and e.get("cmd") == "apply_taxonomy_decisions"]
    assert dones and dones[-1]["ok"] is True

    data = yaml.safe_load(tmp_yaml.read_text(encoding="utf-8"))
    assert data.get("taxonomy_version") != original_version
    records = data.get("records") or data.get("genres") or []
    names = {str(r.get("name", "")).lower(): r for r in records if isinstance(r, dict)}
    rec = names.get("xyzzy wizard genre")
    assert rec is not None, "wizard record not appended"
    edge_targets = {(e.get("target") or "").lower() for e in (rec.get("parent_edges") or [])}
    assert "indie rock" in edge_targets
    alias = names.get("xyzzy wizard variant")
    assert alias is not None and alias.get("kind") == "alias"
```

Note: the YAML's top-level record-list key must be confirmed against the file (`records` vs another name) — adjust the `records =` line to the actual key after one look at `data/layered_genre_taxonomy.yaml`'s top-level structure (read-only). If `handle_apply_taxonomy_decisions` trips on job/cancel state when called directly, mirror the tracked-handler test setup used in `tests/unit/test_worker_escalation_review.py`.

- [ ] **Step 2: Run it**

Run: `python -m pytest -q tests/unit/test_taxonomy_wizard_apply_e2e.py`
Expected: 1 passed (this is a characterization test of existing plumbing — it should pass without production changes; if it fails, the failure is information about the wiring, not something to patch around).

- [ ] **Step 3: Full fast suite**

Run: `python -m pytest -q -m "not slow"` (generous timeout, never piped)
Expected: green apart from any documented pre-existing failures; quote real counts and triage anything new.

- [ ] **Step 4: Commit**

```bash
git add tests/unit/test_taxonomy_wizard_apply_e2e.py
git commit --only -m "test(taxonomy): e2e - wizard-shaped ADD decision applies to an isolated taxonomy" -- tests/unit/test_taxonomy_wizard_apply_e2e.py
```

- [ ] **Step 5: Live verification (Dylan)**

Restart `python tools/serve_web.py` (worker restart trap — the running worker predates the new command; `web/dist` was rebuilt in Task 4). In the GUI Taxonomy tab: open an untriaged term → **Add manually…** → walk the four steps (try a facet and a genre; try a duplicate name to see the validation error) → Stage → **do not click Apply** unless a real addition is actually wanted (Apply writes the real, git-tracked taxonomy YAML with its normal backup + version bump).

---

## Self-review notes (already applied)

- **Spec coverage:** kind fork + instrument-led hint (T3 step 1), placement with two governed presets + similar_to (T3 step 2), facet path without parents (T3), specificity ladder + aliases (T3 step 3), rename auto-alias (T3 buildProposal + test), review/validate/stage gate incl. endpoint-failure lockout (T3 step 4 + tests), untracked validate command single-sourcing validate_proposal (T1), route/schema/client (T2), TermCard mount (T4), E2E through the untouched apply pipeline + live verify (T5). Phase-2 suggestions and umbrella/microgenre deliberately absent.
- **Known judgment calls for implementers:** the autocomplete pick event (click vs mouseDown) and the YAML top-level key in T5 are flagged inline with exactly where to look — both are one-line adaptations, not design decisions.
- **Type consistency:** `TaxonomyProposal` fields used in T3 match `web/src/lib/types.ts:224` and the `GrowthProposal` fields consumed in T1/T5 match `graph_growth.py:189`.
