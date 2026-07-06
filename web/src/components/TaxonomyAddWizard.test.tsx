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
  // GenreAutocomplete's suggestion row commits the pick via onClick (verified
  // in GenreAutocomplete.tsx: `<div onClick={() => pick(s.name)}>`) — use
  // fireEvent.click, not mouseDown, or parentDraft never updates.
  fireEvent.click(suggestion);
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
    await waitFor(() => expect((stage as HTMLButtonElement).disabled).toBe(false));
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
    expect((screen.getByTestId("wizard-next") as HTMLButtonElement).disabled).toBe(true);
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
    await waitFor(() => expect((stage as HTMLButtonElement).disabled).toBe(false));
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
    expect((screen.getByTestId("wizard-stage") as HTMLButtonElement).disabled).toBe(true);
  });

  it("keeps Stage disabled when the validate endpoint fails", async () => {
    vi.mocked(api.taxonomyValidate).mockRejectedValueOnce(new Error("worker busy"));
    renderWizard();
    fireEvent.click(screen.getByTestId("wizard-next"));
    await addParent("indie rock");
    fireEvent.click(screen.getByTestId("wizard-next"));
    fireEvent.click(screen.getByTestId("wizard-next"));
    await waitFor(() => expect(screen.getByTestId("wizard-validate-failed").textContent).toContain("worker busy"));
    expect((screen.getByTestId("wizard-stage") as HTMLButtonElement).disabled).toBe(true);
  });
});
