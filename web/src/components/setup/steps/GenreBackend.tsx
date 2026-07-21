import type { Dispatch, SetStateAction } from "react";
import type { WizardDraft } from "../useWizard";

interface GenreBackendProps {
  draft: WizardDraft;
  setDraft: Dispatch<SetStateAction<WizardDraft>>;
}

interface GenreOption {
  id: string;
  title: string;
  tradeoff: string;
}

const OPTIONS: GenreOption[] = [
  {
    id: "zero_touch",
    title: "Zero-touch (recommended)",
    tradeoff: "Free deterministic tagging from existing metadata — good coverage, no account needed.",
  },
  {
    id: "claude_code",
    title: "Claude Code",
    tradeoff: "Best genre quality — uses your logged-in Claude Code session, no extra billing.",
  },
  {
    id: "anthropic_api",
    title: "Anthropic API",
    tradeoff: "Best genre quality — needs an Anthropic API key; usage is billed per call.",
  },
  {
    id: "skip",
    title: "Skip",
    tradeoff: "No AI genre enrichment — fastest setup, but sparser and less consistent genre tags.",
  },
];

const DEFAULT_PROVIDER = "zero_touch";

// Step 5/7: pick which of the four AI genre-enrichment backends to use later
// (the actual enrichment run happens post-analyze, not here — this just
// records the choice). `zero_touch` is the sensible no-click default; it's
// seeded directly on the draft by useWizard's initial state (so it survives
// a rail-jump that skips this step entirely), not by a mount effect here.
// This component just reads/sets `draft.ai_genre_provider` -- purely
// controlled off the draft.
export default function GenreBackend({ draft, setDraft }: GenreBackendProps) {
  const selected = draft.ai_genre_provider ?? DEFAULT_PROVIDER;

  return (
    <div data-testid="step-genre" className="flex max-w-xl flex-col gap-3">
      <h2 className="text-lg font-semibold text-text">Genre source</h2>
      <p className="text-sm text-muted">Choose how MixArc tags genres for your library.</p>
      <div role="radiogroup" aria-label="Genre backend" className="flex flex-col gap-2">
        {OPTIONS.map((opt) => {
          const isSelected = selected === opt.id;
          return (
            <label
              key={opt.id}
              data-testid={`genre-option-${opt.id}`}
              className={[
                "flex min-h-11 cursor-pointer items-start gap-3 rounded border p-3 text-sm",
                isSelected ? "border-accent bg-panel2" : "border-border bg-panel hover:border-muted",
              ].join(" ")}
            >
              <input
                type="radio"
                name="ai_genre_provider"
                value={opt.id}
                checked={isSelected}
                onChange={() => setDraft((d) => ({ ...d, ai_genre_provider: opt.id }))}
                className="mt-1 h-4 w-4 accent-accent"
              />
              <span className="flex flex-col gap-0.5">
                <span className="font-medium text-text">{opt.title}</span>
                <span className="text-xs text-faint">{opt.tradeoff}</span>
              </span>
            </label>
          );
        })}
      </div>
    </div>
  );
}
