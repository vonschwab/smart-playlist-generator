import type { SetupStatus } from "../../lib/types";
import { useWizard, type WizardStepId } from "./useWizard";
import { Welcome } from "./steps/Welcome";
import { EnvironmentCheck } from "./steps/EnvironmentCheck";
import { MusicFolder } from "./steps/MusicFolder";
import Services from "./steps/Services";
import GenreBackend from "./steps/GenreBackend";
import { Review } from "./steps/Review";
import { Analyze } from "./steps/Analyze";

// Human labels for the rail. Order/ids come from WIZARD_STEPS (useWizard.ts) —
// keep this map exhaustive over that list, not the other way around.
const STEP_LABELS: Record<WizardStepId, string> = {
  welcome: "Welcome",
  environment: "Environment",
  music: "Music library",
  services: "Services",
  genre: "Genre source",
  review: "Review",
  analyze: "Analyze",
};

// All seven steps are wired now (Task 8 shipped Analyze, the last one).
// This is kept as the switch's default arm so the rail and Back/Next still
// work end-to-end if WIZARD_STEPS ever grows a step ahead of its component.
function PlaceholderStep({ id }: { id: WizardStepId }) {
  return (
    <div data-testid={`step-${id}`} className="flex max-w-xl flex-col gap-2">
      <h2 className="text-lg font-semibold text-text">{STEP_LABELS[id]}</h2>
      <p className="text-sm text-muted">This step isn't built yet — coming in a later release.</p>
    </div>
  );
}

// SP-3 first-run wizard shell (Layout B): a left rail listing every step,
// the active step's component, and Back/Next navigation gated on `canNext`.
// Below ~640px the rail collapses to a horizontal progress strip (`max-md:`).
export default function SetupWizard({
  status,
  onSetupComplete,
}: {
  status: SetupStatus;
  // I2: forwarded to the Analyze step, called only on a SUCCESSFUL terminal
  // analyze job so App.tsx can re-fetch setup status without a reload.
  onSetupComplete?: () => void;
}) {
  const { step, steps, draft, setDraft, next, back, goTo, canNext } = useWizard();
  const currentIdx = steps.indexOf(step);

  let body: React.ReactNode;
  switch (step) {
    case "welcome":
      body = <Welcome status={status} />;
      break;
    case "environment":
      body = <EnvironmentCheck status={status} />;
      break;
    case "music":
      body = <MusicFolder draft={draft} setDraft={setDraft} />;
      break;
    case "services":
      body = <Services draft={draft} setDraft={setDraft} />;
      break;
    case "genre":
      body = <GenreBackend draft={draft} setDraft={setDraft} />;
      break;
    case "review":
      body = <Review draft={draft} goTo={goTo} />;
      break;
    case "analyze":
      body = <Analyze onSetupComplete={onSetupComplete} />;
      break;
    default:
      body = <PlaceholderStep id={step} />;
  }

  return (
    <div className="flex min-h-dvh flex-col bg-bg text-text md:flex-row">
      <nav
        data-testid="wizard-rail"
        aria-label="Setup steps"
        className="flex shrink-0 overflow-x-auto border-b border-border bg-panel max-md:flex-row md:w-56 md:flex-col md:overflow-visible md:border-b-0 md:border-r"
      >
        {steps.map((id, i) => {
          const isCurrent = id === step;
          const isDone = i < currentIdx;
          return (
            <button
              key={id}
              type="button"
              data-testid={`rail-step-${id}`}
              aria-current={isCurrent ? "step" : undefined}
              onClick={() => goTo(id)}
              className={[
                "flex min-h-11 shrink-0 items-center gap-2 whitespace-nowrap px-4 py-3 text-left text-sm transition-colors",
                isCurrent ? "bg-border text-text font-medium" : "text-faint hover:text-muted",
              ].join(" ")}
            >
              <span className={isDone ? "text-accent" : "text-faint"} aria-hidden="true">
                {isDone ? "✓" : i + 1}
              </span>
              {STEP_LABELS[id]}
            </button>
          );
        })}
      </nav>

      <div className="flex flex-1 flex-col overflow-auto">
        <div className="flex-1 p-6">{body}</div>
        <div className="flex items-center gap-2 border-t border-border bg-panel px-6 py-4">
          {currentIdx > 0 && (
            <button
              type="button"
              data-testid="wizard-back"
              onClick={back}
              className="rounded border border-border px-4 py-2 pointer-coarse:min-h-11 text-sm text-muted hover:text-text"
            >
              Back
            </button>
          )}
          <button
            type="button"
            data-testid="wizard-next"
            onClick={next}
            disabled={!canNext}
            className="ml-auto rounded bg-accent px-4 py-2 pointer-coarse:min-h-11 text-sm font-semibold text-bg disabled:opacity-50"
          >
            Next
          </button>
        </div>
      </div>
    </div>
  );
}
