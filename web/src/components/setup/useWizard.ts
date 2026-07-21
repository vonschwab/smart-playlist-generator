import { useCallback, useState, type Dispatch, type SetStateAction } from "react";
import type { SetupConfigDraft } from "../../lib/types";

// Ordered step ids for the SP-3 first-run wizard (Layout B — left rail, all
// steps visible). Later tasks (6-8) fill in music/services/genre/analyze;
// this list is the contract they key off of — don't reorder without checking
// every step component's `data-testid="step-<id>"`.
export const WIZARD_STEPS = [
  "welcome",
  "environment",
  "music",
  "services",
  "genre",
  "review",
  "analyze",
] as const;

export type WizardStepId = (typeof WIZARD_STEPS)[number];

// Client-side accumulation of the wizard's answers. Mirrors SetupConfigDraft
// (the POST /api/setup/config body) minus `reconfigure` (decided at submit
// time, not accumulated) and with `music_directory` optional (unset until the
// music step is completed).
export type WizardDraft = Omit<SetupConfigDraft, "music_directory" | "reconfigure"> & {
  music_directory?: string;
};

export interface UseWizardResult {
  step: WizardStepId;
  steps: readonly WizardStepId[];
  draft: WizardDraft;
  setDraft: Dispatch<SetStateAction<WizardDraft>>;
  next: () => void;
  back: () => void;
  goTo: (id: WizardStepId) => void;
  canNext: boolean;
}

// Per-step "may I advance" gate. Only the music step has a hard requirement
// (a chosen library folder) — every other step is skippable/optional, per the
// SP-3 design (services/genre are opt-in enrichment, review is just a summary).
function computeCanNext(step: WizardStepId, draft: WizardDraft): boolean {
  if (step === "music") return Boolean(draft.music_directory);
  return true;
}

// Public/stranger-safe default: seeded on the draft itself (not as a
// mount-effect of the Genre step) so it survives a rail-jump that skips that
// step entirely -- e.g. welcome -> ... -> review straight from the left rail,
// which has no order gate. `claude_code` (the config_loader fallback for an
// *unset* provider) needs an auth session a public user won't have, so the
// draft must never reach submit with this field empty.
const DEFAULT_AI_GENRE_PROVIDER = "zero_touch";

export function useWizard(): UseWizardResult {
  const [step, setStep] = useState<WizardStepId>(WIZARD_STEPS[0]);
  const [draft, setDraft] = useState<WizardDraft>({ ai_genre_provider: DEFAULT_AI_GENRE_PROVIDER });

  const canNext = computeCanNext(step, draft);

  const goTo = useCallback((id: WizardStepId) => setStep(id), []);

  const next = useCallback(() => {
    setStep((cur) => {
      if (!computeCanNext(cur, draft)) return cur;
      const i = WIZARD_STEPS.indexOf(cur);
      return WIZARD_STEPS[Math.min(i + 1, WIZARD_STEPS.length - 1)];
    });
  }, [draft]);

  const back = useCallback(() => {
    setStep((cur) => {
      const i = WIZARD_STEPS.indexOf(cur);
      return WIZARD_STEPS[Math.max(i - 1, 0)];
    });
  }, []);

  return { step, steps: WIZARD_STEPS, draft, setDraft, next, back, goTo, canNext };
}
