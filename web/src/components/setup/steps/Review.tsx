import { useState } from "react";
import { api } from "../../../lib/api";
import type { ApiError } from "../../../lib/api";
import { friendlyError } from "../../../lib/errors";
import { btnGhost, btnPrimary } from "../../../lib/ui";
import type { SetupConfigDraft } from "../../../lib/types";
import type { WizardDraft, WizardStepId } from "../useWizard";

interface ReviewProps {
  draft: WizardDraft;
  goTo: (id: WizardStepId) => void;
}

// Step 6/7: read-only summary of the accumulated draft, right before the
// analyze step submits it. Every field is optional in the draft, so each row
// falls back to a plain "not set"/"skipped" rather than rendering blank.
//
// The confirm button (`wizard-write-config`) is where the draft actually
// leaves the browser: POST /api/setup/config via api.writeConfig. Two error
// shapes matter here:
//  - 409 (ConfigExistsError, app.py) — jsonOrThrow (lib/api.ts) attaches the
//    HTTP status to the thrown Error, so we branch on `err.status === 409`
//    and offer a reconfigure retry with `reconfigure: true`. Robust against
//    the server's exact wording, unlike matching the message text.
//  - anything else (validation, disk, network) — shown inline, draft is left
//    completely untouched so the user can fix a field and retry.
export function Review({ draft, goTo }: ReviewProps) {
  const [writing, setWriting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [conflict, setConflict] = useState(false);

  async function submit(reconfigure: boolean) {
    setWriting(true);
    setError(null);
    const body: SetupConfigDraft = {
      ...draft,
      music_directory: draft.music_directory ?? "",
      reconfigure,
    };
    try {
      await api.writeConfig(body);
      goTo("analyze");
    } catch (e) {
      if (!reconfigure && (e as ApiError)?.status === 409) {
        setConflict(true);
      } else {
        setConflict(false);
        setError(friendlyError(e));
      }
    } finally {
      setWriting(false);
    }
  }

  return (
    <div data-testid="step-review" className="flex max-w-xl flex-col gap-3">
      <h2 className="text-lg font-semibold text-text">Review</h2>
      <p className="text-sm text-muted">Check what MixArc will write to your config before analysis starts.</p>
      <dl className="rounded border border-border bg-panel p-4 text-sm">
        <Row label="Music folder" value={draft.music_directory ?? "not set"} />
        <Row label="Last.fm" value={draft.lastfm ? (draft.lastfm.username || "configured") : "skipped"} />
        <Row label="Discogs" value={draft.discogs ? "configured" : "skipped"} />
        <Row label="Plex" value={draft.plex ? "configured" : "skipped"} />
        <Row label="AI genre provider" value={draft.ai_genre_provider ?? "default"} />
      </dl>

      {conflict && (
        <div role="alert" className="flex flex-col gap-2 rounded border border-warn bg-panel p-3 text-sm">
          <p className="text-text">A config already exists at this location.</p>
          <button
            type="button"
            data-testid="wizard-reconfigure"
            disabled={writing}
            onClick={() => submit(true)}
            className={`${btnGhost} self-start`}
          >
            Overwrite it and continue
          </button>
        </div>
      )}

      {error && (
        <p role="alert" className="text-sm text-danger">
          {error}
        </p>
      )}

      <button
        type="button"
        data-testid="wizard-write-config"
        disabled={writing}
        onClick={() => submit(false)}
        className={`${btnPrimary} self-start`}
      >
        {writing ? "Writing…" : "Write config"}
      </button>
    </div>
  );
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between gap-4 border-b border-hairline py-1.5 last:border-b-0">
      <dt className="text-faint">{label}</dt>
      <dd className="text-text">{value}</dd>
    </div>
  );
}
