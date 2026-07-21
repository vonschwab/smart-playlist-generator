import type { Dispatch, SetStateAction } from "react";
import FolderBrowser from "../FolderBrowser";
import type { WizardDraft } from "../useWizard";

interface MusicFolderProps {
  draft: WizardDraft;
  setDraft: Dispatch<SetStateAction<WizardDraft>>;
}

// Step 3/7: the one hard-required step (useWizard's computeCanNext gates
// Next on `draft.music_directory`). Wraps FolderBrowser, a server-side folder
// picker, and records the chosen path into the draft.
export function MusicFolder({ draft, setDraft }: MusicFolderProps) {
  return (
    <div data-testid="step-music" className="flex max-w-xl flex-col gap-3">
      <h2 className="text-lg font-semibold text-text">Music library</h2>
      <p className="text-sm text-muted">
        Browse to the folder that holds your music library, then click "Use this folder".
      </p>
      {draft.music_directory && (
        <section className="rounded border border-border bg-panel p-3 text-sm">
          <span className="text-faint">Selected: </span>
          <span className="break-all text-text">{draft.music_directory}</span>
        </section>
      )}
      <FolderBrowser onChoose={(p) => setDraft((d) => ({ ...d, music_directory: p }))} />
    </div>
  );
}
