import type { WizardDraft } from "../useWizard";

// Step 6/7: read-only summary of the accumulated draft, right before the
// analyze step submits it. Every field is optional in the draft, so each row
// falls back to a plain "not set"/"skipped" rather than rendering blank.
export function Review({ draft }: { draft: WizardDraft }) {
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
