import type { SetupStatus } from "../../../lib/types";

// Step 1/7: intro. Also carries the setup-gate's own status message
// (`status.detail`, e.g. "No config.yaml — run setup.") so it's visible the
// instant the wizard mounts — web/tests/setup-gate.spec.ts asserts on this
// text without navigating past Welcome.
export function Welcome({ status }: { status: SetupStatus }) {
  return (
    <div data-testid="step-welcome" className="flex max-w-xl flex-col gap-3">
      <h1 className="text-2xl font-semibold text-text">Welcome to MixArc</h1>
      <p className="text-muted">
        MixArc needs a one-time setup before it can build playlists: point it at your music
        library, optionally connect enrichment services, then run the first analysis.
      </p>
      {status.detail && (
        <section className="rounded border border-border bg-panel p-4">
          <p className="font-medium text-text">{status.detail}</p>
        </section>
      )}
      <p className="text-sm text-faint">Click Next to get started.</p>
    </div>
  );
}
