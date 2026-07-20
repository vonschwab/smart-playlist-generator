import type { SetupStatus } from "../lib/types";

// Minimal welcome/instructions gate shown when the app has no config.yaml yet.
// SP-3 replaces the body below with a guided wizard; this component (and the
// App.tsx gate that mounts it) is the permanent piece.
export default function SetupPage({ status }: { status: SetupStatus }) {
  return (
    <main data-testid="setup-page" className="mx-auto flex min-h-dvh max-w-xl flex-col justify-center gap-4 p-6">
      <h1 className="text-2xl font-semibold text-text">Welcome to MixArc</h1>
      <p className="text-muted">MixArc needs a one-time setup before it can build playlists.</p>
      <section className="rounded border border-border bg-panel p-4 space-y-2">
        <p className="font-medium text-text">{status.detail}</p>
        <ol className="list-decimal ml-5 space-y-1 text-sm text-muted">
          <li>Create <code className="font-mono text-text">{status.config_path}</code> (copy <code className="font-mono text-text">config.example.yaml</code> from the project).</li>
          <li>Set <code className="font-mono text-text">library.music_directory</code> to your music folder.</li>
          <li>Restart MixArc — analysis starts from the Tools tab.</li>
        </ol>
        <p className="text-sm opacity-70 text-faint">A guided setup wizard replaces this page in an upcoming release.</p>
      </section>
    </main>
  );
}
