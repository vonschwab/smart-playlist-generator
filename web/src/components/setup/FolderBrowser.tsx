import { useCallback, useEffect, useState } from "react";
import { api } from "../../lib/api";
import type { BrowseResponse } from "../../lib/types";

interface FolderBrowserProps {
  onChoose: (path: string) => void;
}

// Server-side directory browser for the SP-3 music-folder step. The server
// (src/setup/browse.py, GET /api/setup/browse) walks the FS on the machine
// running MixArc — not the browser's FS — since this is a local-first desktop
// app; there's no <input type="file" webkitdirectory> substitute for "which
// folder holds my library" once you're navigating server-side paths.
export default function FolderBrowser({ onChoose }: FolderBrowserProps) {
  const [data, setData] = useState<BrowseResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  // Last path we successfully rendered, so "Retry" re-issues the same
  // request instead of bouncing back to the home dir on a transient failure.
  const [lastPath, setLastPath] = useState<string | undefined>(undefined);

  const load = useCallback((path?: string) => {
    setLoading(true);
    setError(null);
    api
      .browseDir(path)
      .then((resp) => {
        setData(resp);
        setLastPath(resp.path);
        setLoading(false);
      })
      .catch((err: unknown) => {
        setError(err instanceof Error ? err.message : "Couldn't browse that folder.");
        setLoading(false);
      });
  }, []);

  useEffect(() => {
    load();
    // Mount-only: subsequent navigation calls `load` directly from handlers.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div className="flex flex-col gap-3">
      <div className="min-h-5 truncate font-mono text-xs text-faint" data-testid="folder-breadcrumb">
        {data ? data.path : " "}
      </div>

      {loading && (
        <p role="status" className="text-sm text-faint">
          Loading folder…
        </p>
      )}

      {error && (
        <div role="alert" className="rounded border border-danger/40 bg-panel p-3 text-sm text-danger">
          <p>{error}</p>
          <div className="mt-2 flex gap-2">
            <button
              type="button"
              data-testid="folder-retry"
              onClick={() => load(lastPath)}
              className="min-h-11 rounded border border-border px-3 text-xs text-muted hover:text-text"
            >
              Retry
            </button>
            <button
              type="button"
              data-testid="folder-home"
              onClick={() => load()}
              className="min-h-11 rounded border border-border px-3 text-xs text-muted hover:text-text"
            >
              Go to home folder
            </button>
          </div>
        </div>
      )}

      {data && !loading && (
        <>
          <div className="flex max-h-72 flex-col overflow-auto rounded border border-border bg-well">
            {data.parent !== null && (
              <button
                type="button"
                data-testid="folder-up"
                onClick={() => load(data.parent!)}
                className="flex min-h-11 items-center gap-2 border-b border-hairline px-3 text-left text-sm text-muted hover:bg-rowsel hover:text-text"
              >
                <span aria-hidden="true">&uarr;</span>
                <span>.. (up)</span>
              </button>
            )}
            {data.entries.length === 0 ? (
              <p className="px-3 py-3 text-sm text-faint">No subfolders here.</p>
            ) : (
              data.entries.map((entry) => (
                <button
                  key={entry.path}
                  type="button"
                  onClick={() => load(entry.path)}
                  className="flex min-h-11 items-center justify-between gap-3 border-b border-hairline px-3 text-left text-sm text-text last:border-b-0 hover:bg-rowsel"
                >
                  <span className="truncate">{entry.name}</span>
                  {entry.audio_count > 0 && (
                    <span className="shrink-0 text-2xs text-faint">
                      {entry.audio_count.toLocaleString()} tracks
                    </span>
                  )}
                </button>
              ))
            )}
          </div>

          {data.is_music_dir && (
            <p className="text-xs text-accent">This folder looks like a music library.</p>
          )}

          <button
            type="button"
            data-testid="folder-choose"
            onClick={() => onChoose(data.path)}
            className="min-h-11 self-start rounded bg-accent px-4 text-sm font-semibold text-bg"
          >
            Use this folder
          </button>
        </>
      )}
    </div>
  );
}
