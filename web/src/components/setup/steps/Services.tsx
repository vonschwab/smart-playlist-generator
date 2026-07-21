import type { Dispatch, SetStateAction } from "react";
import ServiceCard, { type ServiceField } from "../ServiceCard";
import type { WizardDraft } from "../useWizard";

interface ServicesProps {
  draft: WizardDraft;
  setDraft: Dispatch<SetStateAction<WizardDraft>>;
}

const LASTFM_FIELDS: ServiceField[] = [
  { key: "api_key", label: "API key", placeholder: "Last.fm API key" },
  { key: "username", label: "Username", placeholder: "Last.fm username" },
];
const DISCOGS_FIELDS: ServiceField[] = [
  { key: "token", label: "Personal access token", type: "password", placeholder: "Discogs token" },
];
const PLEX_FIELDS: ServiceField[] = [
  { key: "base_url", label: "Server URL", placeholder: "http://localhost:32400" },
  { key: "token", label: "Plex token", type: "password", placeholder: "X-Plex-Token" },
];

// Narrow an arbitrary draft slice (WizardDraft's lastfm/discogs/plex each
// have a distinct shape — plex is Record<string, unknown>, the others are
// named interfaces) down to the plain string map ServiceCard's controlled
// inputs expect. `unknown` param sidesteps assignability friction between
// those shapes and a Record<string, unknown> parameter type.
function asStringValues(v: unknown): Record<string, string> {
  const out: Record<string, string> = {};
  if (!v || typeof v !== "object") return out;
  for (const [k, val] of Object.entries(v as Record<string, unknown>)) {
    if (typeof val === "string") out[k] = val;
  }
  return out;
}

// Step 4/7: optional service connections. Each card writes straight into the
// draft on every keystroke (no separate "save" step) and can be tested
// in-place via ServiceCard's Test button — a fail only ever shows a status
// pill + fix_hint, never blocks (useWizard's computeCanNext is unconditionally
// true here). "Skip all" clears every service key back to unset, matching the
// draft's pre-visit state.
export default function Services({ draft, setDraft }: ServicesProps) {
  return (
    <div data-testid="step-services" className="flex max-w-xl flex-col gap-4">
      <div className="flex items-center justify-between gap-3">
        <div>
          <h2 className="text-lg font-semibold text-text">Services</h2>
          <p className="text-sm text-muted">
            Optional integrations — connect now or skip and add them later from Settings.
          </p>
        </div>
        <button
          type="button"
          data-testid="services-skip-all"
          onClick={() => setDraft((d) => ({ ...d, lastfm: undefined, discogs: undefined, plex: undefined }))}
          className="min-h-11 shrink-0 rounded border border-border px-3 text-xs text-muted hover:text-text"
        >
          Skip all
        </button>
      </div>

      <ServiceCard
        service="lastfm"
        title="Last.fm"
        description="Scrobble history powers recency-aware playlists."
        fields={LASTFM_FIELDS}
        values={asStringValues(draft.lastfm)}
        onChange={(values) =>
          setDraft((d) => ({ ...d, lastfm: { api_key: values.api_key ?? "", username: values.username ?? "" } }))
        }
      />
      <ServiceCard
        service="discogs"
        title="Discogs"
        description="Collection and genre metadata enrichment."
        fields={DISCOGS_FIELDS}
        values={asStringValues(draft.discogs)}
        onChange={(values) => setDraft((d) => ({ ...d, discogs: { token: values.token ?? "" } }))}
      />
      <ServiceCard
        service="plex"
        title="Plex"
        description="Export generated playlists directly to a Plex server."
        fields={PLEX_FIELDS}
        values={asStringValues(draft.plex)}
        onChange={(values) =>
          setDraft((d) => ({
            ...d,
            plex: { enabled: true, base_url: values.base_url ?? "", token: values.token ?? "" },
          }))
        }
        extraTestFields={{ enabled: true }}
      />
    </div>
  );
}
