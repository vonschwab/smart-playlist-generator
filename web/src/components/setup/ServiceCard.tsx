import { useEffect, useState } from "react";
import { api } from "../../lib/api";
import type { CheckResult } from "../../lib/types";

export interface ServiceField {
  key: string;
  label: string;
  type?: "text" | "password";
  placeholder?: string;
}

interface ServiceCardProps {
  /** Service id as understood by the backend probe (`src/setup/services.py::SERVICES`). */
  service: string;
  title: string;
  description: string;
  fields: ServiceField[];
  /** Current string values for `fields`, keyed by `field.key`. Controlled by the parent step. */
  values: Record<string, string>;
  onChange: (values: Record<string, string>) => void;
  /**
   * Non-editable fields merged into the outgoing `api.testService` payload
   * only (never rendered, never written into the draft here) — e.g. Plex's
   * `enabled: true`, which `_plex_credentials` requires before it'll probe
   * at all (src/setup/services.py).
   */
  extraTestFields?: Record<string, unknown>;
}

const ICON: Record<CheckResult["status"], string> = { pass: "✓", warn: "⚠", fail: "✗" };
const COLOR: Record<CheckResult["status"], string> = {
  pass: "text-accent",
  warn: "text-warn",
  fail: "text-danger",
};
const PILL_BG: Record<CheckResult["status"], string> = {
  pass: "bg-accent/10 border-accent/40",
  warn: "bg-warn/10 border-warn/40",
  fail: "bg-danger/10 border-danger/40",
};

// Matches the app's input token pattern (ArtistAutocomplete.tsx): legible
// light-on-dark panel2 well, 44px tap target, visible focus ring via
// border-accent (the :focus-visible outline in index.css handles keyboard).
const INPUT_CLASS =
  "w-full min-h-11 rounded-md border border-border bg-panel2 px-3 py-2 text-sm text-text placeholder:text-faint outline-none focus:border-accent";

// One optional-service card: name/description, an always-visible inline
// Test button + colored status pill, and credential fields that collapse
// under a "+ Add" affordance until the user opts in (or already has values
// from a prior draft). The Test button intentionally lives OUTSIDE the
// collapsible section — you can probe with whatever's in the draft (including
// "nothing configured") without first expanding the card.
export default function ServiceCard({
  service, title, description, fields, values, onChange, extraTestFields,
}: ServiceCardProps) {
  const hasValues = fields.some((f) => Boolean(values[f.key]));
  const [expanded, setExpanded] = useState(hasValues);
  const [testing, setTesting] = useState(false);
  const [result, setResult] = useState<CheckResult | null>(null);

  // Resync when the parent clears every field's value out from under us --
  // e.g. Services.tsx's "Skip all" resets the draft slice this card reads
  // from. Without this, the fields go blank but the card stays visually
  // expanded instead of collapsing back to "+ Add".
  useEffect(() => {
    if (!hasValues) setExpanded(false);
  }, [hasValues]);

  const setField = (key: string, v: string) => onChange({ ...values, [key]: v });

  // A failed/errored test NEVER throws and NEVER blocks — it only renders a
  // status pill (+ fix_hint on fail). Next-gating for this step is fixed
  // `true` in useWizard's computeCanNext; this handler has no bearing on it.
  const runTest = async () => {
    setTesting(true);
    setResult(null);
    try {
      const r = await api.testService(service, { [service]: { ...values, ...extraTestFields } });
      setResult(r);
    } catch (err) {
      setResult({
        id: service,
        status: "fail",
        summary: err instanceof Error ? err.message : "test request failed",
        fix_hint: null,
      });
    } finally {
      setTesting(false);
    }
  };

  return (
    <div data-testid={`service-card-${service}`} className="rounded border border-border bg-panel p-4">
      <div className="flex items-start justify-between gap-3">
        <div>
          <h3 className="text-sm font-semibold text-text">{title}</h3>
          <p className="text-xs text-faint">{description}</p>
        </div>
        <button
          type="button"
          data-testid="service-test"
          onClick={runTest}
          disabled={testing}
          className="min-h-11 shrink-0 rounded border border-border px-3 text-xs text-muted hover:text-text disabled:opacity-50"
        >
          {testing ? "Testing…" : "Test"}
        </button>
      </div>

      {result && (
        <div className="mt-2 flex flex-col items-start gap-1">
          <span
            data-testid={`service-result-${service}`}
            className={`inline-flex items-center gap-1 rounded-full border px-2.5 py-1 text-xs ${PILL_BG[result.status]} ${COLOR[result.status]}`}
          >
            <span aria-hidden="true">{ICON[result.status]}</span>
            {result.summary}
          </span>
          {result.status === "fail" && result.fix_hint && (
            <p className="text-xs text-faint">{result.fix_hint}</p>
          )}
        </div>
      )}

      {expanded ? (
        <div className="mt-3 flex flex-col gap-2">
          {fields.map((f) => (
            <label key={f.key} className="flex flex-col gap-1 text-xs text-muted">
              {f.label}
              <input
                type={f.type ?? "text"}
                value={values[f.key] ?? ""}
                placeholder={f.placeholder}
                onChange={(e) => setField(f.key, e.target.value)}
                className={INPUT_CLASS}
              />
            </label>
          ))}
          <button
            type="button"
            data-testid={`service-collapse-${service}`}
            onClick={() => setExpanded(false)}
            className="min-h-11 self-start text-xs text-faint hover:text-muted"
          >
            Remove
          </button>
        </div>
      ) : (
        <button
          type="button"
          data-testid={`service-add-${service}`}
          onClick={() => setExpanded(true)}
          className="mt-2 min-h-11 text-left text-xs font-medium text-accent hover:underline"
        >
          + Add {title}
        </button>
      )}
    </div>
  );
}
