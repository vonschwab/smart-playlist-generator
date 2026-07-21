import type { CheckResult, SetupStatus } from "../../../lib/types";

const ICON: Record<CheckResult["status"], string> = { pass: "✓", warn: "⚠", fail: "✗" };
const COLOR: Record<CheckResult["status"], string> = {
  pass: "text-accent",
  warn: "text-warn",
  fail: "text-danger",
};

// Step 2/7: read-only summary of `status.checks` (env/DB/service probes the
// server already ran). No draft state here — just informs the user before
// they continue.
export function EnvironmentCheck({ status }: { status: SetupStatus }) {
  const checks = status.checks ?? [];
  return (
    <div data-testid="step-environment" className="flex max-w-xl flex-col gap-3">
      <h2 className="text-lg font-semibold text-text">Environment check</h2>
      <p className="text-sm text-muted">MixArc verified the following before setup can continue.</p>
      {checks.length === 0 ? (
        <p className="text-sm text-faint">No checks reported.</p>
      ) : (
        <ul className="flex flex-col gap-2">
          {checks.map((c) => (
            <li key={c.id} data-testid={`env-check-${c.id}`} className="rounded border border-border bg-panel p-3">
              <div className="flex items-center gap-2">
                <span className={`${COLOR[c.status]} font-semibold`} aria-hidden="true">{ICON[c.status]}</span>
                <span className="text-sm text-text">{c.summary}</span>
              </div>
              {c.fix_hint && <p className="mt-1 text-xs text-faint">{c.fix_hint}</p>}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
