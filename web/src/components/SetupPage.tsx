import type { SetupStatus } from "../lib/types";
import SetupWizard from "./setup/SetupWizard";

// Gate rendered by App.tsx when state === "needs_setup" (the gate wiring
// itself lives in App.tsx and stays intact). SP-3 (Task 5) replaced the
// static instructions body with the guided wizard shell; this component
// keeps the `data-testid="setup-page"` contract web/tests/setup-gate.spec.ts
// asserts on.
export default function SetupPage({
  status,
  onSetupComplete,
}: {
  status: SetupStatus;
  onSetupComplete?: () => void;
}) {
  return (
    <main data-testid="setup-page" className="min-h-dvh bg-bg">
      <SetupWizard status={status} onSetupComplete={onSetupComplete} />
    </main>
  );
}
