// One error-to-copy path for the whole GUI (discipline S2): no raw
// "Error: …" strings, no stack noise, network failures say what to check.
export function friendlyError(e: unknown): string {
  const raw = e instanceof Error ? e.message : String(e ?? "");
  const msg = raw.replace(/^Error:\s*/, "").trim();
  if (!msg) return "Something went wrong.";
  if (/failed to fetch|networkerror|load failed/i.test(msg)) {
    return "Can't reach the server — is it still running?";
  }
  return msg;
}
