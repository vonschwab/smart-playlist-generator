import { describe, it, expect } from "vitest";
import { friendlyError } from "./errors";

// S2 (docs/UI_UX_DISCIPLINE.md): errors go through one path — no raw
// "Error: …" strings, no bare stack noise, always something actionable.

describe("friendlyError", () => {
  it("uses the message of an Error without the 'Error:' prefix", () => {
    expect(friendlyError(new Error("seed artist not found"))).toBe("seed artist not found");
  });

  it("strips a literal 'Error:' prefix from stringified values", () => {
    expect(friendlyError("Error: HTTP 504")).toBe("HTTP 504");
  });

  it("maps network failures to a server-unreachable hint", () => {
    expect(friendlyError(new TypeError("Failed to fetch"))).toBe(
      "Can't reach the server — is it still running?",
    );
  });

  it("falls back to a generic message for empty values", () => {
    expect(friendlyError("")).toBe("Something went wrong.");
    expect(friendlyError(undefined)).toBe("Something went wrong.");
  });
});
