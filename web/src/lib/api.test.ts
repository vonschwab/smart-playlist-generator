import { describe, it, expect, vi, afterEach } from "vitest";
import { api } from "./api";

function mockFetch(json: unknown, ok = true, status = 200) {
  globalThis.fetch = vi.fn().mockResolvedValue({ ok, status, json: async () => json } as Response);
}
afterEach(() => vi.restoreAllMocks());

describe("setup api", () => {
  it("browseDir passes path and returns entries", async () => {
    mockFetch({ path: "/m", parent: "/", entries: [{ name: "a", path: "/m/a", audio_count: 3 }], is_music_dir: false });
    const r = await api.browseDir("/m");
    expect(r.entries[0].audio_count).toBe(3);
    expect((globalThis.fetch as any).mock.calls[0][0]).toContain("/api/setup/browse");
  });
  it("testService posts config and returns a CheckResult", async () => {
    mockFetch({ id: "lastfm", status: "pass", summary: "ok", fix_hint: null });
    const r = await api.testService("lastfm", { lastfm: { api_key: "k" } });
    expect(r.status).toBe("pass");
  });
  it("writeConfig posts the draft", async () => {
    mockFetch({ ok: true, config_path: "/c/config.yaml" });
    const r = await api.writeConfig({ music_directory: "/m", ai_genre_provider: "zero_touch" });
    expect(r.ok).toBe(true);
  });
});

describe("jsonOrThrow error shape", () => {
  it("attaches the HTTP status to the thrown Error so callers can branch without string-matching", async () => {
    mockFetch({ detail: "config.yaml already exists" }, false, 409);
    await expect(api.writeConfig({ music_directory: "/m" })).rejects.toMatchObject({
      message: "config.yaml already exists",
      status: 409,
    });
  });

  it("falls back to a generic HTTP status message when the body has no detail", async () => {
    mockFetch({}, false, 500);
    await expect(api.writeConfig({ music_directory: "/m" })).rejects.toMatchObject({
      message: "HTTP 500",
      status: 500,
    });
  });
});
