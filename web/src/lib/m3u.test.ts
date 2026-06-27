import { describe, it, expect, vi, afterEach } from "vitest";
import { toM3U8Filename, downloadM3U8 } from "./m3u";
import type { TrackOut } from "./types";

function track(artist: string): TrackOut {
  return {
    position: 0,
    artist,
    title: "T",
    album: "",
    duration_ms: 0,
    file_path: "/x.mp3",
    genres: [],
  };
}

describe("toM3U8Filename", () => {
  it("appends .m3u8 when the extension is missing", () => {
    expect(toM3U8Filename("Alvvays — 2026-06-25")).toBe("Alvvays — 2026-06-25.m3u8");
  });

  it("does not double the extension", () => {
    expect(toM3U8Filename("mix.m3u8")).toBe("mix.m3u8");
  });

  it("strips filesystem-illegal characters", () => {
    expect(toM3U8Filename('a/b\\c:d*e?f"g<h>i|j')).toBe("abcdefghij.m3u8");
  });

  it("collapses internal whitespace and trims", () => {
    expect(toM3U8Filename("  my   mix  ")).toBe("my mix.m3u8");
  });

  it("falls back to 'playlist' when nothing usable remains", () => {
    expect(toM3U8Filename("///")).toBe("playlist.m3u8");
  });

  it("preserves the em-dash separator", () => {
    expect(toM3U8Filename("A — B")).toBe("A — B.m3u8");
  });
});

describe("downloadM3U8", () => {
  afterEach(() => vi.restoreAllMocks());

  it("downloads with a sanitized .m3u8 filename derived from the given name", () => {
    (URL as unknown as { createObjectURL: () => string }).createObjectURL = vi.fn(() => "blob:x");
    (URL as unknown as { revokeObjectURL: () => void }).revokeObjectURL = vi.fn();
    let captured = "";
    vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(function (
      this: HTMLAnchorElement,
    ) {
      captured = this.download;
    });

    downloadM3U8([track("Alvvays")], "a/b: mix");

    expect(captured).toBe("ab mix.m3u8");
  });
});
