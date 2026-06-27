import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen, fireEvent, cleanup } from "@testing-library/react";
import { ExportM3U8Dialog } from "./ExportM3U8Dialog";
import type { TrackOut } from "../lib/types";

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

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe("ExportM3U8Dialog", () => {
  it("pre-fills the name field with the default name when opened", () => {
    render(
      <ExportM3U8Dialog
        open
        onOpenChange={() => {}}
        tracks={[track("Alvvays")]}
        defaultName="Alvvays — 2026-06-25"
      />,
    );
    const input = screen.getByTestId("m3u8-name") as HTMLInputElement;
    expect(input.value).toBe("Alvvays — 2026-06-25");
  });

  it("downloads with the sanitized entered name and closes on Download", () => {
    (URL as unknown as { createObjectURL: () => string }).createObjectURL = vi.fn(() => "blob:x");
    (URL as unknown as { revokeObjectURL: () => void }).revokeObjectURL = vi.fn();
    let captured = "";
    vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(function (
      this: HTMLAnchorElement,
    ) {
      captured = this.download;
    });
    const onOpenChange = vi.fn();

    render(
      <ExportM3U8Dialog
        open
        onOpenChange={onOpenChange}
        tracks={[track("Alvvays")]}
        defaultName="Alvvays — 2026-06-25"
      />,
    );
    const input = screen.getByTestId("m3u8-name") as HTMLInputElement;
    fireEvent.change(input, { target: { value: "my mix" } });
    fireEvent.click(screen.getByTestId("m3u8-download"));

    expect(captured).toBe("my mix.m3u8");
    expect(onOpenChange).toHaveBeenCalledWith(false);
  });

  it("disables Download when the name is blank", () => {
    render(
      <ExportM3U8Dialog
        open
        onOpenChange={() => {}}
        tracks={[track("Alvvays")]}
        defaultName="Alvvays — 2026-06-25"
      />,
    );
    const input = screen.getByTestId("m3u8-name") as HTMLInputElement;
    fireEvent.change(input, { target: { value: "   " } });
    expect((screen.getByTestId("m3u8-download") as HTMLButtonElement).disabled).toBe(true);
  });
});
