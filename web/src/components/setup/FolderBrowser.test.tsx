import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen, fireEvent, waitFor, cleanup } from "@testing-library/react";
import FolderBrowser from "./FolderBrowser";
import { api } from "../../lib/api";

// This project doesn't enable vitest `globals`, so RTL's own auto-cleanup
// (gated on a global `afterEach`) never registers — see SetupWizard.test.tsx.
afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe("FolderBrowser", () => {
  it("lists entries and reports a chosen folder", async () => {
    vi.spyOn(api, "browseDir").mockResolvedValue({
      path: "/home/u/Music", parent: "/home/u",
      entries: [{ name: "FLAC", path: "/home/u/Music/FLAC", audio_count: 8412 }],
      is_music_dir: false,
    });
    const onChoose = vi.fn();
    render(<FolderBrowser onChoose={onChoose} />);
    await waitFor(() => screen.getByText("FLAC"));
    expect(screen.getByText(/8,?412/)).toBeTruthy();
    fireEvent.click(screen.getByText("FLAC")); // descend
    await waitFor(() => expect(api.browseDir).toHaveBeenCalledWith("/home/u/Music/FLAC"));
  });

  it("shows an inline error and recovers when browseDir is rejected", async () => {
    vi.spyOn(api, "browseDir")
      .mockResolvedValueOnce({
        path: "/home/u", parent: null,
        entries: [{ name: "bad", path: "/home/u/bad", audio_count: 0 }],
        is_music_dir: false,
      })
      .mockRejectedValueOnce(new Error("Not a directory: /home/u/bad"))
      .mockResolvedValueOnce({
        path: "/home/u", parent: null,
        entries: [{ name: "bad", path: "/home/u/bad", audio_count: 0 }],
        is_music_dir: false,
      });
    const onChoose = vi.fn();
    render(<FolderBrowser onChoose={onChoose} />);
    await waitFor(() => screen.getByText("bad"));
    fireEvent.click(screen.getByText("bad"));
    await waitFor(() => screen.getByRole("alert"));
    expect(screen.getByText(/Not a directory/)).toBeTruthy();
    // Recover via retry — shouldn't crash, should re-fetch.
    fireEvent.click(screen.getByTestId("folder-retry"));
    await waitFor(() => expect(api.browseDir).toHaveBeenCalledTimes(3));
  });
});
