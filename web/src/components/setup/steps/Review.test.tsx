import { describe, it, expect, afterEach, vi } from "vitest";
import { render, screen, fireEvent, cleanup, waitFor } from "@testing-library/react";
import { Review } from "./Review";
import { api, type ApiError } from "../../../lib/api";
import type { WizardDraft } from "../useWizard";

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

function apiError(message: string, status: number): ApiError {
  const err = new Error(message) as ApiError;
  err.status = status;
  return err;
}

const draft: WizardDraft = { music_directory: "/music", ai_genre_provider: "zero_touch" };

describe("Review", () => {
  it("a 409 status surfaces the reconfigure affordance, and retrying calls writeConfig with reconfigure:true", async () => {
    const writeConfig = vi
      .spyOn(api, "writeConfig")
      .mockRejectedValueOnce(apiError("config.yaml already exists at /c/config.yaml", 409))
      .mockResolvedValueOnce({ ok: true, config_path: "/c/config.yaml" });
    const goTo = vi.fn();

    render(<Review draft={draft} goTo={goTo} />);
    fireEvent.click(screen.getByTestId("wizard-write-config"));

    await waitFor(() => screen.getByTestId("wizard-reconfigure"));
    expect(goTo).not.toHaveBeenCalled();

    fireEvent.click(screen.getByTestId("wizard-reconfigure"));

    await waitFor(() => expect(goTo).toHaveBeenCalledWith("analyze"));
    expect(writeConfig).toHaveBeenLastCalledWith(
      expect.objectContaining({ reconfigure: true })
    );
  });

  it("a non-409 error shows the error inline and keeps the draft (no reconfigure affordance, no goTo)", async () => {
    vi.spyOn(api, "writeConfig").mockRejectedValue(apiError("disk full", 500));
    const goTo = vi.fn();

    render(<Review draft={draft} goTo={goTo} />);
    fireEvent.click(screen.getByTestId("wizard-write-config"));

    await waitFor(() => screen.getByRole("alert"));
    expect(screen.getByRole("alert").textContent).toMatch(/disk full/);
    expect(screen.queryByTestId("wizard-reconfigure")).toBeNull();
    expect(goTo).not.toHaveBeenCalled();
  });

  // C1: the rail lets a user reach Review without ever visiting the Music
  // step (no order gate on goTo). The confirm button must never be clickable
  // with no folder chosen — this is defense in depth alongside the
  // authoritative server-side guard in config_writer.write_config.
  it("disables the confirm button and shows a hint back to Music when draft has no music_directory", () => {
    const writeConfig = vi.spyOn(api, "writeConfig");
    const goTo = vi.fn();
    const noFolderDraft = { ai_genre_provider: "zero_touch" };

    render(<Review draft={noFolderDraft} goTo={goTo} />);

    const confirm = screen.getByTestId("wizard-write-config");
    expect(confirm.hasAttribute("disabled")).toBe(true);
    expect(screen.getByText(/choose a music folder first/i)).toBeTruthy();

    fireEvent.click(confirm);
    expect(writeConfig).not.toHaveBeenCalled();
  });

  it("the missing-folder hint's button navigates back to the Music step", () => {
    const goTo = vi.fn();
    const noFolderDraft = { ai_genre_provider: "zero_touch" };

    render(<Review draft={noFolderDraft} goTo={goTo} />);
    fireEvent.click(screen.getByRole("button", { name: /music library/i }));

    expect(goTo).toHaveBeenCalledWith("music");
  });

  it("leaves the confirm button enabled when draft has a music_directory", () => {
    render(<Review draft={draft} goTo={vi.fn()} />);
    expect(screen.getByTestId("wizard-write-config").hasAttribute("disabled")).toBe(false);
    expect(screen.queryByText(/choose a music folder first/i)).toBeNull();
  });
});
