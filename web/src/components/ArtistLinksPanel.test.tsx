import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen, fireEvent, cleanup, waitFor } from "@testing-library/react";
import type { ArtistLinksListResponse } from "../lib/types";

vi.mock("../lib/api", () => ({
  api: {
    artistLinksList: vi.fn(async () => ({ groups: [] })),
    artistLinksSave: vi.fn(async () => ({ ok: true, count: 1 })),
    artistsSearch: vi.fn(async () => ({ items: ["Smog", "Bill Callahan"] })),
  },
}));
import { api } from "../lib/api";
import { ArtistLinksPanel } from "./ArtistLinksPanel";

afterEach(() => { cleanup(); vi.clearAllMocks(); });

describe("ArtistLinksPanel", () => {
  it("builds a sibling group and saves it", async () => {
    render(<ArtistLinksPanel />);
    await waitFor(() => expect(api.artistLinksList).toHaveBeenCalled());

    fireEvent.click(screen.getByTestId("link-type-sibling"));
    // add two members via the autocomplete
    for (const name of ["Smog", "Bill Callahan"]) {
      fireEvent.change(screen.getByTestId("artist-autocomplete-input"), { target: { value: name } });
      const opt = await screen.findAllByTestId("artist-suggestion");
      fireEvent.click(opt.find((o) => o.textContent === name) ?? opt[0]);
    }
    fireEvent.click(screen.getByTestId("add-group"));
    fireEvent.click(screen.getByTestId("save-links"));

    await waitFor(() => expect(api.artistLinksSave).toHaveBeenCalled());
    const arg = vi.mocked(api.artistLinksSave).mock.calls[0][0];
    expect(arg.groups[0]).toEqual({ type: "sibling", members: ["Smog", "Bill Callahan"] });
  });

  it("shows an empty state when there are no links", async () => {
    render(<ArtistLinksPanel />);
    await waitFor(() => expect(api.artistLinksList).toHaveBeenCalled());
    expect(await screen.findByText(/no artist links yet/i)).toBeTruthy();
  });

  it("disables Add group until a type and 2+ members are picked", async () => {
    render(<ArtistLinksPanel />);
    await waitFor(() => expect(api.artistLinksList).toHaveBeenCalled());
    expect((screen.getByTestId("add-group") as HTMLButtonElement).disabled).toBe(true);

    fireEvent.click(screen.getByTestId("link-type-alias"));
    expect((screen.getByTestId("add-group") as HTMLButtonElement).disabled).toBe(true);

    fireEvent.change(screen.getByTestId("artist-autocomplete-input"), { target: { value: "Smog" } });
    const opt = await screen.findAllByTestId("artist-suggestion");
    fireEvent.click(opt[0]);
    expect((screen.getByTestId("add-group") as HTMLButtonElement).disabled).toBe(true);
  });

  it("reloads from the server when save fails", async () => {
    vi.mocked(api.artistLinksSave).mockRejectedValueOnce(new Error("boom"));
    render(<ArtistLinksPanel />);
    await waitFor(() => expect(api.artistLinksList).toHaveBeenCalledTimes(1));

    fireEvent.click(screen.getByTestId("save-links"));
    await waitFor(() => expect(api.artistLinksSave).toHaveBeenCalledTimes(1));
    await waitFor(() => expect(api.artistLinksList).toHaveBeenCalledTimes(2));
  });

  it("keeps Save disabled and never calls artistLinksSave while the initial load is still in flight", async () => {
    vi.mocked(api.artistLinksList).mockImplementationOnce(() => new Promise(() => {}));
    render(<ArtistLinksPanel />);
    await waitFor(() => expect(api.artistLinksList).toHaveBeenCalled());

    expect(await screen.findByText(/loading/i)).toBeTruthy();
    expect((screen.getByTestId("save-links") as HTMLButtonElement).disabled).toBe(true);

    fireEvent.click(screen.getByTestId("save-links"));
    expect(api.artistLinksSave).not.toHaveBeenCalled();
  });

  it("shows an error with a retry option (and keeps Save disabled) when the initial load fails", async () => {
    vi.mocked(api.artistLinksList).mockRejectedValueOnce(new Error("network down"));
    render(<ArtistLinksPanel />);

    expect(await screen.findByRole("alert")).toBeTruthy();
    expect((screen.getByTestId("save-links") as HTMLButtonElement).disabled).toBe(true);

    fireEvent.click(screen.getByTestId("save-links"));
    expect(api.artistLinksSave).not.toHaveBeenCalled();

    fireEvent.click(screen.getByTestId("retry-load"));
    await waitFor(() => expect(api.artistLinksList).toHaveBeenCalledTimes(2));
    expect((screen.getByTestId("save-links") as HTMLButtonElement).disabled).toBe(false);
  });

  it("folds a valid pending draft into the save payload when Save is clicked without Add group", async () => {
    render(<ArtistLinksPanel />);
    await waitFor(() => expect(api.artistLinksList).toHaveBeenCalled());

    fireEvent.click(screen.getByTestId("link-type-alias"));
    for (const name of ["Smog", "Bill Callahan"]) {
      fireEvent.change(screen.getByTestId("artist-autocomplete-input"), { target: { value: name } });
      const opt = await screen.findAllByTestId("artist-suggestion");
      fireEvent.click(opt.find((o) => o.textContent === name) ?? opt[0]);
    }
    // Note: no click on "add-group" here — Save alone must not drop the draft.
    fireEvent.click(screen.getByTestId("save-links"));

    await waitFor(() => expect(api.artistLinksSave).toHaveBeenCalled());
    const arg = vi.mocked(api.artistLinksSave).mock.calls[0][0];
    expect(arg.groups).toEqual([{ type: "alias", members: ["Smog", "Bill Callahan"] }]);
  });

  it("blocks Save and shows an inline hint when the pending draft is partial", async () => {
    render(<ArtistLinksPanel />);
    await waitFor(() => expect(api.artistLinksList).toHaveBeenCalled());

    fireEvent.click(screen.getByTestId("link-type-alias"));
    fireEvent.change(screen.getByTestId("artist-autocomplete-input"), { target: { value: "Smog" } });
    const opt = await screen.findAllByTestId("artist-suggestion");
    fireEvent.click(opt[0]);

    expect(await screen.findByTestId("draft-warning")).toBeTruthy();
    expect((screen.getByTestId("save-links") as HTMLButtonElement).disabled).toBe(true);

    fireEvent.click(screen.getByTestId("save-links"));
    expect(api.artistLinksSave).not.toHaveBeenCalled();
  });

  it("renders gracefully when a loaded group has a malformed (non-array) members field", async () => {
    // Deliberately malformed (members is not an array) to exercise the defensive render guard.
    vi.mocked(api.artistLinksList).mockResolvedValueOnce({
      groups: [{ type: "alias", members: "not-an-array" }],
    } as unknown as ArtistLinksListResponse);
    render(<ArtistLinksPanel />);
    await waitFor(() => expect(api.artistLinksList).toHaveBeenCalled());

    expect(await screen.findByText(/unreadable member list/i)).toBeTruthy();
    expect(screen.getByTestId("remove-group-0")).toBeTruthy();
  });
});
