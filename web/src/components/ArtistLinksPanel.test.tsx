import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen, fireEvent, cleanup, waitFor } from "@testing-library/react";

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
});
