import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen, fireEvent, cleanup, waitFor } from "@testing-library/react";

vi.mock("../lib/api", () => ({
  api: { artistsSearch: vi.fn(async () => ({ items: ["Alex G", "(Sandy) Alex G"] })) },
}));
import { api } from "../lib/api";
import { ArtistAutocomplete } from "./ArtistAutocomplete";

afterEach(() => { cleanup(); vi.clearAllMocks(); });

describe("ArtistAutocomplete", () => {
  it("queries the api and picks a suggestion", async () => {
    const onPick = vi.fn();
    render(<ArtistAutocomplete onPick={onPick} />);
    fireEvent.change(screen.getByTestId("artist-autocomplete-input"), { target: { value: "alex" } });
    await waitFor(() => expect(api.artistsSearch).toHaveBeenCalled());
    const opt = await screen.findAllByTestId("artist-suggestion");
    fireEvent.click(opt[0]);
    expect(onPick).toHaveBeenCalledWith("Alex G");
  });
});
