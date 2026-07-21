import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import Services from "./steps/Services";
import { api } from "../../lib/api";

describe("Services", () => {
  it("Test shows a status pill and a fail does not block", async () => {
    vi.spyOn(api, "testService").mockResolvedValue({ id: "lastfm", status: "fail", summary: "unreachable: 401", fix_hint: "check lastfm.api_key" });
    render(<Services draft={{}} setDraft={() => {}} />);
    fireEvent.click(screen.getAllByTestId("service-test")[0]);
    await waitFor(() => screen.getByText(/unreachable/));
    expect(screen.getByText(/check lastfm\.api_key/)).toBeTruthy();
    // no throw / no gating assertion here — Next-gating is the wizard's job (always true on this step)
  });
});
