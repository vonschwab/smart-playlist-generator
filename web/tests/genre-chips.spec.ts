import { test, expect } from "@playwright/test";

// Staged seeds fetch graph-canonical genres (POST /api/tracks/genres) and render
// them most-specific first: 6 chips + a "+N" overflow pill with a title tooltip.
// The search dropdown is intentionally untouched by this feature.
test("staged seeds show canonical genres with +N overflow", async ({ page }) => {
  await page.addInitScript(() => {
    localStorage.setItem("pg_mode", JSON.stringify("seeds"));
    localStorage.setItem(
      "pg_seed_tracks",
      JSON.stringify([
        { track_id: "k0", title: "Sundown", artist: "Acetone", album: "Cindy", genres: ["rock"], duration_ms: 200000, file_path: "/0.flac" },
      ]),
    );
  });

  // 7 ordered genres -> 6 chips + "+1"
  await page.route("**/api/tracks/genres", (route) =>
    route.fulfill({
      json: {
        k0: ["slowcore", "sadcore", "dream pop", "noise pop", "indie rock", "alternative rock", "rock"],
      },
    }),
  );

  await page.goto("/");

  // Canonical genres replaced the "rock" placeholder, most-specific first.
  await expect(page.getByText("slowcore")).toBeVisible();
  await expect(page.getByText("alternative rock")).toBeVisible();
  // 7th genre is behind the overflow pill.
  const overflow = page.getByTestId("genre-overflow");
  await expect(overflow).toHaveText("+1");
  await expect(overflow).toHaveAttribute("title", "rock");
});

test("staged seeds keep placeholder genres when the genres API fails", async ({ page }) => {
  await page.addInitScript(() => {
    localStorage.setItem("pg_mode", JSON.stringify("seeds"));
    localStorage.setItem(
      "pg_seed_tracks",
      JSON.stringify([
        { track_id: "k0", title: "Sundown", artist: "Acetone", album: "Cindy", genres: ["slowcore"], duration_ms: 200000, file_path: "/0.flac" },
      ]),
    );
  });
  await page.route("**/api/tracks/genres", (route) => route.abort());

  await page.goto("/");
  // Fallback: the metadata genres carried from the dropdown still render.
  await expect(page.getByText("slowcore")).toBeVisible();
});
