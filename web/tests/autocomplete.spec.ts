import { test, expect } from "@playwright/test";

test("seed search appends a second page when the dropdown is scrolled", async ({ page }) => {
  await page.route("**/api/tracks/search**", async (route) => {
    const url = new URL(route.request().url());
    const offset = Number(url.searchParams.get("offset") ?? "0");
    const items = Array.from({ length: 25 }, (_, k) => {
      const i = offset + k;
      return {
        track_id: `t${i}`,
        title: `Song ${i}`,
        artist: "Beach House",
        album: "Bloom",
        duration_ms: 200000,
        file_path: `/m/${i}.flac`,
        genres: ["dream pop"],
      };
    });
    await route.fulfill({ json: { items, has_more: offset === 0 } }); // one extra page only
  });

  await page.goto("/");

  // SeedTrackSection (and its search input) only renders in SEEDS mode
  await page.getByRole("combobox", { name: "Generation mode" }).selectOption("seeds");

  await page.getByTestId("seed-search-input").fill("beach");

  // Wait for debounce + network
  await page.waitForTimeout(400);

  await expect(page.getByText("Song 0", { exact: true })).toBeVisible();
  await expect(page.getByText("Song 24", { exact: true })).toBeVisible();

  // Scroll the dropdown to trigger infinite-scroll load
  const list = page.locator("ul.overflow-auto").first();
  await list.evaluate((el) => { el.scrollTop = el.scrollHeight; });

  await expect(page.getByText("Song 40", { exact: true })).toBeVisible();
});
