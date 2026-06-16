import { test, expect } from "@playwright/test";

test("seed search loads a further page on scroll (beyond the auto-prefetched page)", async ({ page }) => {
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
    await route.fulfill({ json: { items, has_more: true } }); // always another page
  });

  await page.goto("/");
  // The seed search input only mounts in "seeds" generation mode.
  await page.getByRole("combobox", { name: "Generation mode" }).selectOption("seeds");
  await page.getByTestId("seed-search-input").fill("beach");

  // First page (Song 0-24) plus the auto-prefetched second page (Song 25-49).
  await expect(page.getByText("Song 49", { exact: true })).toBeVisible();
  // Song 60 belongs to a THIRD page that only a scroll can load.
  await expect(page.getByText("Song 60", { exact: true })).toHaveCount(0);

  const list = page.locator("ul.overflow-auto").first();
  await list.evaluate((el) => { el.scrollTop = el.scrollHeight; });

  await expect(page.getByText("Song 60", { exact: true })).toBeVisible();
});
