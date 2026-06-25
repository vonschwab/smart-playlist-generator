import { test, expect } from "@playwright/test";
import type { Page } from "@playwright/test";

async function generate(page: Page) {
  await page.goto("/");
  await page.getByTestId("seed-input").fill("Acetone");
  await page.getByRole("button", { name: /Generate/ }).click();
  await expect(page.getByTestId("track-table")).toBeVisible({ timeout: 15000 });
}

test("edit genres: typed-but-not-Entered genre is saved on Save", async ({ page }) => {
  // The for_album / search reads hit no DB in the Playwright server; stub them.
  await page.route("**/api/genres/for_album**", (r) => r.fulfill({ json: { genres: [] } }));
  await page.route("**/api/genres/search**", (r) => r.fulfill({ json: { items: [] } }));

  let editBody: { genres?: string[] } = {};
  await page.route("**/api/edit_genres", async (route) => {
    editBody = route.request().postDataJSON();
    await route.fulfill({
      json: {
        ok: true, resolved: editBody.genres ?? [], unknown: [],
        added: editBody.genres ?? [], removed: [], no_change: false,
      },
    });
  });

  await generate(page);
  await page.getByTestId("kebab-btn").nth(1).click();
  await page.getByText(/Edit genres for album/).click();
  await expect(page.getByTestId("edit-genres-dialog")).toBeVisible();

  // Type a genre but DO NOT press Enter, then click Save.
  await page.getByTestId("genre-input").fill("dream pop");
  await page.getByRole("button", { name: /^Save/ }).click();

  // The flush appended the typed genre to the saved payload (the empty-override bug fix).
  await expect.poll(() => editBody.genres ?? []).toContain("dream pop");
  await expect(page.getByTestId("edit-genres-dialog")).toBeHidden();
});

test("refresh genres for generation fires a job without error", async ({ page }) => {
  let called = false;
  await page.route("**/api/refresh_genre_artifact", async (route) => {
    called = true;
    await route.fulfill({ json: { job_id: "job-1" } });
  });

  await generate(page);
  await page.getByRole("button", { name: /Refresh genres for generation/ }).click();
  await expect.poll(() => called).toBe(true);
});
