import { test, expect, type Page } from "@playwright/test";

// The generated playlist must survive a page reload (localStorage restore).

async function generate(page: Page) {
  await page.getByTestId("seed-input").fill("Acetone");
  await page.getByRole("button", { name: "▸ Generate" }).click();
  await expect(page.getByTestId("track-table")).toBeVisible({ timeout: 15000 });
}

test("the playlist survives a page reload", async ({ page }) => {
  await page.goto("/");
  await generate(page);
  await expect(page.getByText("Sundown")).toBeVisible();

  await page.reload();

  // Restored from localStorage — no regeneration needed.
  await expect(page.getByTestId("track-table")).toBeVisible();
  await expect(page.getByText("Sundown")).toBeVisible();
});
