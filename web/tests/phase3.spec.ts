import { test, expect } from "@playwright/test";
import type { Page } from "@playwright/test";

async function generate(page: Page) {
  await page.goto("/");
  await page.getByTestId("seed-input").fill("Acetone");
  await page.getByRole("button", { name: /Generate/ }).click();
  await expect(page.getByTestId("track-table")).toBeVisible({ timeout: 15000 });
}

test("diagnostics empty state then content after generate", async ({ page }) => {
  await page.goto("/");
  await page.getByTestId("tab-diagnostics").click();
  await expect(page.getByTestId("diagnostics-empty")).toBeVisible();
  await generate(page);
  await page.getByTestId("tab-diagnostics").click();
  await expect(page.getByTestId("diagnostics-content")).toBeVisible();
  await expect(page.getByTestId("transition-bars")).toBeVisible();
});

test("blacklist tab lists entries and can add an artist", async ({ page }) => {
  await page.goto("/");
  await page.getByTestId("tab-blacklist").click();
  await expect(page.getByTestId("blacklist-panel")).toBeVisible();
  // Fake worker returns Nick Drake as a blacklisted artist.
  await expect(page.getByText("Nick Drake")).toBeVisible();
  await page.getByTestId("blacklist-search").fill("Coldplay");
  await page.getByTestId("blacklist-add").click();
  // Refetch happens; panel stays visible (no crash).
  await expect(page.getByTestId("blacklist-panel")).toBeVisible();
});

test("completed job shows re-run button", async ({ page }) => {
  await generate(page);
  await expect(page.getByTestId("job-rerun").first()).toBeVisible({ timeout: 10000 });
});
