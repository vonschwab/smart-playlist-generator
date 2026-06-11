import { test, expect } from "@playwright/test";

test("Tools tab is visible and dry-run analyze completes", async ({ page }) => {
  await page.goto("/");

  // Switch to Tools tab
  await page.getByRole("button", { name: /tools/i }).click();
  await expect(page.getByText("Analyze Library")).toBeVisible();
  await expect(page.getByText("Enrich Genres")).toBeVisible();

  // Enable dry run
  await page.getByLabel(/dry run/i).check();

  // Click the first Run button (Analyze Library card)
  await page.getByRole("button", { name: /^run$/i }).first().click();

  // Wait for the Run button to re-enable (job completes, analyzeJobId clears)
  await expect(page.getByRole("button", { name: /^run$/i }).first()).toBeEnabled({
    timeout: 10000,
  });
});

test("Enrich all pending button is visible", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", { name: /tools/i }).click();
  await expect(page.getByRole("button", { name: /enrich all pending/i })).toBeVisible();
});
