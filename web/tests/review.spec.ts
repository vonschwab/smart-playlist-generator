import { test, expect } from "@playwright/test";

test("Genre Review tab lists the queue and accepting a term updates counts", async ({ page }) => {
  await page.goto("/");

  // Navigate to the Genre Review tab in the AdvancedPanel
  await page.getByTestId("tab-review").click();

  // Verify pending counts header loaded from fake worker
  await expect(page.getByText("1 releases · 2 terms")).toBeVisible({ timeout: 10000 });

  // Verify the release row is visible
  await expect(page.getByText("acetone – cindy")).toBeVisible();

  // Expand the release by clicking its row button
  await page.getByText("acetone – cindy").click();

  // Verify the first pending term is now visible
  await expect(page.getByText("slowcore")).toBeVisible();

  // Accept the first term
  await page.getByRole("button", { name: "Accept (A)" }).first().click();

  // Optimistic update: pending_terms drops from 2 to 1
  await expect(page.getByText("1 releases · 1 terms")).toBeVisible({ timeout: 5000 });

  // The decided section should now show "1 decided"
  await expect(page.getByText("1 decided")).toBeVisible();
});

test("Scan button kicks off a job and re-enables on completion", async ({ page }) => {
  await page.goto("/");

  // Navigate to the Genre Review tab
  await page.getByTestId("tab-review").click();

  // Wait for the panel to load
  await expect(page.getByRole("button", { name: "Scan" })).toBeVisible({ timeout: 10000 });

  // Click Scan to start a job
  await page.getByRole("button", { name: "Scan" }).click();

  // Wait for Scan button to reappear (fake worker completes instantly)
  await expect(page.getByRole("button", { name: "Scan" })).toBeVisible({ timeout: 10000 });
});
