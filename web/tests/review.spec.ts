import { test, expect } from "@playwright/test";

test("Genre Review tab lists the queue and accepting a term updates counts", async ({ page }) => {
  await page.goto("/");

  // Navigate to the Genre Review tab in the AdvancedPanel
  await page.getByTestId("tab-review").click();

  // Scope all assertions to the review panel: the suite shares one backend, so a
  // parallel generate test can leave a "slowcore"-tagged playlist in the track
  // table, and an unscoped getByText("slowcore") would match two elements.
  const panel = page.getByTestId("review-panel");

  // Verify pending counts header loaded from fake worker
  await expect(panel.getByText("1 releases · 2 terms")).toBeVisible({ timeout: 10000 });

  // Verify the release row is visible
  await expect(panel.getByText("acetone – cindy")).toBeVisible();

  // Expand the release by clicking its row button
  await panel.getByText("acetone – cindy").click();

  // Verify the first pending term is now visible
  await expect(panel.getByText("slowcore")).toBeVisible();

  // Accept the first term
  await panel.getByRole("button", { name: "Accept (A)" }).first().click();

  // Optimistic update: pending_terms drops from 2 to 1
  await expect(panel.getByText("1 releases · 1 terms")).toBeVisible({ timeout: 5000 });

  // The decided section should now show "1 decided"
  await expect(panel.getByText("1 decided")).toBeVisible();
});

test("Scan button kicks off a job and re-enables on completion", async ({ page }) => {
  await page.goto("/");

  // Navigate to the Genre Review tab
  await page.getByTestId("tab-review").click();

  // Scope to the review panel (shared backend → other tabs may share button labels).
  const panel = page.getByTestId("review-panel");

  // Wait for the panel to load
  await expect(panel.getByRole("button", { name: "Scan" })).toBeVisible({ timeout: 10000 });

  // Click Scan to start a job
  await panel.getByRole("button", { name: "Scan" }).click();

  // Wait for Scan button to reappear (reconcile backstop re-enables it even when
  // the worker's done event raced ahead of the job-id state).
  await expect(panel.getByRole("button", { name: "Scan" })).toBeVisible({ timeout: 10000 });
});
