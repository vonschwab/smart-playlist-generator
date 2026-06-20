import { test, expect } from "@playwright/test";

test("Genre Review tab renders album row and Accept updates decided count", async ({ page }) => {
  await page.goto("/");

  // Navigate to the Genre Review tab in the AdvancedPanel
  await page.getByTestId("tab-review").click();

  // Scope all assertions to the review panel
  const panel = page.getByTestId("review-panel");

  // Header counts loaded from fake worker: 1 pending · 0 decided
  await expect(panel.getByText("1 pending · 0 decided")).toBeVisible({ timeout: 10000 });

  // Album row is visible
  await expect(panel.getByText("Slowdive – Souvlaki")).toBeVisible();

  // Expand the album row by clicking it
  await panel.getByText("Slowdive – Souvlaki").click();

  // Proposed genre chip shows "shoegaze"
  await expect(panel.getByText(/shoegaze/)).toBeVisible();

  // Click Accept
  await panel.getByRole("button", { name: "Accept (A)" }).first().click();

  // Optimistic update: pending drops to 0, decided becomes 1
  await expect(panel.getByText("0 pending · 1 decided")).toBeVisible({ timeout: 5000 });
});

test("Publish decided button appears when decided_albums > 0", async ({ page }) => {
  await page.goto("/");

  await page.getByTestId("tab-review").click();
  const panel = page.getByTestId("review-panel");

  // Wait for panel to load
  await expect(panel.getByText("Slowdive – Souvlaki")).toBeVisible({ timeout: 10000 });

  // Expand and accept to create a decided album
  await panel.getByText("Slowdive – Souvlaki").click();
  await panel.getByRole("button", { name: "Accept (A)" }).first().click();

  // Publish decided button appears
  await expect(panel.getByRole("button", { name: /Publish decided/ })).toBeVisible({ timeout: 5000 });
});
