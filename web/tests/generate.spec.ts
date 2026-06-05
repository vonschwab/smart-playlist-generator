import { test, expect } from "@playwright/test";

test("generate loop renders tracks, stats, and logs", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByText("Playlist Generator")).toBeVisible();

  // Fill seed input and click Generate
  await page.locator('input[placeholder="Acetone, Mazzy Star"]').fill("Acetone");
  await page.getByRole("button", { name: /Generate/ }).click();

  // Fake worker returns a 2-track playlist
  await expect(page.getByTestId("track-table")).toBeVisible({ timeout: 15000 });
  await expect(page.getByText("Sundown")).toBeVisible();
  await expect(page.getByText("Taxi")).toBeVisible();

  // Quality stats rendered
  await expect(page.getByText("distinct artists")).toBeVisible();

  // Logs streamed over websocket
  await expect(page.getByTestId("log-panel")).toContainText("fake: starting", { timeout: 5000 });
});
