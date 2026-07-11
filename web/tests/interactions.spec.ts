import { test, expect } from "@playwright/test";
import type { Page } from "@playwright/test";

async function generate(page: Page) {
  await page.goto("/");
  await page.getByTestId("seed-input").fill("Acetone");
  await page.getByRole("button", { name: /Generate/ }).click();
  await expect(page.getByTestId("track-table")).toBeVisible({ timeout: 15000 });
}

test("play button mounts the mini-player", async ({ page }) => {
  await generate(page);
  await page.getByTestId("play-btn").first().click();
  await expect(page.getByTestId("mini-player")).toBeVisible();
});

test("context menu shows expected items", async ({ page }) => {
  await generate(page);
  // Open context menu via the kebab button on the second row (index 1)
  const row = page.getByTestId("track-table").locator("tbody tr").nth(1);
  await row.hover();
  await page.getByTestId("kebab-btn").nth(1).click();
  await expect(page.getByText("Blacklist this track")).toBeVisible();
  await expect(page.getByText(/Edit genres for album/)).toBeVisible();
});

test("blacklist dims the row", async ({ page }) => {
  await generate(page);
  // Open context menu and click blacklist via the kebab button on the second row
  const row = page.getByTestId("track-table").locator("tbody tr").nth(1);
  await row.hover();
  await page.getByTestId("kebab-btn").nth(1).click();
  await page.getByText("Blacklist this track").click();
  await expect(page.getByText("blacklisted").first()).toBeVisible();
});

test("M3U8 export opens a rename dialog and downloads with the chosen name", async ({ page }) => {
  await generate(page);
  await page.getByTestId("export-m3u8").click();

  // A rename dialog appears, pre-filled with "<first track artist> — <YYYY-MM-DD>".
  await expect(page.getByTestId("export-m3u8-dialog")).toBeVisible();
  await expect(page.getByTestId("m3u8-name")).toHaveValue(/^Acetone — \d{4}-\d{2}-\d{2}$/);

  // The user can rename before exporting; the download uses the entered name.
  await page.getByTestId("m3u8-name").fill("My Mix");
  const downloadPromise = page.waitForEvent("download");
  await page.getByTestId("m3u8-download").click();
  const download = await downloadPromise;
  expect(download.suggestedFilename()).toBe("My Mix.m3u8");
});
