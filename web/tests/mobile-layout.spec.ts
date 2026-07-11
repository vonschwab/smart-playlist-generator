import { test, expect, type Page } from "@playwright/test";

// S5 gate: after generating on a phone, the playlist owns the viewport.

test.use({
  viewport: { width: 390, height: 844 },
  deviceScaleFactor: 2,
  isMobile: true,
  hasTouch: true,
});

async function generate(page: Page) {
  await page.goto("/");
  await page.getByTestId("seed-input").fill("Acetone");
  await page.getByRole("button", { name: "▸ Generate" }).click();
  await expect(page.getByTestId("track-table")).toBeVisible({ timeout: 15000 });
}

test("controls collapse after generation so the playlist owns the screen", async ({ page }) => {
  await generate(page);
  await expect(page.getByTestId("controls-summary")).toBeVisible();
  await expect(page.getByTestId("seed-input")).toBeHidden();
  // The playlist starts in the top third of the screen, not below the fold.
  const table = await page.getByTestId("track-table").boundingBox();
  expect(table!.y).toBeLessThan(300);
});

test("the summary bar reopens the controls with form state intact", async ({ page }) => {
  await generate(page);
  await page.getByTestId("controls-summary-open").tap();
  const seed = page.getByTestId("seed-input");
  await expect(seed).toBeVisible();
  await expect(seed).toHaveValue("Acetone");
  await expect(page.getByTestId("controls-summary")).toHaveCount(0);
});

test("desktop keeps controls expanded after generation", async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 900 });
  await generate(page);
  await expect(page.getByTestId("seed-input")).toBeVisible();
  await expect(page.getByTestId("controls-summary")).toHaveCount(0);
});
