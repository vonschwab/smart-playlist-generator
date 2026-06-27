import { test, expect, type Page } from "@playwright/test";

const WIDTHS = [375, 768, 1024, 1440] as const;

async function generate(page: Page) {
  await page.getByTestId("seed-input").fill("Acetone");
  await page.getByRole("button", { name: /Generate/ }).click();
  await expect(page.getByTestId("track-table")).toBeVisible({ timeout: 15000 });
}

for (const width of WIDTHS) {
  test(`responsive @ ${width}px: no collapse, no overflow`, async ({ page }) => {
    await page.setViewportSize({ width, height: 900 });
    await page.goto("/");
    await expect(page.getByText("Playlist Generator")).toBeVisible();

    // 1) The Artist input must never collapse (the original bug).
    const seed = page.getByTestId("seed-input");
    await expect(seed).toBeVisible();
    const box = await seed.boundingBox();
    expect(box, "seed input has a layout box").not.toBeNull();
    expect(box!.width, `seed input width @ ${width}px`).toBeGreaterThan(120);

    // 2) No horizontal overflow of the document.
    const overflow = await page.evaluate(
      () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
    );
    expect(overflow, `horizontal overflow @ ${width}px`).toBeLessThanOrEqual(1);

    // 3) Shell mode matches the breakpoint.
    if (width < 1024) {
      await expect(page.getByTestId("mobile-tabbar")).toBeVisible();
    } else {
      await expect(page.getByTestId("mobile-tabbar")).toHaveCount(0);
    }

    // 4) Generate works and the table renders without overflowing either.
    await generate(page);
    const overflowAfter = await page.evaluate(
      () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
    );
    expect(overflowAfter, `overflow after generate @ ${width}px`).toBeLessThanOrEqual(1);

    await page.screenshot({ path: `test-results/responsive/${width}.png`, fullPage: true });
  });
}

test("mobile tab bar swaps regions @ 375px", async ({ page }) => {
  await page.setViewportSize({ width: 375, height: 900 });
  await page.goto("/");
  await page.getByTestId("tab-mobile-jobs").click();
  await expect(page.getByTestId("jobs-panel")).toBeVisible();
  await page.getByTestId("tab-mobile-logs").click();
  await expect(page.getByTestId("log-panel")).toBeVisible();
});
