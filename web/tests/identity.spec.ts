import { test, expect } from "@playwright/test";

// I1 (docs/UI_UX_DISCIPLINE.md): the served HTML owns its identity — real
// title, icons, theme-color, and a manifest so Add-to-Home-Screen works.

test("the served app declares its identity", async ({ page, request }) => {
  await page.goto("/");
  await expect(page).toHaveTitle("Playlist Generator");

  const themeColor = page.locator('meta[name="theme-color"]');
  await expect(themeColor).toHaveAttribute("content", "#0f1115");

  const touchIcon = await page.locator('link[rel="apple-touch-icon"]').getAttribute("href");
  expect(touchIcon).toBeTruthy();
  expect((await request.get(touchIcon!)).status()).toBe(200);

  // Self-hosted fonts are served from the dist root (I2).
  expect((await request.get("/fonts/spline-sans.woff2")).status()).toBe(200);

  const manifestHref = await page.locator('link[rel="manifest"]').getAttribute("href");
  expect(manifestHref).toBeTruthy();
  const manifest = await request.get(manifestHref!);
  expect(manifest.status()).toBe(200);
  const body = await manifest.json();
  expect(body.display).toBe("standalone");
  expect(body.name).toBe("Playlist Generator");
});
