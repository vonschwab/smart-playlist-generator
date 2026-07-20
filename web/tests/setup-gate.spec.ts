import { test, expect } from "@playwright/test";

test("needs_setup shows the setup page instead of the generator", async ({ page }) => {
  await page.route("**/api/setup/status", (route) =>
    route.fulfill({ json: { state: "needs_setup", config_path: "/home/u/.config/mixarc/config.yaml",
                           config_exists: false, music_directory: null, db_path: null,
                           track_count: null, detail: "No config.yaml — run setup." } }));
  await page.goto("/");
  await expect(page.getByTestId("setup-page")).toBeVisible();
  await expect(page.getByText(/No config\.yaml/)).toBeVisible();
});

test("ready state shows the normal app", async ({ page }) => {
  await page.route("**/api/setup/status", (route) =>
    route.fulfill({ json: { state: "ready", config_path: "x", config_exists: true,
                           music_directory: "/m", db_path: "/d", track_count: 12, detail: "Ready." } }));
  await page.goto("/");
  await expect(page.getByTestId("setup-page")).toHaveCount(0);
});
