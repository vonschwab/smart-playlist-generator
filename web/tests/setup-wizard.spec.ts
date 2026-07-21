import { test, expect } from "@playwright/test";

test("wizard walkthrough writes config and reaches analyze", async ({ page }) => {
  await page.route("**/api/setup/status", (r) => r.fulfill({ json: {
    state: "needs_setup", config_path: "/c", config_exists: false, music_directory: null,
    db_path: null, track_count: null, detail: "", checks: [{ id: "python_version", status: "pass", summary: "3.13", fix_hint: null }] } }));
  await page.route("**/api/setup/browse**", (r) => r.fulfill({ json: {
    path: "/home/u/Music", parent: "/home/u", entries: [{ name: "FLAC", path: "/home/u/Music/FLAC", audio_count: 8412 }], is_music_dir: true } }));
  await page.route("**/api/setup/config", (r) => r.fulfill({ json: { ok: true, config_path: "/c/config.yaml" } }));
  await page.route("**/api/tools/analyze", (r) => r.fulfill({ json: { job_id: "job1" } }));

  await page.goto("/");
  await expect(page.getByTestId("wizard-rail")).toBeVisible();
  await page.getByTestId("wizard-next").click(); // welcome -> environment
  await page.getByTestId("wizard-next").click(); // -> music
  // getByText would also match the instructional prose ("...then click
  // \"Use this folder\"."), which quotes the button's own label — scope to
  // the button role to disambiguate.
  await page.getByRole("button", { name: "Use this folder" }).click();
  await page.getByTestId("wizard-next").click(); // -> services
  await page.getByTestId("wizard-next").click(); // -> genre
  await page.getByTestId("wizard-next").click(); // -> review
  await page.getByTestId("wizard-write-config").click(); // writes config -> analyze
  await expect(page.getByTestId("step-analyze")).toBeVisible();
});
