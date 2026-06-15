import { test, expect } from "@playwright/test";

// Regression: seeds mode must send seed_tracks (display strings) parallel to
// seed_track_ids. The worker gates the seeds branch on seed_tracks and the
// generator zips the two lists, so sending IDs alone fails generation with a
// cryptic "Invalid mode or missing parameters" error.
test("seeds mode sends seed_tracks parallel to seed_track_ids", async ({ page }) => {
  await page.addInitScript(() => {
    localStorage.setItem("pg_mode", JSON.stringify("seeds"));
    localStorage.setItem(
      "pg_seed_tracks",
      JSON.stringify([
        { track_id: "k0", title: "Sundown", artist: "Acetone", album: "Cindy", genres: ["slowcore"], duration_ms: 200000, file_path: "/0.flac" },
        { track_id: "k1", title: "Taxi", artist: "Mazzy Star", album: "So Tonight", genres: ["dreampop"], duration_ms: 210000, file_path: "/1.flac" },
      ]),
    );
  });

  await page.goto("/");

  const reqPromise = page.waitForRequest(
    (r) => r.url().includes("/api/generate") && r.method() === "POST",
  );
  await page.getByRole("button", { name: /Generate/ }).click();
  const body = JSON.parse((await reqPromise).postData() ?? "{}");

  expect(body.mode).toBe("seeds");
  expect(body.seed_track_ids).toEqual(["k0", "k1"]);
  // Parallel arrays — same length is the backend contract
  expect(body.seed_tracks).toHaveLength(body.seed_track_ids.length);
  expect(body.seed_tracks[0]).toContain("Sundown");
});
