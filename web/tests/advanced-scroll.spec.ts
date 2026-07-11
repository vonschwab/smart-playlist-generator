import { test, expect } from "@playwright/test";

// The Advanced panel's Diagnostics/Blacklist sub-tabs must scroll their own
// content on a phone, not clip it. Seed a long playlist via localStorage
// (persistence) so Diagnostics overflows the viewport.

const longPlaylist = {
  name: "Scroll Test",
  track_count: 40,
  metrics: { mean_transition: 0.8, min_transition: 0.6, p10_transition: 0.65, p90_transition: 0.95, distinct_artists: 12 },
  tracks: Array.from({ length: 40 }, (_, i) => ({
    position: i,
    rating_key: `t${i}`,
    artist: `Artist ${i}`,
    title: `Track ${i}`,
    album: `Album ${i}`,
    duration_ms: 200000,
    file_path: `/m/${i}.flac`,
    genres: ["indie"],
    transition_score: i === 0 ? null : 0.5 + (i % 5) * 0.08,
  })),
};

test.use({ viewport: { width: 390, height: 844 }, deviceScaleFactor: 2, isMobile: true, hasTouch: true });

test("the Advanced tab is labelled 'Advanced'", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByTestId("tab-mobile-diag")).toHaveText("Advanced");
});

test("Diagnostics scrolls its content instead of clipping", async ({ page }) => {
  await page.addInitScript((pl) => {
    localStorage.setItem("pg_current_playlist", JSON.stringify(pl));
  }, longPlaylist);
  await page.goto("/");

  await page.getByTestId("tab-mobile-diag").tap();
  const content = page.getByTestId("diagnostics-content");
  await expect(content).toBeVisible();

  const metrics = await content.evaluate((el) => {
    el.scrollTop = 9999;
    return { scrollTop: el.scrollTop, scrollHeight: el.scrollHeight, clientHeight: el.clientHeight };
  });
  // Content overflows the panel...
  expect(metrics.scrollHeight).toBeGreaterThan(metrics.clientHeight);
  // ...and the panel is a real scroll container (scrollTop moved off 0).
  expect(metrics.scrollTop).toBeGreaterThan(0);
  // ...bounded to the viewport, not spilling past it.
  expect(metrics.clientHeight).toBeLessThanOrEqual(844);
});
