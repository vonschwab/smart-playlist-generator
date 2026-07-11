import { test, expect, type Page } from "@playwright/test";
import { mkdirSync, writeFileSync } from "fs";
import { resolve } from "path";
import { fileURLToPath } from "url";

// Mobile audit sweep: screenshots every mobile surface at iPhone size and
// measures rendered touch-target / micro-type violations. Not a pass/fail
// gate (assertions are minimal) — it produces evidence under
// test-results/mobile-audit/ for design review.

// Anchor output to web/test-results (gitignored) regardless of process cwd.
const testsDir = fileURLToPath(new URL(".", import.meta.url));
const OUT = resolve(
  testsDir,
  "..",
  "test-results",
  (process.env.PW_BROWSER || "chromium") === "webkit"
    ? "mobile-audit-webkit"
    : "mobile-audit",
);

// PW_BROWSER=webkit runs the sweep on the Safari engine (closest to iPhone).
const browserName = (process.env.PW_BROWSER as "chromium" | "webkit") || "chromium";

test.use({
  browserName,
  viewport: { width: 390, height: 844 }, // iPhone 14/15 logical points
  deviceScaleFactor: 2,
  isMobile: browserName === "chromium", // webkit doesn't support isMobile on desktop builds
  hasTouch: true,
});

async function shot(page: Page, name: string) {
  await page.waitForTimeout(250);
  await page.screenshot({ path: `${OUT}/${name}.png` });
}

// Rendered-size audit: interactive elements smaller than 44x44 CSS px (Apple
// HIG / WCAG 2.5.8-ish floor) and visible text under 12px.
async function measure(page: Page, label: string) {
  return page.evaluate((lbl) => {
    const clickable = Array.from(
      document.querySelectorAll<HTMLElement>(
        'button, a[href], input, select, textarea, [role="button"], [onclick]',
      ),
    );
    const small: Array<Record<string, unknown>> = [];
    for (const el of clickable) {
      const r = el.getBoundingClientRect();
      if (r.width === 0 || r.height === 0) continue; // hidden
      if (r.height < 44 || r.width < 44) {
        small.push({
          tag: el.tagName.toLowerCase(),
          text: (el.textContent || (el as HTMLInputElement).placeholder || "").trim().slice(0, 40),
          testid: el.getAttribute("data-testid") || undefined,
          w: Math.round(r.width),
          h: Math.round(r.height),
        });
      }
    }
    const tiny: Array<Record<string, unknown>> = [];
    const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_ELEMENT);
    while (walker.nextNode()) {
      const el = walker.currentNode as HTMLElement;
      if (!el.innerText || el.children.length > 0) continue;
      const r = el.getBoundingClientRect();
      if (r.width === 0 || r.height === 0) continue;
      const fs = parseFloat(getComputedStyle(el).fontSize);
      if (fs < 12) {
        tiny.push({ text: el.innerText.trim().slice(0, 40), px: fs });
      }
    }
    const overflow =
      document.documentElement.scrollWidth - document.documentElement.clientWidth;
    return { label: lbl, horizontalOverflowPx: overflow, smallTargets: small, tinyText: tiny };
  }, label);
}

test("mobile audit sweep @ 390x844", async ({ page }) => {
  test.setTimeout(120000);
  mkdirSync(OUT, { recursive: true });
  const measurements: unknown[] = [];

  await page.goto("/");
  await expect(page.getByText("Playlist Generator")).toBeVisible();
  await shot(page, "01-playlist-initial");
  measurements.push(await measure(page, "playlist-initial"));

  // Bottom tabs
  for (const tab of ["jobs", "diag", "logs"] as const) {
    await page.getByTestId(`tab-mobile-${tab}`).tap();
    await shot(page, `02-tab-${tab}`);
    measurements.push(await measure(page, `tab-${tab}`));
  }
  await page.getByTestId("tab-mobile-playlist").tap();

  // Generate with the fake worker and capture the result table
  await page.getByTestId("seed-input").fill("Acetone");
  await page.getByRole("button", { name: /Generate/ }).click();
  await expect(page.getByTestId("track-table")).toBeVisible({ timeout: 15000 });
  await shot(page, "03-playlist-generated");
  measurements.push(await measure(page, "playlist-generated"));

  // Center Tools tab, if present in the mobile path
  const tools = page.getByRole("button", { name: /^Tools$/i }).first();
  if (await tools.isVisible().catch(() => false)) {
    await tools.tap();
    await shot(page, "04-tools");
    measurements.push(await measure(page, "tools"));
    const gen = page.getByRole("button", { name: /^Generate$/i }).first();
    if (await gen.isVisible().catch(() => false)) await gen.tap();
  }

  // Any export/dialog affordance visible post-generation
  const exportBtn = page
    .getByRole("button", { name: /export|m3u8|plex/i })
    .first();
  if (await exportBtn.isVisible().catch(() => false)) {
    await exportBtn.tap();
    await shot(page, "05-export-dialog");
    measurements.push(await measure(page, "export-dialog"));
    await page.keyboard.press("Escape");
  }

  writeFileSync(`${OUT}/measurements.json`, JSON.stringify(measurements, null, 2));
});
