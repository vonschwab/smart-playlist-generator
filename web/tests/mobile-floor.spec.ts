import { test, expect, type Page } from "@playwright/test";

// Gate for docs/UI_UX_DISCIPLINE.md rules T1/T2/V3/V4 at iPhone size:
// every target >=24px (WCAG 2.5.8), primaries >=44px (HIG), fields >=16px
// font (iOS focus-zoom), dialogs/popovers fit the viewport, row actions
// visible on touch. Runs with touch emulation so pointer:coarse applies.

test.use({
  viewport: { width: 390, height: 844 },
  deviceScaleFactor: 2,
  isMobile: true,
  hasTouch: true,
});

const FLOOR = 24;
const PRIMARY = 44;

async function generate(page: Page) {
  await page.goto("/");
  await page.getByTestId("seed-input").fill("Acetone");
  await page.getByRole("button", { name: "▸ Generate" }).click();
  await expect(page.getByTestId("track-table")).toBeVisible({ timeout: 15000 });
}

// Undersized interactive elements. The effective target is the wrapping
// <label> when present — WCAG measures the clickable area, not the glyph.
async function undersized(page: Page, scopeSel?: string) {
  return page.evaluate(
    ({ floor, scope }) => {
      const root: ParentNode = scope ? (document.querySelector(scope) ?? document) : document;
      const els = Array.from(
        root.querySelectorAll<HTMLElement>('button, a[href], input, select, textarea, [role="button"]'),
      );
      const seen = new Set<Element>();
      const bad: Array<{ desc: string; w: number; h: number }> = [];
      for (const el of els) {
        const target = el.closest("label") ?? el;
        if (seen.has(target)) continue;
        seen.add(target);
        const r = target.getBoundingClientRect();
        if (r.width === 0 || r.height === 0) continue;
        if (r.width < floor || r.height < floor) {
          const label =
            el.getAttribute("data-testid") ||
            (el.textContent || (el as HTMLInputElement).placeholder || el.getAttribute("title") || el.tagName).trim().slice(0, 30);
          bad.push({ desc: label, w: Math.round(r.width), h: Math.round(r.height) });
        }
      }
      return bad;
    },
    { floor: FLOOR, scope: scopeSel ?? null },
  );
}

async function fieldsUnder16px(page: Page) {
  return page.evaluate(() => {
    const bad: Array<{ desc: string; px: number }> = [];
    for (const el of Array.from(document.querySelectorAll<HTMLElement>("input, select, textarea"))) {
      const r = el.getBoundingClientRect();
      if (r.width === 0 || r.height === 0) continue;
      const px = parseFloat(getComputedStyle(el).fontSize);
      if (px < 16) {
        bad.push({
          desc: el.getAttribute("data-testid") || (el as HTMLInputElement).placeholder || el.tagName,
          px,
        });
      }
    }
    return bad;
  });
}

test("initial view: every target >=24px, every field >=16px font", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByText("Playlist Generator")).toBeVisible();
  expect(await undersized(page)).toEqual([]);
  expect(await fieldsUnder16px(page)).toEqual([]);
});

test("primary controls meet the 44px floor", async ({ page }) => {
  await page.goto("/");
  const primaries = [
    page.getByRole("button", { name: "▸ Generate" }),
    page.getByRole("button", { name: "↻ New Seeds" }),
    page.getByRole("button", { name: "generate", exact: true }),
    page.getByRole("button", { name: "tools", exact: true }),
    page.getByTestId("tab-mobile-playlist"),
    page.getByTestId("tab-mobile-jobs"),
    page.getByTestId("tab-mobile-diag"),
    page.getByTestId("tab-mobile-logs"),
  ];
  for (const loc of primaries) {
    const b = await loc.boundingBox();
    expect(b, String(loc)).not.toBeNull();
    expect(b!.height, String(loc)).toBeGreaterThanOrEqual(PRIMARY);
  }
});

test("row actions are visible and tappable after generation", async ({ page }) => {
  await generate(page);
  const kebab = page.getByTestId("kebab-btn").first();
  const opacity = await kebab.evaluate((el) => parseFloat(getComputedStyle(el).opacity));
  expect(opacity, "kebab visible on touch").toBeGreaterThanOrEqual(0.5);
  for (const tid of ["kebab-btn", "play-btn"]) {
    const b = await page.getByTestId(tid).first().boundingBox();
    expect(b!.width, tid).toBeGreaterThanOrEqual(FLOOR);
    expect(b!.height, tid).toBeGreaterThanOrEqual(FLOOR);
  }
  expect(await undersized(page)).toEqual([]);
});

test("export dialog fits the viewport with tappable controls", async ({ page }) => {
  await generate(page);
  await page.getByTestId("export-m3u8").tap();
  const dlg = page.getByTestId("export-m3u8-dialog");
  await expect(dlg).toBeVisible();
  const db = await dlg.boundingBox();
  expect(db!.height).toBeLessThanOrEqual(844 * 0.9);
  for (const name of ["Cancel", "Download"]) {
    const b = await dlg.getByRole("button", { name }).boundingBox();
    expect(b!.height, name).toBeGreaterThanOrEqual(PRIMARY);
  }
  expect(await undersized(page, '[data-testid="export-m3u8-dialog"]')).toEqual([]);
});

for (const width of [390, 320]) {
  test(`style popover stays within the viewport @ ${width}px`, async ({ page }) => {
    await page.setViewportSize({ width, height: 844 });
    await page.goto("/");
    await page.getByRole("button", { name: /style/i }).tap();
    const card = page.locator("#style-popover-card");
    await expect(card).toBeVisible();
    const b = (await card.boundingBox())!;
    expect(b.x).toBeGreaterThanOrEqual(0);
    expect(b.x + b.width).toBeLessThanOrEqual(width);
    expect(b.y + b.height).toBeLessThanOrEqual(844);
  });
}

test("tools view does not scroll horizontally on a phone", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", { name: "tools", exact: true }).tap();
  await expect(page.getByText("Analyze Library")).toBeVisible();
  // The panel is its own scroller, so document-level overflow checks can't
  // see a sideways-scrolling Tools grid — measure the scroller itself.
  const overflow = await page.evaluate(() => {
    let worst = 0;
    for (const el of Array.from(document.querySelectorAll<HTMLElement>("div"))) {
      if (el.scrollWidth > el.clientWidth + 1 && el.clientWidth > 0) {
        worst = Math.max(worst, el.scrollWidth - el.clientWidth);
      }
    }
    return worst;
  });
  expect(overflow).toBeLessThanOrEqual(1);
});

test("mini-player transport meets the floor with a padded seek bar", async ({ page }) => {
  await generate(page);
  await page.getByTestId("play-btn").first().tap();
  const mp = page.getByTestId("mini-player");
  await expect(mp).toBeVisible();
  for (const title of ["Previous", "Next", /Play|Pause/]) {
    const b = await mp.getByTitle(title).boundingBox();
    expect(b!.width, String(title)).toBeGreaterThanOrEqual(PRIMARY);
    expect(b!.height, String(title)).toBeGreaterThanOrEqual(PRIMARY);
  }
  const seek = await mp.getByTestId("seek-bar").boundingBox();
  expect(seek, "padded seek hit area").not.toBeNull();
  expect(seek!.height).toBeGreaterThanOrEqual(16);
});
