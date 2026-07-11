import { describe, it, expect } from "vitest";
import { readdirSync, readFileSync } from "fs";
import { join, resolve } from "path";

// Design-system gates (docs/UI_UX_DISCIPLINE.md C1/C2/C4). These are the
// enforcement half of the 2026-07-09 audit: theme tokens are the only
// colors, the type scale is the named scale, and every token pair that
// carries text passes WCAG contrast against the actual hex values.

// vitest runs with cwd = web/ (vite.config.ts lives there).
const SRC = resolve(process.cwd(), "src");

function sourceFiles(dir: string): string[] {
  const out: string[] = [];
  for (const e of readdirSync(dir, { withFileTypes: true })) {
    const p = join(dir, e.name);
    if (e.isDirectory()) out.push(...sourceFiles(p));
    else if (/\.(tsx?|css)$/.test(e.name) && !/\.test\./.test(e.name) && e.name !== "index.css") {
      out.push(p);
    }
  }
  return out;
}

describe("token adherence (C1)", () => {
  it("components use theme tokens, never raw hex colors", () => {
    const offenders: string[] = [];
    for (const f of sourceFiles(SRC)) {
      const text = readFileSync(f, "utf8");
      for (const [i, line] of text.split("\n").entries()) {
        const m = line.match(/#[0-9a-fA-F]{3,8}\b/g);
        if (m) offenders.push(`${f.replace(SRC, "")}:${i + 1} ${m.join(",")}`);
      }
    }
    expect(offenders, `raw hex outside index.css @theme:\n${offenders.join("\n")}`).toEqual([]);
  });

  it("components use the named type scale, never arbitrary text-[Npx]", () => {
    const offenders: string[] = [];
    for (const f of sourceFiles(SRC)) {
      const text = readFileSync(f, "utf8");
      for (const [i, line] of text.split("\n").entries()) {
        const m = line.match(/text-\[\d+px\]/g);
        if (m) offenders.push(`${f.replace(SRC, "")}:${i + 1} ${m.join(",")}`);
      }
    }
    expect(offenders, `arbitrary font sizes:\n${offenders.join("\n")}`).toEqual([]);
  });

  it("components do not use stock Tailwind palette colors outside the theme", () => {
    const offenders: string[] = [];
    const palette =
      /(?:text|bg|border)-(?:red|orange|amber|yellow|lime|green|emerald|teal|cyan|sky|blue|indigo|violet|purple|fuchsia|pink|rose|slate|gray|zinc|neutral|stone)-\d{2,3}/g;
    for (const f of sourceFiles(SRC)) {
      const text = readFileSync(f, "utf8");
      for (const [i, line] of text.split("\n").entries()) {
        const m = line.match(palette);
        if (m) offenders.push(`${f.replace(SRC, "")}:${i + 1} ${m.join(",")}`);
      }
    }
    expect(offenders, `stock palette classes:\n${offenders.join("\n")}`).toEqual([]);
  });
});

describe("local-first assets (I2)", () => {
  it("index.css loads no remote resources", () => {
    const css = readFileSync(join(SRC, "index.css"), "utf8");
    // A render-blocking third-party fetch in a local-first app: fonts and
    // every other asset are self-hosted.
    expect(css).not.toMatch(/https?:\/\//);
  });
});

// ── WCAG contrast against the live token values ─────────────────────────────

function theme(): Record<string, string> {
  const css = readFileSync(join(SRC, "index.css"), "utf8");
  const vars: Record<string, string> = {};
  for (const m of css.matchAll(/--color-(\w+):\s*(#[0-9a-fA-F]{6})/g)) vars[m[1]] = m[2];
  return vars;
}

function luminance(hex: string): number {
  const c = [1, 3, 5].map((i) => {
    const x = parseInt(hex.slice(i, i + 2), 16) / 255;
    return x <= 0.03928 ? x / 12.92 : ((x + 0.055) / 1.055) ** 2.4;
  });
  return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
}

function contrast(a: string, b: string): number {
  const [l1, l2] = [luminance(a), luminance(b)].sort((x, y) => y - x);
  return (l1 + 0.05) / (l2 + 0.05);
}

describe("token contrast (C2)", () => {
  const t = theme();
  // Text tokens that carry copy must pass AA normal-text 4.5:1 on both surfaces.
  for (const name of ["text", "muted", "faint"]) {
    for (const surface of ["panel", "bg"]) {
      it(`${name} on ${surface} >= 4.5:1`, () => {
        expect(contrast(t[name], t[surface])).toBeGreaterThanOrEqual(4.5);
      });
    }
  }
  it("chipText on chip >= 4.5:1", () => {
    expect(contrast(t.chipText, t.chip)).toBeGreaterThanOrEqual(4.5);
  });
  // Status/accent colors render as large text or UI glyphs: 3:1 floor.
  for (const name of ["accent", "warn", "danger", "info"]) {
    it(`${name} on panel >= 3:1`, () => {
      expect(t[name], `--color-${name} must exist in @theme`).toBeDefined();
      expect(contrast(t[name], t.panel)).toBeGreaterThanOrEqual(3);
    });
  }
});
