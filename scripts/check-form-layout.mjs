// Production-browser regression for field spacing, persistent results and mobile footer access.
// Usage: node scripts/check-form-layout.mjs ORIGIN [SCREENSHOT_DIRECTORY]
import assert from "node:assert/strict";
import { mkdir } from "node:fs/promises";
import { chromium } from "playwright-core";

const [origin, screenshots] = process.argv.slice(2);
if (!origin) throw new Error("Usage: node scripts/check-form-layout.mjs ORIGIN [SCREENSHOT_DIRECTORY]");
if (screenshots) await mkdir(screenshots, { recursive: true });
const browser = await chromium.launch({ channel: "chrome" });
const failures = [];
try {
  for (const width of [1440, 390]) for (const lang of ["fr", "en"]) for (const theme of ["light", "dark"]) {
    const name = `${width}-${lang}-${theme}`;
    const height = width === 1440 ? 900 : 844;
    const context = await browser.newContext({ viewport: { width, height }, colorScheme: theme });
    try {
      await context.addCookies([
        { name: "kv-lang", value: lang, url: origin },
        { name: "kv-theme", value: theme, url: origin },
      ]);
      const page = await context.newPage();
      await page.goto(`${origin}/?model=qwen3-32b&gpu=h100&n=2`, { waitUntil: "networkidle" });
      const box = (locator) => locator.boundingBox();
      const weights = page.getByRole("radiogroup", { name: lang === "fr" ? "Format des poids" : "Weight format", exact: true }).locator("..");
      const kv = page.getByRole("radiogroup", { name: lang === "fr" ? "Format du KV cache" : "KV cache format", exact: true }).locator("..");
      const windowField = page.locator("#ctx").locator("../..");
      const advanced = page.locator(".advanced > summary");
      if (width === 1440) await page.locator(".form").evaluate((el) => { el.scrollTop = el.scrollHeight; });
      if (screenshots) await page.screenshot({ path: `${screenshots}/${name}-fields.png` });
      for (const [above, below] of [[weights, kv], [kv, windowField], [windowField, advanced]]) {
        const a = await box(above), b = await box(below);
        assert.ok(b.y - (a.y + a.height) >= 16, `${name}: field gap is ${b.y - a.y - a.height}px`);
      }
      await advanced.click();
      const reserve = page.locator("#reserve");
      await reserve.fill("15");
      await reserve.press("ArrowRight");
      assert.equal(await reserve.inputValue(), "16");
      const reserveField = reserve.locator("..");
      const visibleMemory = page.getByLabel(lang === "fr" ? /Mémoire visible par GPU/ : /Visible memory per GPU/).locator("..");
      const a = await box(reserveField), b = await box(visibleMemory);
      assert.ok(b.y - a.y - a.height >= 12, `${name}: advanced controls run together`);
      assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), `${name}: horizontal overflow`);
      if (width === 1440) {
        const kpis = await box(page.locator("#results-kpis"));
        const summary = await box(page.locator(".result-summary"));
        assert.ok((kpis && kpis.y >= 0 && kpis.y + kpis.height <= height) || (summary && summary.y >= 0 && summary.y + summary.height <= height), `${name}: results out of view`);
      } else {
        await page.locator(".assumptions > summary").click();
        await page.evaluate(() => window.scrollTo(0, document.documentElement.scrollHeight));
        const footer = await box(page.locator(".assumptions")), bar = await box(page.locator(".result-summary"));
        assert.ok(footer.y + footer.height <= bar.y, `${name}: footer hidden behind the summary`);
        if (screenshots) await page.screenshot({ path: `${screenshots}/${name}-footer.png` });
        await reserve.focus();
        const r = await box(reserve), s = await box(page.locator(".result-summary"));
        assert.ok(r.y >= 0 && r.y + r.height <= s.y, `${name}: focused reserve hidden behind the summary`);
        await page.getByRole("button", { name: lang === "fr" ? "Switch to English" : "Passer en français" }).focus();
        for (let i = 0; i < 100; i++) {
          await page.keyboard.press("Tab");
          await page.evaluate(() => new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve))));
          const focus = await page.evaluate(() => {
            const el = document.activeElement;
            if (!el || el === document.body || el.classList.contains("result-summary")) return null;
            const rect = el.getBoundingClientRect();
            const bar = document.querySelector(".result-summary").getBoundingClientRect();
            return { label: el.getAttribute("aria-label") || el.id || el.textContent.trim(), top: rect.top, bottom: rect.bottom, limit: bar.top };
          });
          const visible = !focus || (focus.top >= 0 && focus.bottom <= focus.limit + 1);
          if (!visible && screenshots) await page.screenshot({ path: `${screenshots}/${name}-hidden-focus.png` });
          assert.ok(visible, `${name}: keyboard focus obscured: ${JSON.stringify(focus)}`);
        }
      }
      console.log(`PASS ${name}: field gaps, advanced controls, visible results, footer, no overflow`);
    } catch (error) {
      failures.push(`${name}: ${error.message}`);
    } finally {
      await context.close();
    }
  }
} finally {
  await browser.close();
}
if (failures.length) {
  console.error(failures.join("\n"));
  process.exitCode = 1;
}
