// Browser smoke test: the page must load its CSS and JavaScript and react to the language toggle.
// Usage: node scripts/smoke-browser.mjs ORIGIN [BASE_PATH]
// The browser is, in order: KV_SMOKE_BROWSER_PATH, the Playwright-managed Chromium, then the
// installed Google Chrome. Without one, the check is skipped unless KV_SMOKE_BROWSER=require.
import { chromium } from "playwright-core";

const [origin, basePath = ""] = process.argv.slice(2);
if (!origin) {
  console.error("Usage: node scripts/smoke-browser.mjs ORIGIN [BASE_PATH]");
  process.exit(2);
}

async function launchBrowser() {
  const attempts = [
    ...(process.env.KV_SMOKE_BROWSER_PATH ? [{ executablePath: process.env.KV_SMOKE_BROWSER_PATH }] : []),
    {},
    { channel: "chrome" },
  ];
  for (const options of attempts) {
    try {
      return await chromium.launch(options);
    } catch {
      // Try the next browser.
    }
  }
  return null;
}

const browser = await launchBrowser();
if (!browser) {
  if (process.env.KV_SMOKE_BROWSER === "require") {
    console.error("FAIL: no browser available for the browser check.");
    process.exit(1);
  }
  console.log("Browser check skipped: no browser found.");
  process.exit(0);
}

const problems = [];
try {
  const context = await browser.newContext({ locale: "fr-FR" });
  const page = await context.newPage();
  page.on("pageerror", (error) => problems.push(`page error: ${error.message}`));
  page.on("requestfailed", (request) => problems.push(`request failed: ${request.url()} (${request.failure()?.errorText})`));
  page.on("response", (response) => {
    if (response.status() >= 400) problems.push(`HTTP ${response.status()}: ${response.url()}`);
  });

  // The trailing slash exercises the redirect a reverse proxy may see.
  await page.goto(`${origin}${basePath}/`, { waitUntil: "networkidle" });
  const pathname = () => new URL(page.url()).pathname;
  if (!pathname().startsWith(basePath || "/")) problems.push(`landed on ${page.url()}, outside ${basePath}`);
  await page.locator('section[aria-label="Résultats"]').waitFor();

  // Stylesheet applied.
  const headerDisplay = await page.locator(".header").evaluate((el) => getComputedStyle(el).display);
  if (headerDisplay !== "flex") problems.push(`stylesheet not applied (.header display is ${headerDisplay})`);

  // JavaScript hydrated: the language toggle switches the page to English without a reload.
  await page.getByRole("button", { name: "Switch to English" }).click();
  await page.locator('section[aria-label="Results"]').waitFor({ timeout: 10_000 });
  if ((await page.getAttribute("html", "lang")) !== "en") problems.push("<html lang> did not switch to en");

  // The cookie set by the client makes the server render English, still under the base path.
  await page.reload({ waitUntil: "networkidle" });
  await page.locator('section[aria-label="Results"]').waitFor();
  await page.getByRole("button", { name: "Passer en français" }).click();
  await page.locator('section[aria-label="Résultats"]').waitFor({ timeout: 10_000 });
  if (!pathname().startsWith(basePath || "/")) problems.push(`URL left the base path: ${page.url()}`);
} catch (error) {
  problems.push(String(error));
} finally {
  await browser.close();
}

if (problems.length) {
  console.error(`FAIL: browser check for ${origin}${basePath}/`);
  for (const p of problems) console.error(`  ${p}`);
  process.exit(1);
}
console.log(`Browser check passed for ${origin}${basePath}/ (CSS applied, JavaScript hydrated, FR/EN toggle).`);
