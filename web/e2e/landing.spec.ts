import AxeBuilder from "@axe-core/playwright";
import { expect, test } from "@playwright/test";

const seriousViolations = async (page: import("@playwright/test").Page) => {
  const results = await new AxeBuilder({ page }).analyze();
  return results.violations.filter((v) => v.impact === "serious" || v.impact === "critical").map((v) => `${v.id}: ${v.help}`);
};

test.describe("landing without JavaScript", () => {
  test.use({ javaScriptEnabled: false });

  test("reads as a complete article", async ({ page }) => {
    await page.goto("/");
    await expect(page.locator("h1")).toHaveText("Every finding, proven before a doctor sees it.");
    for (const heading of ["No location, no finding.", "It has to point.", "Anyone can draw a heatmap. We test ours.", "Eight small shocks.", "Two lines of sight.", "The notes testify too.", "A sentence that cannot lie.", "Every step leaves a receipt.", "We test our own warnings.", "What it will not do."]) {
      await expect(page.getByRole("heading", { name: heading })).toBeVisible();
    }
    await expect(page.locator("img.film-hero")).toBeVisible();
    await expect(page.getByText("Decision support only.")).toBeVisible();
  });

  test("uses no em or en dashes anywhere in visible copy", async ({ page }) => {
    await page.goto("/");
    const text = await page.evaluate(() => document.body.innerText);
    expect(text).not.toMatch(/[–—]/);
  });

});

test("prerendered markup has no serious accessibility violations", async ({ page }) => {
  // axe injects itself as a script, so instead of disabling JavaScript this blocks the page's
  // bundles and clears the classes the inline head script set: what remains is the no-JS page.
  await page.route("**/assets/*.js", (route) => route.abort());
  await page.goto("/");
  await page.evaluate(() => (document.documentElement.className = ""));
  expect(await seriousViolations(page)).toEqual([]);
});

test.describe("landing with the story engine", () => {
  test("boots the WebGL story without errors and keeps the hero readable", async ({ page }) => {
    const errors: string[] = [];
    page.on("pageerror", (e) => errors.push(e.message));
    page.on("console", (m) => m.type() === "error" && errors.push(m.text()));
    await page.goto("/");
    await page.waitForFunction(() => /story-gl|story-failed/.test(document.documentElement.className));
    await expect(page.locator("html")).toHaveClass(/story-gl/);
    await expect(page.locator("h1")).toBeVisible();
    expect(errors).toEqual([]);
  });

  test("has no serious accessibility violations after hydration", async ({ page }) => {
    await page.goto("/");
    await page.waitForFunction(() => /story-gl|story-failed/.test(document.documentElement.className));
    expect(await seriousViolations(page)).toEqual([]);
  });

  test("the ledger demo detects tampering in the browser", async ({ page }) => {
    await page.goto("/");
    const status = page.getByRole("status").filter({ hasText: /Chain verified|Verification failed|Computing/ });
    await expect(status).toHaveText(/Chain verified/);
    await page.getByLabel("Doctor decision, entry #4").fill("reject");
    await expect(status).toHaveText(/Verification failed at entry 4/);
    await page.getByRole("button", { name: "Restore the original values" }).click();
    await expect(status).toHaveText(/Chain verified/);
  });
});

test.describe("phone and reduced motion", () => {
  test("phones get the article layout without horizontal scroll", async ({ browser }) => {
    const ctx = await browser.newContext({ viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true });
    const page = await ctx.newPage();
    await page.goto("/");
    await expect(page.locator("html")).not.toHaveClass(/scrolly/);
    expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(390);
    await ctx.close();
  });

  test("reduced motion gets the article layout", async ({ browser }) => {
    const ctx = await browser.newContext({ reducedMotion: "reduce", viewport: { width: 1440, height: 900 } });
    const page = await ctx.newPage();
    await page.goto("/");
    await expect(page.locator("html")).toHaveClass(/reduce-motion/);
    await expect(page.locator("html")).not.toHaveClass(/scrolly/);
    await ctx.close();
  });
});
