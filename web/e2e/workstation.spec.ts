import AxeBuilder from "@axe-core/playwright";
import { expect, test } from "@playwright/test";

test.beforeEach(async ({ page }) => {
  await page.goto("/read?case=chest");
  await expect(page.locator('[data-finding] button[aria-current="true"]')).toBeVisible();
});

test("opens the sample chest case with real findings", async ({ page }) => {
  const findings = page.locator('aside[aria-label="Findings"] [data-finding]');
  await expect(findings).toHaveCount(5);
  await expect(page.getByRole("button", { name: /Pneumonia/ })).toBeVisible();
  await expect(page.getByText("Real output: TorchXRayVision DenseNet121")).toBeVisible();
});

test("selecting a finding highlights its note span within 100 ms", async ({ page }) => {
  const button = page.locator('[data-finding] button').filter({ hasText: "Pneumonia" });
  const elapsed = await button.evaluate(async (el) => {
    const t0 = performance.now();
    (el as HTMLButtonElement).click();
    await new Promise<void>((resolve) => {
      const check = () => {
        const span = Array.from(document.querySelectorAll('section[aria-label="Clinical notes"] button')).find((b) => b.textContent === "fever and productive cough");
        if (span?.className.includes("bg-pencil-yellow/15")) resolve();
        else requestAnimationFrame(check);
      };
      check();
    });
    return performance.now() - t0;
  });
  expect(elapsed).toBeLessThan(100);
});

test("keyboard map: next finding, layer toggle, shortcut sheet, palette", async ({ page }) => {
  const current = () => page.locator('[data-finding] button[aria-current="true"]').innerText();
  const before = await current();
  await page.keyboard.press("j");
  await expect.poll(current).not.toBe(before);

  const heatmap = page.getByRole("button", { name: /Heatmap/ });
  const pressed = await heatmap.getAttribute("aria-pressed");
  await page.keyboard.press("1");
  await expect(heatmap).toHaveAttribute("aria-pressed", pressed === "true" ? "false" : "true");

  await page.keyboard.press("?");
  await expect(page.getByRole("dialog", { name: "Keyboard and mouse" })).toBeVisible();
  await page.keyboard.press("Escape");

  await page.keyboard.press("Control+k");
  await expect(page.getByPlaceholder("Type a command or a finding")).toBeVisible();
});

test("accepting a sample finding explains that samples are not logged", async ({ page }) => {
  await page.getByRole("button", { name: "Accept finding" }).click();
  await expect(page.getByRole("status")).toContainText("Sample cases are not logged");
});

test("has no serious accessibility violations", async ({ page }) => {
  const results = await new AxeBuilder({ page }).exclude("canvas").analyze();
  const serious = results.violations.filter((v) => v.impact === "serious" || v.impact === "critical").map((v) => `${v.id}: ${v.help}`);
  expect(serious).toEqual([]);
});
