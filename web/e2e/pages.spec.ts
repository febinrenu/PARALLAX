import AxeBuilder from "@axe-core/playwright";
import { expect, test } from "@playwright/test";

const serious = async (page: import("@playwright/test").Page) =>
  (await new AxeBuilder({ page }).analyze()).violations.filter((v) => v.impact === "serious" || v.impact === "critical").map((v) => v.id);

test("validation shows the real leakage audit", async ({ page }) => {
  await page.goto("/validation");
  await expect(page.getByRole("heading", { name: "Leakage audit" })).toBeVisible();
  await expect(page.getByRole("rowheader", { name: /Brain Tumor MRI \(Kaggle\)/ })).toBeVisible();
  await expect(page.getByRole("cell", { name: "2,019" })).toBeVisible();
  expect(await serious(page)).toEqual([]);
});

test("models lists every declared resource with its licence", async ({ page }) => {
  await page.goto("/models");
  await expect(page.getByText("TorchXRayVision DenseNet121 (all)")).toBeVisible();
  await expect(page.getByText("FracAtlas").first()).toBeVisible();
  expect(await serious(page)).toEqual([]);
});

test("the sample report renders the fixed-template sentences", async ({ page }) => {
  await page.goto("/report/sample-chest");
  await expect(page.getByText("Doctor, consider pneumonia in the right upper zone (moderate confidence).")).toBeVisible();
  await expect(page.getByText(/Ledger head/)).toBeVisible();
});

test("unknown routes get the not-found page", async ({ page }) => {
  await page.goto("/read/nope/extra");
  await page.goto("/models/nope");
  await expect(page.getByRole("heading", { name: "No page here." })).toBeVisible();
});
