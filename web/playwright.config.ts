import { defineConfig, devices } from "@playwright/test";

// Runs against the production build (`pnpm run e2e` builds first). Sample-mode tests need no
// analysis server; the live-upload path is covered by backend/tests/api.
export default defineConfig({
  testDir: "e2e",
  timeout: 45_000,
  fullyParallel: true,
  reporter: [["list"]],
  use: {
    baseURL: "http://localhost:4174",
    ...devices["Desktop Chrome"],
    viewport: { width: 1440, height: 900 },
    // Headless CI has no GPU: SwiftShader gives WebGL2 in software.
    launchOptions: { args: ["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"] },
  },
  webServer: {
    command: "npx vite preview --port 4174 --strictPort",
    url: "http://localhost:4174",
    reuseExistingServer: true,
    timeout: 60_000,
  },
});
