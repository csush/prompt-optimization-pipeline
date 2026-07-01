const { defineConfig } = require("@playwright/test");

module.exports = defineConfig({
  testDir: "./tests",
  use: {
    headless: true,
    viewport: { width: 1200, height: 900 },
    acceptDownloads: true,
  },
  projects: [{ name: "chromium", use: { browserName: "chromium" } }],
});