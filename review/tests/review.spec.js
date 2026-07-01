const { test, expect } = require("@playwright/test");
const path = require("node:path");
const fs = require("node:fs");

const FIXTURE = path.resolve(__dirname, "..", "fixtures", "sample_rollouts.jsonl");
const HTML = "file://" + path.resolve(__dirname, "..", "index.html");

async function loadFixture(page) {
  await page.goto(HTML);
  await page.waitForSelector("#file-input", { state: "attached" });
  await page.setInputFiles("#file-input", FIXTURE);
  await page.waitForSelector("#trace-card:not([hidden])");
}

async function readLabels(page) {
  return await page.evaluate(() => {
    const k = Object.keys(localStorage).find((x) => x.startsWith("trace-labels::"));
    return k ? JSON.parse(localStorage.getItem(k)) : {};
  });
}

test.describe("trace review interface", () => {
  test("loads rollouts.jsonl and displays the first trace", async ({ page }) => {
    await loadFixture(page);
    await expect(page.locator("#counter")).toContainText("1 of 5");
    await expect(page.locator("#rollout-badge")).toContainText("rollout 1");
    await expect(page.locator("#phase-badge")).toContainText("optimize");
    await expect(page.locator("#verdict-badge")).toContainText("✓ correct");
    await expect(page.locator("#pred")).toContainText("18");
    await expect(page.locator("#gold")).toContainText("18");
    await expect(page.locator("#student-out .answer-line")).toContainText("#### 18");
    await expect(page.locator("#feedback")).toContainText("Correct");
  });

  test("Pass labels a trace, persists to localStorage, and advances", async ({ page }) => {
    await loadFixture(page);
    await page.click("#pass");
    await expect(page.locator("#counter")).toContainText("2 of 5");
    const labels = await readLabels(page);
    expect(labels).toBeTruthy();
    const firstKey = Object.keys(labels)[0];
    expect(labels[firstKey].verdict).toBe("pass");
  });

  test("Fail with a note saves both verdict and notes", async ({ page }) => {
    await loadFixture(page);
    await page.click("#next");
    await expect(page.locator("#counter")).toContainText("2 of 5");
    await page.fill("#notes", "partial credit miscount at final add");
    await page.click("#fail");
    const labels = await readLabels(page);
    const failKey = Object.keys(labels).find((x) => x.startsWith("optimize#2:"));
    expect(labels[failKey].verdict).toBe("fail");
    expect(labels[failKey].notes).toBe("partial credit miscount at final add");
  });

  test("Defer records the verdict without advancing", async ({ page }) => {
    await loadFixture(page);
    await page.click("#defer");
    await expect(page.locator("#counter")).toContainText("1 of 5");
    await expect(page.locator("#pass")).not.toHaveClass(/active/);
    await expect(page.locator("#defer")).toHaveClass(/active/);
  });

  test("navigation buttons and counter update", async ({ page }) => {
    await loadFixture(page);
    await expect(page.locator("#prev")).toBeDisabled();
    await page.click("#next");
    await expect(page.locator("#counter")).toContainText("2 of 5");
    await page.click("#prev");
    await expect(page.locator("#counter")).toContainText("1 of 5");
    for (let i = 0; i < 4; i++) await page.click("#next");
    await expect(page.locator("#counter")).toContainText("5 of 5");
    await expect(page.locator("#next")).toBeDisabled();
  });

  test("keyboard shortcuts work (2/arrow/D/U)", async ({ page }) => {
    await loadFixture(page);
    await page.keyboard.press("2"); // Fail on trace 1, advance to 2
    await expect(page.locator("#counter")).toContainText("2 of 5");
    await page.keyboard.press("ArrowLeft"); // back to 1
    await expect(page.locator("#counter")).toContainText("1 of 5");
    await expect(page.locator("#fail")).toHaveClass(/active/);
    await page.keyboard.press("u"); // undo -> label removed
    await expect(page.locator("#fail")).not.toHaveClass(/active/);
    await page.keyboard.press("ArrowRight"); // to 2
    await page.keyboard.press("d"); // defer (no advance)
    await expect(page.locator("#counter")).toContainText("2 of 5");
    await expect(page.locator("#defer")).toHaveClass(/active/);
  });

  test("labels persist across reload", async ({ page }) => {
    await loadFixture(page);
    await page.click("#pass"); // label trace 1 as pass, now on trace 2
    await page.reload();
    await page.waitForSelector("#file-input", { state: "attached" });
    await page.setInputFiles("#file-input", FIXTURE);
    await page.waitForSelector("#trace-card:not([hidden])");
    await expect(page.locator("#counter")).toContainText("1 of 5");
    await expect(page.locator("#pass")).toHaveClass(/active/);
    await expect(page.locator("#counts")).toContainText("Pass 1");
  });

  test("collapsed system prompt is expandable and shows content", async ({ page }) => {
    await loadFixture(page);
    const details = page.locator("#prompt-details");
    await expect(page.locator("#prompt-text")).toContainText("helpful assistant");
    await expect(details).not.toHaveAttribute("open", "");
    await page.click("#prompt-details summary");
    await expect(details).toHaveAttribute("open", "");
    await expect(page.locator("#prompt-text")).toBeVisible();
  });

  test("Cmd+Enter saves the note and advances", async ({ page }) => {
    await loadFixture(page);
    await page.fill("#notes", "ok");
    await page.keyboard.press("Control+Enter");
    await expect(page.locator("#counter")).toContainText("2 of 5");
  });

  test("export downloads labels.jsonl with one line per labeled trace", async ({ page }) => {
    await loadFixture(page);
    await page.click("#pass"); // trace 1
    await page.click("#next");
    await page.click("#fail"); // trace 2 (advance) -> trace 3
    const [download] = await Promise.all([
      page.waitForEvent("download"),
      page.click("#export"),
    ]);
    expect(download.suggestedFilename()).toContain("labels.jsonl");
    const tmp = path.resolve(__dirname, "..", "downloaded-labels.jsonl");
    await download.saveAs(tmp);
    const lines = fs.readFileSync(tmp, "utf8").trim().split("\n").filter(Boolean);
    expect(lines.length).toBeGreaterThanOrEqual(2);
    const first = JSON.parse(lines[0]);
    expect(first.verdict).toMatch(/^(pass|fail|defer)$/);
    fs.unlinkSync(tmp);
  });

  test("jump-to-id moves to the matching rollout", async ({ page }) => {
    await loadFixture(page);
    await page.fill("#jump-id", "5");
    await page.click("#jump-btn");
    await expect(page.locator("#counter")).toContainText("5 of 5");
    await expect(page.locator("#rollout-badge")).toContainText("rollout 5");
  });

  test("context-exceeded trace renders with student output hidden and feedback shown", async ({ page }) => {
    await loadFixture(page);
    for (let i = 0; i < 2; i++) await page.click("#next");
    await expect(page.locator("#counter")).toContainText("3 of 5");
    await expect(page.locator("#student-block")).toBeHidden();
    await expect(page.locator("#feedback")).toContainText("context window");
    await expect(page.locator("#verdict-badge")).toContainText("✗ incorrect");
  });
});