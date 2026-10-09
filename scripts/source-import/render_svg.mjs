import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { pathToFileURL } from "node:url";

const [, , input, output, scale = "1"] = process.argv;
if (!input || !output) {
  console.error("usage: node scripts/source-import/render_svg.mjs board.svg board.png [scale]");
  process.exit(2);
}

function findPlaywright() {
  const roots = [path.join(os.homedir(), ".npm", "_npx")];
  for (const root of roots) {
    if (!fs.existsSync(root)) continue;
    for (const dir of fs.readdirSync(root)) {
      const candidate = path.join(root, dir, "node_modules", "playwright-core", "index.mjs");
      if (fs.existsSync(candidate)) return candidate;
    }
  }
  return null;
}

function findShell() {
  const cache = path.join(os.homedir(), "Library", "Caches", "ms-playwright");
  if (!fs.existsSync(cache)) return undefined;
  const shells = fs.readdirSync(cache).filter((d) => d.startsWith("chromium_headless_shell-")).sort().reverse();
  for (const dir of shells) {
    for (const arch of ["chrome-headless-shell-mac-arm64", "chrome-headless-shell-mac-x64", "chrome-headless-shell-linux64"]) {
      const exe = path.join(cache, dir, arch, "chrome-headless-shell");
      if (fs.existsSync(exe)) return exe;
    }
  }
  return undefined;
}

const core = findPlaywright();
if (!core) {
  console.error("playwright-core not found; run `npx -y playwright@1.56.1 --version` once, then retry");
  process.exit(1);
}
const { chromium } = await import(pathToFileURL(core).href);
const svg = fs.readFileSync(input, "utf8");
const match = svg.match(/viewBox="0 0 ([\d.]+) ([\d.]+)"/);
if (!match) {
  console.error("the SVG has no viewBox starting at 0 0");
  process.exit(1);
}
const width = Math.ceil(+match[1]);
const height = Math.ceil(+match[2]);
const browser = await chromium.launch({ executablePath: findShell() });
const page = await browser.newPage({ viewport: { width, height }, deviceScaleFactor: +scale });
await page.setContent(`<body style="margin:0;background:#fff">${svg}</body>`);
const overflow = await page.evaluate(() => {
  const root = document.querySelector("svg");
  const box = root.getBoundingClientRect();
  const outside = [];
  for (const el of root.querySelectorAll("text")) {
    const r = el.getBoundingClientRect();
    if (r.right > box.right + 1 || r.bottom > box.bottom + 1 || r.left < box.left - 1 || r.top < box.top - 1) {
      outside.push(el.textContent.slice(0, 40));
    }
  }
  return outside;
});
await page.screenshot({ path: output });
await browser.close();
console.log(`rendered ${output} ${width}x${height}`);
if (overflow.length) {
  console.log(`text outside the viewBox (${overflow.length}): ${overflow.join(" | ")}`);
  process.exit(3);
}
