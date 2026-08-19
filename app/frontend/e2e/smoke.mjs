// End-to-end smoke test of the NeuroVision-X demo. Drives headless Chrome over
// the DevTools Protocol against the REAL backend and the REAL evaluation
// artifacts, and asserts on RENDERED PIXELS - a viewport that draws pure black
// satisfies every DOM assertion, so "the element exists" proves nothing here.
//
// Run it before showing the demo to anyone:
//
//   uvicorn app.backend.main:app        # terminal 1, from the repo root
//   npm run dev                         # terminal 2, in app/frontend
//   npm run test:e2e                    # terminal 3
//
// Pass a different URL as the first argument if the dev server moved:
//
//   node e2e/smoke.mjs http://localhost:5199/
//
// Uses no npm dependencies: Node 22 ships a global WebSocket and Chrome speaks
// CDP over it. Exits non-zero if any check fails, and always reports console
// errors from the page.
import { spawn } from "node:child_process";
import { existsSync, mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const CHROME_CANDIDATES = [
  process.env.CHROME_PATH,
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
  "/Applications/Chromium.app/Contents/MacOS/Chromium",
  "/usr/bin/google-chrome",
  "/usr/bin/chromium",
].filter(Boolean);

const CHROME = CHROME_CANDIDATES.find((p) => existsSync(p));
if (!CHROME) {
  console.error(
    "No Chrome found. Set CHROME_PATH to a Chrome or Chromium binary.\nLooked in:\n  " +
      CHROME_CANDIDATES.join("\n  "),
  );
  process.exit(2);
}

const BASE = process.argv[2] ?? "http://localhost:5173/";
const PORT = 9344;

const profile = mkdtempSync(join(tmpdir(), "e2e-"));
const chrome = spawn(CHROME, [
  "--headless=new",
  `--remote-debugging-port=${PORT}`,
  `--user-data-dir=${profile}`,
  "--window-size=1680,1050",
  "--force-device-scale-factor=1",
  "--no-first-run",
  "--disable-gpu",
  "about:blank",
]);
chrome.stderr.on("data", () => {});
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function target() {
  for (let i = 0; i < 50; i++) {
    try {
      const tabs = await (await fetch(`http://127.0.0.1:${PORT}/json/list`)).json();
      const page = tabs.find((t) => t.type === "page");
      if (page?.webSocketDebuggerUrl) return page.webSocketDebuggerUrl;
    } catch {
      /* not up yet */
    }
    await sleep(200);
  }
  throw new Error("chrome never exposed a CDP target");
}

const ws = new WebSocket(await target());
await new Promise((r) => (ws.onopen = r));
let nextId = 1;
const pending = new Map();
const consoleErrors = [];
ws.onmessage = (e) => {
  const m = JSON.parse(e.data);
  if (m.id && pending.has(m.id)) {
    pending.get(m.id)(m);
    pending.delete(m.id);
  }
  if (m.method === "Runtime.consoleAPICalled" && ["error", "assert"].includes(m.params.type)) {
    consoleErrors.push((m.params.args ?? []).map((a) => a.value ?? a.description ?? "").join(" "));
  }
  if (m.method === "Runtime.exceptionThrown") {
    consoleErrors.push(m.params.exceptionDetails?.exception?.description ?? "exception");
  }
};
const send = (method, params = {}) =>
  new Promise((resolve) => {
    const id = nextId++;
    pending.set(id, resolve);
    ws.send(JSON.stringify({ id, method, params }));
  });

async function js(expression) {
  const res = await send("Runtime.evaluate", {
    expression,
    returnByValue: true,
    awaitPromise: true,
  });
  if (res.result?.exceptionDetails) {
    throw new Error("JS threw: " + res.result.exceptionDetails.text + " " +
      (res.result.exceptionDetails.exception?.description ?? ""));
  }
  return res.result?.result?.value;
}

let passed = 0;
const failures = [];
function check(name, condition, detail = "") {
  if (condition) {
    passed++;
    console.log(`  ok   ${name}`);
  } else {
    failures.push(`${name}${detail ? " -- " + detail : ""}`);
    console.log(`  FAIL ${name}${detail ? " -- " + detail : ""}`);
  }
}

// --- helpers evaluated in the page -----------------------------------------
// Fingerprint a canvas by summing its pixels: distinguishes "rendered
// something" from "rendered black", and detects that a control changed what
// is on screen without needing a screenshot diff.
const FINGERPRINT = `(function(i){
  const c = document.querySelectorAll('canvas')[i];
  if (!c) return null;
  const ctx = c.getContext('2d');
  const d = ctx.getImageData(0, 0, c.width, c.height).data;
  let sum = 0, nonBlack = 0, colored = 0;
  for (let p = 0; p < d.length; p += 4) {
    const r = d[p], g = d[p+1], b = d[p+2];
    sum += r + g + b;
    if (r + g + b > 12) nonBlack++;
    if (Math.abs(r-g) > 18 || Math.abs(g-b) > 18) colored++;
  }
  return { sum, nonBlack, colored, w: c.width, h: c.height };
})`;

const clickText = (t) =>
  `(function(){const b=[...document.querySelectorAll('button')].find(e=>e.textContent.trim()===${JSON.stringify(t)});if(!b)return 'MISSING';if(b.disabled)return 'DISABLED';b.click();return 'ok';})()`;

const bodyText = `document.body.innerText`;

await send("Page.enable");
await send("Runtime.enable");
await send("Page.navigate", { url: BASE });
await sleep(3000);

console.log("\n1. Startup and case list");
const health = await js(bodyText);
check("header shows the experiment name", /baseline_unet3d/.test(health));
check("header shows a split label", /(test|val) split/.test(health), health.split("\n")[0]);
check("header does NOT mislabel test as val", !/val split/.test(health) || !/eval_test/.test(await js(`fetch('/api/health').then(r=>r.json()).then(h=>h.eval_dir)`)));
check("empty state invites an action", /Pick a case to begin/.test(health));
const caseCount = await js(
  `[...document.querySelectorAll('button')].filter(e=>/^BraTS2021_/.test(e.textContent.trim())).length`,
);
check("case list is populated", caseCount > 100, `${caseCount} cases`);

console.log("\n2. Load a case and confirm pixels actually render");
await js(clickText.length ? `(function(){[...document.querySelectorAll('button')].find(e=>e.textContent.includes('BraTS2021_00156')).click();return 'ok';})()` : "");
await sleep(9000);
const canvasCount = await js(`document.querySelectorAll('canvas').length`);
check("three viewports plus the ribbon are present", canvasCount >= 4, `${canvasCount} canvases`);

const fp = [];
for (let i = 0; i < 3; i++) fp.push(await js(`${FINGERPRINT}(${i})`));
for (let i = 0; i < 3; i++) {
  check(`viewport ${i} is not blank`, fp[i] && fp[i].nonBlack > 1000, JSON.stringify(fp[i]));
}

// Whether a panel SHOULD show colour is a property of the data, not an
// assumption: a mid-slice can legitimately contain no tumour (case 00156's
// coronal tumour spans slices 2-74, and the view opens at 85). Ask the
// profile which slices have tumour and assert the overlay agrees.
const planes = ["axial", "coronal", "sagittal"];
const planeProfile = await js(
  `fetch('/api/cases/BraTS2021_00156/profile').then(r=>r.json()).then(p=>p.planes)`,
);
const shown = await js(
  `[...document.body.innerText.matchAll(/(\\d+) \\/ (\\d+)/g)].map(m=>Number(m[1]))`,
);
for (let i = 0; i < 3; i++) {
  const tumourHere = (planeProfile[planes[i]]?.tumor ?? [])[shown[i]] ?? 0;
  const hasColour = fp[i] && fp[i].colored > 200;
  check(
    `viewport ${i} (${planes[i]}) overlay matches the data at slice ${shown[i]}`,
    tumourHere > 0 ? hasColour : !hasColour,
    `tumour_fraction=${tumourHere} colored=${fp[i]?.colored}`,
  );
}

console.log("\n3. Modality switching changes the image");
const beforeMod = await js(`${FINGERPRINT}(0)`);
check("FLAIR button responds", (await js(clickText("FLAIR"))) === "ok");
await sleep(2500);
const afterMod = await js(`${FINGERPRINT}(0)`);
check("switching to FLAIR changes the pixels", beforeMod.sum !== afterMod.sum,
  `${beforeMod.sum} vs ${afterMod.sum}`);
await js(clickText("T1CE"));
await sleep(2000);

console.log("\n4. Overlay modes");
const predFp = await js(`${FINGERPRINT}(0)`);
check("Truth mode selectable", (await js(clickText("Truth"))) === "ok");
await sleep(1200);
const truthFp = await js(`${FINGERPRINT}(0)`);
check("truth overlay differs from prediction", predFp.sum !== truthFp.sum,
  "identical sums would mean one mask is being drawn for both");
check("Disagreement mode selectable", (await js(clickText("Disagreement"))) === "ok");
await sleep(1200);
const disFp = await js(`${FINGERPRINT}(0)`);
check("disagreement overlay renders", disFp.colored > 0);
const disLegend = await js(bodyText);
check("legend switches to FN/FP", /False negative/.test(disLegend) && /False positive/.test(disLegend));
await js(clickText("Prediction"));
await sleep(1200);

console.log("\n5. Predictive entropy layer");
const beforeEnt = await js(`${FINGERPRINT}(0)`);
const entResult = await js(clickText("Predictive entropy"));
check("entropy toggle is enabled for a case with logits", entResult === "ok", entResult);
await sleep(2000);
const afterEnt = await js(`${FINGERPRINT}(0)`);
check("entropy layer visibly changes the render", beforeEnt.sum !== afterEnt.sum,
  `${beforeEnt.sum} vs ${afterEnt.sum} - equal means the layer is invisible`);
const entText = await js(bodyText);
check("entropy is labelled as single pass", /single pass/.test(entText));
check("entropy is NOT called epistemic-only or MC-dropout",
  !/MC-dropout/i.test(entText) && !/epistemic uncertainty/i.test(entText));
await js(clickText("Predictive entropy"));
await sleep(800);

console.log("\n6. Slice navigation");
const idxBefore = await js(`(document.body.innerText.match(/(\\d+) \\/ (\\d+)/) || [])[0]`);
// The focusable element is the ribbon's role="slider" wrapper, not the
// canvas inside it - the canvas is a painting surface with no semantics.
const ribbonFocus = await js(
  `(function(){const r=document.querySelector('[role="slider"]');if(!r)return 'MISSING';r.focus();return document.activeElement===r;})()`,
);
check("slice ribbon is focusable", ribbonFocus === true, String(ribbonFocus));
const ribbonAria = await js(
  `(function(){const r=document.querySelector('[role="slider"]');return r?{now:r.getAttribute('aria-valuenow'),max:r.getAttribute('aria-valuemax'),label:r.getAttribute('aria-label')}:null;})()`,
);
check("ribbon exposes its slice position to assistive tech",
  ribbonAria && ribbonAria.now !== null && ribbonAria.max !== null,
  JSON.stringify(ribbonAria));
await js(
  `(function(){const r=document.querySelector('[role="slider"]');for(let i=0;i<8;i++)r.dispatchEvent(new KeyboardEvent('keydown',{key:'ArrowRight',bubbles:true,cancelable:true}));return 'ok';})()`,
);
await sleep(1200);
const idxAfter = await js(`(document.body.innerText.match(/(\\d+) \\/ (\\d+)/) || [])[0]`);
check("ArrowRight advances the slice", idxBefore !== idxAfter, `${idxBefore} -> ${idxAfter}`);

console.log("\n7. Expand a plane");
const expandResult = await js(clickText("Sagittal"));
check("plane label expands the viewport", expandResult === "ok", expandResult);
await sleep(1500);
const expandedCanvases = await js(`document.querySelectorAll('canvas').length`);
check("expanded view shows one plane plus the ribbon", expandedCanvases === 2, `${expandedCanvases}`);
const expandedFp = await js(`${FINGERPRINT}(0)`);
check("expanded viewport still renders", expandedFp.nonBlack > 1000);

console.log("\n8. Case switching does not strand stale data");
await js(
  `(function(){[...document.querySelectorAll('button')].find(e=>e.textContent.includes('BraTS2021_00412')).click();return 'ok';})()`,
);
await sleep(1000);
await js(
  `(function(){[...document.querySelectorAll('button')].find(e=>e.textContent.includes('BraTS2021_01636')).click();return 'ok';})()`,
);
await sleep(10000);
const switched = await js(bodyText);
check("rapid case switch settles without an error banner", !/No response from the API/.test(switched));
const finalFp = await js(`${FINGERPRINT}(0)`);
check("viewport renders after a fast double switch", finalFp && finalFp.nonBlack > 500,
  JSON.stringify(finalFp));

console.log("\n9. Console hygiene");
check("no console errors or uncaught exceptions", consoleErrors.length === 0,
  consoleErrors.slice(0, 3).join(" | "));

console.log("\n10. Structured report panel");
// Load a known case explicitly rather than relying on whatever section 8
// left active, so this section's expectations are self-contained. The
// expected values are read from the REAL API response, never hardcoded -
// a stale fixture would otherwise let this pass against a panel that
// silently drifted from what report.py actually produces.
const REPORT_CASE = "BraTS2021_00156";
await js(
  `(function(){[...document.querySelectorAll('button')].find(e=>e.textContent.includes(${JSON.stringify(REPORT_CASE)})).click();return 'ok';})()`,
);
// Case-detail (which the Report toggle's enabled state depends on) is a
// separate fetch from the click itself and a fixed sleep proved racy this
// deep into the suite, with many prior fetches behind it - poll instead of
// guessing a sleep long enough for the slowest run.
let reportToggleResult = "DISABLED";
for (let i = 0; i < 20; i++) {
  reportToggleResult = await js(clickText("Report"));
  if (reportToggleResult === "ok") break;
  await sleep(1000);
}
const apiReport = await (await fetch(`http://localhost:8000/api/report/${REPORT_CASE}`)).json();

// Tracked separately from section 9's check (which already ran, and so
// cannot see errors this section's own interactions might introduce) -
// this is the check that actually covers the report panel's interactions.
const consoleErrorsBeforePanel = consoleErrors.length;

check(
  "Report toggle is present and enabled for a case with a report",
  reportToggleResult === "ok",
  reportToggleResult,
);
await sleep(2500);

const panelRect = await js(
  `(function(){const p=document.querySelector('[aria-label="Structured report"]');if(!p)return null;const r=p.getBoundingClientRect();return {w:r.width,h:r.height};})()`,
);
check(
  "opening the toggle reveals the panel at measured, non-zero geometry",
  !!panelRect && panelRect.w > 100 && panelRect.h > 100,
  JSON.stringify(panelRect),
);

const panelText = await js(
  `(function(){const p=document.querySelector('[aria-label="Structured report"]');return p ? p.innerText : '';})()`,
);
check("panel text includes the case id", panelText.includes(REPORT_CASE));
check(
  "panel text includes the non-diagnostic disclaimer",
  panelText.includes("not a diagnostic tool"),
);
check("panel text includes the atlas name", panelText.includes(apiReport.anatomy.atlas.name));
check(
  "panel text includes the eloquence classification name",
  panelText.includes(apiReport.eloquence.classification),
);

// Scoped to the badge element itself (`span.bg-surface-raised` is unique
// inside the panel - the only other `bg-surface-raised` is the disclaimer
// `div`), not the whole panel's text: the "Not claimed" section legitimately
// contains the phrase "ground truth" in its midline-shift caveat, so a
// panel-wide substring match would false-positive on prose that has nothing
// to do with which segmentation the report describes. Case-insensitive
// because the badge carries Tailwind's `uppercase`, which Chrome's innerText
// reflects (unlike textContent).
const badgeText = await js(
  `(function(){const b=document.querySelector('[aria-label="Structured report"] span.bg-surface-raised');return b ? b.innerText : null;})()`,
);
check(
  "badge says Model prediction, never Ground truth, for a prediction-sourced report",
  !!badgeText && /model prediction/i.test(badgeText) && !/ground truth/i.test(badgeText),
  JSON.stringify(badgeText),
);

// The structure table's row order is the SERVER's (frac_of_structure
// descending) - re-sorted client-side, it would bury exactly the row
// report.py's own docstring calls out: a structure holding a small share
// of the tumour but mostly destroyed itself.
const renderedStructureOrder = await js(
  `(function(){const t=document.querySelectorAll('[aria-label="Structured report"] table')[0];if(!t)return [];return [...t.querySelectorAll('tbody tr')].map(tr=>tr.querySelector('td').textContent.trim());})()`,
);
const expectedStructureOrder = apiReport.anatomy.structures.map((s) => s.structure);
check(
  "structure table renders rows in the API's own order",
  JSON.stringify(renderedStructureOrder) === JSON.stringify(expectedStructureOrder),
  `${JSON.stringify(renderedStructureOrder)} vs ${JSON.stringify(expectedStructureOrder)}`,
);

const notClaimedCount = await js(
  `document.querySelectorAll('[aria-label="Structured report"] li').length`,
);
check(
  "Not Claimed section renders every entry the API returned",
  notClaimedCount === apiReport.not_claimed.length,
  `${notClaimedCount} rendered vs ${apiReport.not_claimed.length} from the API`,
);

// Close and confirm the viewport underneath is still alive, the same
// pixel-level discipline the rest of this file uses throughout.
const closeResult = await js(
  `(function(){const b=document.querySelector('[aria-label="Close report"]');if(!b)return 'MISSING';b.click();return 'ok';})()`,
);
check("close button is present and closes the panel", closeResult === "ok", closeResult);
await sleep(1000);
const panelAfterClose = await js(
  `document.querySelector('[aria-label="Structured report"]') !== null`,
);
check("panel is gone from the DOM after closing", panelAfterClose === false);
const afterCloseFp = await js(`${FINGERPRINT}(0)`);
check(
  "viewport still renders non-blank pixels after closing the report panel",
  afterCloseFp && afterCloseFp.nonBlack > 1000,
  JSON.stringify(afterCloseFp),
);

// Reopen for this case, then switch to a different case while the panel is
// open - the same stale-data hazard section 8 already covers for volumes,
// here for the report panel. It must never go on describing the case that
// is no longer on screen.
await js(clickText("Report"));
await sleep(2000);
const OTHER_CASE = "BraTS2021_00412";
await js(
  `(function(){[...document.querySelectorAll('button')].find(e=>e.textContent.includes(${JSON.stringify(OTHER_CASE)})).click();return 'ok';})()`,
);
await sleep(4000);
const panelAfterSwitch = await js(
  `(function(){const p=document.querySelector('[aria-label="Structured report"]');return p ? p.innerText : null;})()`,
);
const strandedOnOldCase =
  panelAfterSwitch !== null &&
  panelAfterSwitch.includes(REPORT_CASE) &&
  !panelAfterSwitch.includes(OTHER_CASE);
check(
  "switching cases with the panel open never leaves it showing the previous case's report",
  !strandedOnOldCase,
  panelAfterSwitch === null ? "panel closed on switch" : panelAfterSwitch.slice(0, 160),
);

check(
  "no console errors from the report panel's interactions",
  consoleErrors.length === consoleErrorsBeforePanel,
  consoleErrors.slice(consoleErrorsBeforePanel).join(" | "),
);

console.log(`\n${passed} passed, ${failures.length} failed`);
if (failures.length) {
  console.log("\nFAILURES:");
  for (const f of failures) console.log("  - " + f);
}
ws.close();
chrome.kill();
process.exit(failures.length ? 1 : 0);
