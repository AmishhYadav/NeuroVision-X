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
//   node e2e/smoke.mjs http://localhost:5199/app
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

// "/" now serves the landing page (main.tsx routes on pathname); the tool
// this harness actually exercises lives at "/app". "/clinical" (section 11
// below) is a THIRD, independent route - the live DICOM-upload page - on
// the same origin.
const BASE = process.argv[2] ?? "http://localhost:5173/app";
const CLINICAL_URL = `${new URL(BASE).origin}/clinical`;
const PORT = 9344;

const profile = mkdtempSync(join(tmpdir(), "e2e-"));
const chrome = spawn(CHROME, [
  "--headless=new",
  `--remote-debugging-port=${PORT}`,
  `--user-data-dir=${profile}`,
  "--window-size=1680,1050",
  "--force-device-scale-factor=1",
  "--no-first-run",
  // The 3D twin (section 2a) needs a real WebGL context - "--disable-gpu"
  // (the old headless-Chrome recipe) killed that outright, which crashed
  // BrainTwinScene's <Canvas> on mount and, with it, the worker that logs
  // the "[twin] mesh" line this file polls for. SwiftShader gives headless
  // Chrome a software WebGL implementation instead; recent Chrome disables
  // it by default as "unsafe" unless asked for explicitly.
  "--use-gl=swiftshader",
  "--enable-unsafe-swiftshader",
  "--ignore-gpu-blocklist",
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
// Every debug/log line the page prints, in order - this is how section 2a
// below recovers the twin worker's `[twin] mesh ...` timing line without the
// harness having to poll the DOM (the worker logs to the console, not to
// anything visible on screen).
const consoleMessages = [];
ws.onmessage = (e) => {
  const m = JSON.parse(e.data);
  if (m.id && pending.has(m.id)) {
    pending.get(m.id)(m);
    pending.delete(m.id);
  }
  if (m.method === "Runtime.consoleAPICalled" && ["error", "assert"].includes(m.params.type)) {
    consoleErrors.push((m.params.args ?? []).map((a) => a.value ?? a.description ?? "").join(" "));
  }
  if (m.method === "Runtime.consoleAPICalled" && ["debug", "log"].includes(m.params.type)) {
    consoleMessages.push((m.params.args ?? []).map((a) => a.value ?? a.description ?? "").join(" "));
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
// Which experiment the server was started on is a config choice, not
// something this harness may assume - read it from the real API response
// instead of hardcoding a name, the same discipline the rest of this file
// already applies to every other expected value.
const healthJson = await js(`fetch('/api/health').then(r=>r.json())`);
check("header shows the experiment name", health.includes(healthJson.experiment), healthJson.experiment);
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

console.log("\n2a. 3D twin is the default view");
// The twin worker meshes brain + tumour off the main thread and logs one
// [twin] mesh line per case when it finishes - poll for it (up to 40s)
// rather than a fixed sleep, since mesh time depends on the machine and this
// is also how the author verifies the surfaceNets perf work: printing the
// line surfaces the actual timing numbers in the run output below.
let twinMeshLine = null;
for (let i = 0; i < 40; i++) {
  twinMeshLine = consoleMessages.find((m) => /\[twin\] mesh BraTS2021_00156:/.test(m));
  if (twinMeshLine) break;
  await sleep(1000);
}
check("twin worker meshed the selected case", !!twinMeshLine, twinMeshLine ?? "no [twin] mesh line seen");
if (twinMeshLine) console.log("     " + twinMeshLine);

const twinPressed = await js(
  `(function(){const b=[...document.querySelectorAll('button')].find(e=>e.textContent.includes('3D twin'));return b ? b.getAttribute('aria-pressed') : null;})()`,
);
check("3D twin button is pressed by default", twinPressed === "true", String(twinPressed));

const twinCanvasPresent = await js(
  `(function(){return [...document.querySelectorAll('canvas')].some(c=>{const r=c.getBoundingClientRect();return r.width>100 && r.height>100;});})()`,
);
check("a WebGL canvas is present with non-zero size", twinCanvasPresent === true, String(twinCanvasPresent));

// Switch to the flat scan view for the rest of this section and sections
// 3-8 below, which all assert on canvas pixels in the three-viewport grid -
// the twin is a single WebGL canvas with none of that structure. The
// choice persists across case switches (App.tsx keeps twinOpen out of the
// per-case reset effect), so this one click carries through the rest of
// the file.
check("Scan view button switches the active view", (await js(clickText("Scan view"))) === "ok");
await sleep(3000);

// Section 2's own canvas checks assume the flat scan view (three viewports
// plus the ribbon) - switched to just above in 2a, since the 3D twin is now
// what a case click opens by default.
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

console.log("\n10. Plain-language report page");
// Load a known case explicitly rather than relying on whatever section 8
// left active, so this section's expectations are self-contained. The
// expected values are read from the REAL API response, never hardcoded -
// a stale fixture would otherwise let this pass against a page that
// silently drifted from what report.py / reportInterpretation.ts actually
// produce.
const REPORT_CASE = "BraTS2021_00156";
await js(
  `(function(){[...document.querySelectorAll('button')].find(e=>e.textContent.includes(${JSON.stringify(REPORT_CASE)})).click();return 'ok';})()`,
);
// Case-detail (which the Report button's enabled state depends on) is a
// separate fetch from the click itself and a fixed sleep proved racy this
// deep into the suite, with many prior fetches behind it - poll instead of
// guessing a sleep long enough for the slowest run.
let reportButtonResult = "DISABLED";
for (let i = 0; i < 20; i++) {
  reportButtonResult = await js(clickText("Report"));
  if (reportButtonResult === "ok") break;
  await sleep(1000);
}
const apiReport = await (await fetch(`http://localhost:8000/api/report/${REPORT_CASE}`)).json();

// Tracked separately from section 9's check (which already ran, and so
// cannot see errors this section's own interactions might introduce) -
// this is the check that actually covers the report page's interactions.
const consoleErrorsBeforeReport = consoleErrors.length;

check(
  "Report button is present and enabled for a case with a report",
  reportButtonResult === "ok",
  reportButtonResult,
);
// The button now navigates (pushState to /report/<caseId>) instead of
// opening a side panel - give the new page's own effect (its own fetch of
// the same report, plus render) a moment to settle before reading the DOM.
await sleep(2500);

const reportPath = await js(`location.pathname`);
check("Report opens on its own route", reportPath === `/report/${REPORT_CASE}`, reportPath);

const pageText = await js(bodyText);
check("report page shows the case id", pageText.includes(REPORT_CASE));
check("at-a-glance section present", /At a glance/.test(pageText));

// Every category section's title, in the order ReportPage renders them.
// "Ventricles, white matter and tissue type" is the one optional section
// (buildSurroundings in reportInterpretation.ts returns null when the
// report carries no `involvement` block) - allow it to be absent only in
// that case, so this check still catches a genuinely missing section.
const CATEGORY_TITLES = [
  "How big it is",
  "What it is made of",
  "Where it is",
  "Which brain regions it touches",
  "Its shape",
  "One mass or several",
  "Ventricles, white matter and tissue type",
  "Nearness to 'eloquent' regions",
  "What this report does not say",
];
const missingTitles = CATEGORY_TITLES.filter((title) => {
  if (title === "Ventricles, white matter and tissue type" && apiReport.involvement === undefined) {
    return false;
  }
  return !pageText.includes(title);
});
check("every category section present", missingTitles.length === 0, JSON.stringify(missingTitles));

// formatVolumeMl's exact rendering (see lib/report.ts): mm3 -> mL at one
// decimal. Read from the real API response rather than reformatted here a
// second way, so this check catches a real formatting drift instead of
// agreeing with itself.
const expectedWtMl = `${(apiReport.burden.volumes.vol_WT_mm3 / 1000).toFixed(1)} mL`;
check("whole-tumour volume from the API appears on the page", pageText.includes(expectedWtMl), expectedWtMl);

const notClaimedCount = await js(
  `document.querySelectorAll('#section-limits ol > li').length`,
);
check(
  "Not Claimed list renders every entry the API returned",
  notClaimedCount === apiReport.not_claimed.length,
  `${notClaimedCount} rendered vs ${apiReport.not_claimed.length} from the API`,
);

// A real safety property, not a rendering check: the limits section is
// deliberately ALLOWED to use these words (it exists to say what is NOT
// claimed), and the collapsed "Full technical data" <details> is raw
// provenance keys rather than prose - strip both, then assert nothing else
// on the page strays into a grade/prognosis claim this pipeline has no
// basis for.
const claimFreeText = await js(`(function(){
  const clone = document.body.cloneNode(true);
  const limits = clone.querySelector('#section-limits');
  if (limits) limits.remove();
  for (const d of clone.querySelectorAll('details')) d.remove();
  return clone.innerText;
})()`);
check(
  "report page makes no grade/prognosis claim outside the limits section",
  !/\bgrade\b|prognos|malignan|aggressiv/i.test(claimFreeText),
);

const backClickResult = await js(
  `(function(){const a=[...document.querySelectorAll('a')].find(a=>a.textContent.includes('Back to viewer'));if(!a)return 'MISSING';a.click();return 'ok';})()`,
);
check("back-to-viewer link is present and clickable", backClickResult === "ok", backClickResult);
await sleep(3000);

const afterBackPath = await js(`location.pathname`);
const afterBackSearch = await js(`location.search`);
check(
  "back link returns to the viewer on the same case",
  afterBackPath === "/app" && afterBackSearch === `?case=${REPORT_CASE}`,
  `${afterBackPath}${afterBackSearch}`,
);

// App remounts fresh on this navigation (see main.tsx's no-router Root) and
// re-fetches the case list before it can mark this case selected - poll for
// aria-current rather than a fixed sleep, same rationale as the Report
// button's enabled state above.
let caseSelectedAfterBack = false;
for (let i = 0; i < 10; i++) {
  caseSelectedAfterBack = await js(
    `(function(){const b=[...document.querySelectorAll('button')].find(e=>e.textContent.includes(${JSON.stringify(REPORT_CASE)}));return b ? b.getAttribute('aria-current') === 'true' : false;})()`,
  );
  if (caseSelectedAfterBack) break;
  await sleep(1000);
}
check("the case is selected after coming back", caseSelectedAfterBack === true, String(caseSelectedAfterBack));

// Note: the twin is the active view again here (twinOpen resets to its
// default on this fresh mount, and the scan-view choice made back in 2a was
// component state, lost on navigation) - no scan-view pixel checks belong
// after this point.
check(
  "no console errors from the report page's interactions",
  consoleErrors.length === consoleErrorsBeforeReport,
  consoleErrors.slice(consoleErrorsBeforeReport).join(" | "),
);

console.log("\n11. Clinical upload page - refusal path");
// Deliberately the FASTEST possible refusal, and the one requiring no
// external stack (no dcm2niix, no ANTs, no HD-BET): a completely empty file
// makes `create_clinical_job` (app/backend/clinical_jobs.py) raise
// "dicom_zip is empty" before a job is even queued, which the backend
// answers with a plain 400 + {detail}. This never reaches a ClinicalJob at
// all, so it exercises ClinicalUploadPanel's error path rather than
// RefusalBanner (a `state="refused"` job) - see this harness's module
// docstring convention of testing what is fast and available rather than
// fabricating a real DICOM fixture just for this.
await send("Page.navigate", { url: CLINICAL_URL });
await sleep(1500);

const clinicalLoadedText = await js(bodyText);
check(
  "clinical page loads with an upload panel",
  /Upload a DICOM study/.test(clinicalLoadedText),
);
// Case-insensitive: the header link carries Tailwind's `uppercase`, which
// Chrome's innerText reflects (unlike textContent) - the same trap section
// 10's badge-text check below already documents.
check(
  "back-to-landing link is present",
  /neurovision-x/i.test(clinicalLoadedText),
);

// Drive the REAL <input type="file"> the same way a user would: construct a
// File in-page (0 bytes - the empty-upload case), attach it via a
// DataTransfer, and dispatch the change event React's onChange listens for.
const fileAttached = await js(`(function(){
  const input = document.querySelector('input[type="file"]');
  if (!input) return 'MISSING_INPUT';
  const file = new File([], 'empty.zip', { type: 'application/zip' });
  const dt = new DataTransfer();
  dt.items.add(file);
  input.files = dt.files;
  input.dispatchEvent(new Event('change', { bubbles: true }));
  return 'ok';
})()`);
check("file input accepts a picked file", fileAttached === "ok", fileAttached);
await sleep(300);

const uploadResult = await js(clickText("Upload study"));
check("Upload study button is enabled once a file is picked", uploadResult === "ok", uploadResult);

// Poll rather than a fixed sleep: the request itself is fast, but this is
// still a real network round trip through the dev proxy.
let clinicalErrorText = "";
for (let i = 0; i < 20; i++) {
  clinicalErrorText = await js(bodyText);
  if (/dicom_zip is empty/.test(clinicalErrorText)) break;
  await sleep(500);
}
check(
  "backend's actual refusal reason (\"dicom_zip is empty\") is shown, not a generic message",
  /dicom_zip is empty/.test(clinicalErrorText),
  clinicalErrorText.slice(0, 200),
);
check(
  "no job/progress UI appears for a rejected upload (it never became a job)",
  !/Declined — not segmented/.test(clinicalErrorText),
);

console.log(`\n${passed} passed, ${failures.length} failed`);
if (failures.length) {
  console.log("\nFAILURES:");
  for (const f of failures) console.log("  - " + f);
}
ws.close();
chrome.kill();
process.exit(failures.length ? 1 : 0);
