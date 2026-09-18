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
import { existsSync, mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
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
// Evidence trail for Target.attachedToTarget, in case auto-attach doesn't
// actually deliver a worker's console traffic - printed if the [twin] mesh
// check still can't find its line, rather than silently weakening that check.
const attachedTargets = [];
ws.onmessage = (e) => {
  const m = JSON.parse(e.data);
  if (m.id && pending.has(m.id)) {
    pending.get(m.id)(m);
    pending.delete(m.id);
  }
  if (m.method === "Target.attachedToTarget") {
    const info = m.params?.targetInfo ?? {};
    attachedTargets.push({ sessionId: m.params?.sessionId, type: info.type, url: info.url });
    if (info.type === "worker") {
      // A dedicated Worker (the twin mesher) gets its own CDP target - its
      // console.log calls land on THAT session's Runtime domain, not the
      // page's, so Runtime must be enabled per-session (flat mode: pass
      // sessionId at the top level of the command, not inside params).
      console.log(`     [cdp] attached worker target: ${info.url}`);
      send("Runtime.enable", {}, m.params.sessionId);
    }
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
const send = (method, params = {}, sessionId) =>
  new Promise((resolve) => {
    const id = nextId++;
    pending.set(id, resolve);
    const message = sessionId ? { id, sessionId, method, params } : { id, method, params };
    ws.send(JSON.stringify(message));
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
// Dedicated Workers (the twin mesher) spawned after this point get their
// own CDP target and, with flatten mode, announce themselves via
// Target.attachedToTarget (handled in ws.onmessage above) instead of
// requiring this harness to discover and attach to them itself.
await send("Target.setAutoAttach", {
  autoAttach: true,
  waitForDebuggerOnStart: false,
  flatten: true,
});
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
else console.log("     [cdp] attached targets seen so far: " + JSON.stringify(attachedTargets));

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
// ReportPage.tsx renders the overview as a hero card (a headline paragraph
// plus a facts grid) with no "At a glance" heading of its own - only the
// REMAINING sections carry an <h2> title. The hero's headline text is the
// thing to check is actually there.
const heroHeadline = await js(
  `(function(){const p=document.querySelector('p.font-condensed.text-2xl');return p ? p.innerText.trim() : null;})()`,
);
check(
  "overview hero renders a headline",
  !!heroHeadline && heroHeadline.length > 20,
  JSON.stringify(heroHeadline),
);

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

console.log("\n12. Clinical study viewer - twin, layers, report, pathology");
// Covers T1 (badge/caution), T2 (heat-layer switch), T3 (report dialog) and
// T5 (confirmed-pathology -> CNS5 line) in one section, all against the SAME
// live "done" clinical job - opening it fresh once is cheaper than four
// separate navigations, and it is what a real demo session actually does:
// one job, click through everything on it.
const CLINICAL_JOB = process.env.NVX_E2E_CLINICAL_JOB ?? null;
const consoleErrorsBeforeClinicalViewer = consoleErrors.length;

let clinicalViewerJobId = CLINICAL_JOB;
let clinicalViewerDecision = null;
if (clinicalViewerJobId) {
  // A job id was pinned via the env var - still need its decision for the
  // badge/caution-strip checks below, so look it up rather than guessing.
  const pinnedJob = await (
    await fetch(`http://localhost:8000/api/clinical/jobs/${clinicalViewerJobId}`)
  ).json();
  clinicalViewerDecision = pinnedJob.gatekeeper_decision?.decision ?? null;
} else {
  // list_clinical_jobs() (app/backend/clinical_jobs.py) already returns
  // newest-first, so the first "done" entry IS the newest done job.
  const clinicalJobsList = await (await fetch("http://localhost:8000/api/clinical/jobs")).json();
  const newestDone = (clinicalJobsList.jobs ?? []).find((j) => j.state === "done");
  if (newestDone) {
    clinicalViewerJobId = newestDone.job_id;
    clinicalViewerDecision = newestDone.gatekeeper_decision?.decision ?? null;
  }
}

if (!clinicalViewerJobId) {
  console.log("     skipped: no done clinical job (set NVX_E2E_CLINICAL_JOB)");
} else {
  console.log(`     using clinical job ${clinicalViewerJobId} (decision=${clinicalViewerDecision})`);

  // Re-queried fresh each time rather than held as a DOM handle - CDP
  // Runtime.evaluate with returnByValue can't hand back a live element
  // reference, and the twin view never remounts the canvas between these
  // steps anyway, so re-finding it by size is cheap and always current.
  const TWIN_CANVAS_DATA_URL_LEN = `(function(){
    const c = [...document.querySelectorAll('canvas')].find((c) => {
      const r = c.getBoundingClientRect();
      return r.width > 100 && r.height > 100;
    });
    return c ? c.toDataURL('image/png').length : 0;
  })()`;
  // A coarse 2D re-draw of the WebGL backbuffer (readable only because the
  // twin's <Canvas> sets preserveDrawingBuffer - see BrainTwinScene.tsx)
  // into a tiny offscreen canvas, counting pixels that are not
  // near-black. Cheaper than reading the full-resolution ImageData.
  const TWIN_CANVAS_NONBLACK_COUNT = `(function(){
    const c = [...document.querySelectorAll('canvas')].find((c) => {
      const r = c.getBoundingClientRect();
      return r.width > 100 && r.height > 100;
    });
    if (!c) return -1;
    const off = document.createElement('canvas');
    off.width = 64;
    off.height = 64;
    const ctx = off.getContext('2d');
    ctx.drawImage(c, 0, 0, 64, 64);
    const d = ctx.getImageData(0, 0, 64, 64).data;
    let n = 0;
    for (let p = 0; p < d.length; p += 4) {
      if (Math.max(d[p], d[p + 1], d[p + 2]) > 40) n++;
    }
    return n;
  })()`;
  const TWIN_CANVAS_CENTRE = `(function(){
    const c = [...document.querySelectorAll('canvas')].find((c) => {
      const r = c.getBoundingClientRect();
      return r.width > 100 && r.height > 100;
    });
    if (!c) return null;
    const r = c.getBoundingClientRect();
    return { x: r.x + r.width / 2, y: r.y + r.height / 2 };
  })()`;

  await send("Page.navigate", { url: `${CLINICAL_URL}?job=${clinicalViewerJobId}` });
  await sleep(2000);

  // --- 1. Twin button enables once all four volumes + mask have loaded ----
  let twinEnabled = false;
  for (let i = 0; i < 60; i++) {
    twinEnabled = await js(
      `(function(){const b=document.querySelector('[data-testid="clinical-view-twin"]');return b ? !b.disabled : false;})()`,
    );
    if (twinEnabled) break;
    await sleep(1000);
  }
  check("twin button enables once volumes are loaded", twinEnabled === true, String(twinEnabled));

  // --- 2. Switch to the twin, wait for the worker's mesh pass ------------
  const twinClickResult = await js(
    `(function(){const b=document.querySelector('[data-testid="clinical-view-twin"]');if(!b)return 'MISSING';if(b.disabled)return 'DISABLED';b.click();return 'ok';})()`,
  );
  check("3D twin button is clickable", twinClickResult === "ok", twinClickResult);

  const clinicalMeshRe = new RegExp(`\\[twin\\] mesh ${clinicalViewerJobId}`);
  let clinicalMeshLine = null;
  for (let i = 0; i < 40; i++) {
    clinicalMeshLine = consoleMessages.find((m) => clinicalMeshRe.test(m));
    if (clinicalMeshLine) break;
    await sleep(1000);
  }
  check("twin worker meshed this clinical job", !!clinicalMeshLine, clinicalMeshLine ?? "no [twin] mesh line seen");
  if (clinicalMeshLine) console.log("     " + clinicalMeshLine);
  else console.log("     [cdp] attached targets seen so far: " + JSON.stringify(attachedTargets));

  const twinPressedAfterClick = await js(
    `(function(){const b=document.querySelector('[data-testid="clinical-view-twin"]');return b ? b.getAttribute('aria-pressed') : null;})()`,
  );
  check("twin view button is pressed after clicking", twinPressedAfterClick === "true", String(twinPressedAfterClick));

  // --- 3/4. Pixels: non-trivial and actually non-black -------------------
  const d0 = await js(TWIN_CANVAS_DATA_URL_LEN);
  check("twin canvas is non-trivial", d0 > 5000, `dataURL length ${d0}`);
  const nonBlack0 = await js(TWIN_CANVAS_NONBLACK_COUNT);
  check("twin renders non-black pixels", nonBlack0 > 50, `${nonBlack0} of 4096 sampled px`);

  // --- 5. Orbit drag changes the rendered frame ---------------------------
  const canvasCentre = await js(TWIN_CANVAS_CENTRE);
  if (canvasCentre) {
    await send("Input.dispatchMouseEvent", {
      type: "mousePressed",
      x: canvasCentre.x,
      y: canvasCentre.y,
      button: "left",
      buttons: 1,
      clickCount: 1,
    });
    await send("Input.dispatchMouseEvent", {
      type: "mouseMoved",
      x: canvasCentre.x + 60,
      y: canvasCentre.y,
      button: "left",
      buttons: 1,
    });
    await send("Input.dispatchMouseEvent", {
      type: "mouseMoved",
      x: canvasCentre.x + 120,
      y: canvasCentre.y,
      button: "left",
      buttons: 1,
    });
    await send("Input.dispatchMouseEvent", {
      type: "mouseReleased",
      x: canvasCentre.x + 120,
      y: canvasCentre.y,
      button: "left",
      buttons: 0,
    });
    await sleep(800);
  }
  const d1 = await js(TWIN_CANVAS_DATA_URL_LEN);
  check(
    "orbit drag changes the rendered frame",
    !!canvasCentre && d1 !== d0,
    canvasCentre ? `${d0} vs ${d1}` : "no twin canvas found to drag",
  );

  // --- 6. Switching the heat layer repaints the twin ----------------------
  const heatSwitchResult = await js(`(function(){
    const group = document.querySelector('[role="group"][aria-label="Heat overlay"]');
    if (!group) return { status: 'MISSING_GROUP' };
    const buttons = [...group.querySelectorAll('button')];
    // The first enabled option that isn't already pressed and isn't the
    // "off" state - labels come from the component's own heatOptions, never
    // hardcoded here.
    const candidate = buttons.find(
      (b) => !b.disabled && b.getAttribute('aria-pressed') !== 'true' && !/^(none|off)/i.test(b.textContent.trim()),
    );
    if (!candidate) {
      return {
        status: 'NO_CANDIDATE',
        labels: buttons.map((b) => ({
          label: b.textContent.trim(),
          disabled: b.disabled,
          pressed: b.getAttribute('aria-pressed'),
        })),
      };
    }
    candidate.click();
    return { status: 'ok', label: candidate.textContent.trim() };
  })()`);
  check(
    "a heat overlay option is available to switch to",
    heatSwitchResult.status === "ok",
    JSON.stringify(heatSwitchResult),
  );
  if (heatSwitchResult.status === "ok") {
    console.log(`     switched heat overlay to "${heatSwitchResult.label}"`);
    await sleep(1500);
    const d2 = await js(TWIN_CANVAS_DATA_URL_LEN);
    check("switching the heat layer repaints the twin", d2 !== d1, `${d1} vs ${d2}`);
  }

  // --- 7. Badge mirrors the gatekeeper decision ---------------------------
  const twinBadgeText = await js(
    `(function(){const b=document.querySelector('[data-testid="twin-badge"]');return b ? b.innerText : null;})()`,
  );
  check(
    "twin badge is present and non-empty",
    !!twinBadgeText && twinBadgeText.trim().length > 0,
    String(twinBadgeText),
  );
  console.log(`     gatekeeper decision for this job: ${clinicalViewerDecision}`);
  if (clinicalViewerDecision === "proceed_with_caution") {
    check(
      "twin badge reads caution for a proceed_with_caution job",
      /caution/i.test(twinBadgeText ?? ""),
      String(twinBadgeText),
    );
    const cautionStripPresent = await js(
      `!!document.querySelector('[data-testid="clinical-caution-strip"]')`,
    );
    check(
      "caution strip is shown for a proceed_with_caution job",
      cautionStripPresent === true,
      String(cautionStripPresent),
    );
  }

  // --- 8. Report dialog opens and shows this job's real data -------------
  const reportBeforeOpen = await (
    await fetch(`http://localhost:8000/api/clinical/jobs/${clinicalViewerJobId}/report`)
  ).json();

  const reportToggleResult = await js(
    `(function(){const b=document.querySelector('[data-testid="clinical-report-toggle"]');if(!b)return 'MISSING';if(b.disabled)return 'DISABLED';b.click();return 'ok';})()`,
  );
  check("report toggle opens the report", reportToggleResult === "ok", reportToggleResult);

  let clinicalDialogText = null;
  for (let i = 0; i < 20; i++) {
    clinicalDialogText = await js(
      `(function(){const d=document.querySelector('[aria-label="Structured report"]');return d ? d.innerText : null;})()`,
    );
    if (clinicalDialogText) break;
    await sleep(500);
  }
  check("structured report dialog opens", !!clinicalDialogText, "dialog never appeared");
  check(
    "report dialog shows this job's case id",
    !!clinicalDialogText && clinicalDialogText.includes(reportBeforeOpen.case_id),
    reportBeforeOpen.case_id,
  );

  // The rendered structure NAME is a humanised, dictionary-mapped string
  // (humanStructureName in lib/reportInterpretation.ts), not the atlas
  // token this API returns - re-deriving that whole lookup table here would
  // duplicate business logic this harness has no business owning. Instead,
  // tie the check to structures[0]'s own NUMBERS, formatted the same simple
  // way formatPercent does (one decimal, a trailing "%") - that still proves
  // the row rendered is really this structure, at this index, from this
  // job's real report, without hand-copying the name dictionary.
  const firstStructure = reportBeforeOpen.anatomy?.structures?.[0] ?? null;
  if (firstStructure) {
    const expectedRegionPct = `${(firstStructure.frac_of_structure * 100).toFixed(1)}% of region`;
    const expectedTumourPct = `${(firstStructure.frac_of_tumour * 100).toFixed(1)}% of tumour`;
    check(
      "report's regions list renders anatomy.structures[0]'s own numbers",
      !!clinicalDialogText &&
        clinicalDialogText.includes(expectedRegionPct) &&
        clinicalDialogText.includes(expectedTumourPct),
      `${expectedRegionPct} / ${expectedTumourPct}`,
    );
  } else {
    console.log("     skipped: this job's report has no anatomy.structures entries");
  }

  // --- 9. Confirmed pathology -> CNS5 line, and it survives a reload -----
  const pathologyUrl = `http://localhost:8000/api/clinical/jobs/${clinicalViewerJobId}/pathology`;
  const resetPathology = () =>
    fetch(pathologyUrl, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ IDH: "Not entered", histology: "Not entered" }),
    });

  await resetPathology();

  const idhLabel = reportBeforeOpen.molecular?.markers?.IDH?.label ?? null;
  check("this job's report carries a molecular block with an IDH marker", !!idhLabel, JSON.stringify(reportBeforeOpen.molecular));

  if (idhLabel) {
    const setSelectValue = (label, value) => `(function(){
      const sel = document.querySelector('select[aria-label=${JSON.stringify(label)}]');
      if (!sel) return 'MISSING';
      const setter = Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, 'value').set;
      setter.call(sel, ${JSON.stringify(value)});
      sel.dispatchEvent(new Event('change', { bubbles: true }));
      return 'ok';
    })()`;

    const idhSetResult = await js(setSelectValue(idhLabel, "Wildtype"));
    check("IDH select accepts an entered value", idhSetResult === "ok", idhSetResult);
    await sleep(1000);
    const histologySetResult = await js(setSelectValue("Histology", "Glioblastoma pattern"));
    check("Histology select accepts an entered value", histologySetResult === "ok", histologySetResult);
    await sleep(1000);

    let cns5LineText = null;
    for (let i = 0; i < 20; i++) {
      cns5LineText = await js(
        `(function(){const el=document.querySelector('[data-testid="cns5-line"]');return el ? el.innerText : null;})()`,
      );
      if (cns5LineText) break;
      await sleep(500);
    }
    check("CNS5 line appears once both entries are made", !!cns5LineText, "no [data-testid=cns5-line] found within 10s");

    // Read the resolved name/source from the server itself, after the
    // in-page edits, rather than hardcoding the expected copy.
    const reportAfterEntry = await (
      await fetch(`http://localhost:8000/api/clinical/jobs/${clinicalViewerJobId}/report`)
    ).json();
    const cns5Name = reportAfterEntry.molecular?.cns5?.name ?? null;
    const cns5Source = reportAfterEntry.molecular?.cns5?.source ?? null;
    check(
      "CNS5 line names the resolved classification",
      !!cns5LineText && !!cns5Name && cns5LineText.includes(cns5Name),
      `expected to find "${cns5Name}" in "${cns5LineText}"`,
    );
    check(
      "CNS5 line cites its source",
      !!cns5LineText && !!cns5Source && cns5LineText.includes(cns5Source),
      `expected to find "${cns5Source}" in "${cns5LineText}"`,
    );

    // --- Reload: the entered value must survive a fresh mount -----------
    await send("Page.navigate", { url: `${CLINICAL_URL}?job=${clinicalViewerJobId}` });
    await sleep(3000);

    let reloadReportToggleResult = "DISABLED";
    for (let i = 0; i < 20; i++) {
      reloadReportToggleResult = await js(
        `(function(){const b=document.querySelector('[data-testid="clinical-report-toggle"]');if(!b)return 'MISSING';if(b.disabled)return 'DISABLED';b.click();return 'ok';})()`,
      );
      if (reloadReportToggleResult === "ok") break;
      await sleep(500);
    }
    check(
      "report toggle re-opens after reload",
      reloadReportToggleResult === "ok",
      reloadReportToggleResult,
    );

    let idhValueAfterReload = null;
    for (let i = 0; i < 20; i++) {
      idhValueAfterReload = await js(
        `(function(){const sel=document.querySelector('select[aria-label=${JSON.stringify(idhLabel)}]');return sel ? sel.value : null;})()`,
      );
      if (idhValueAfterReload === "Wildtype") break;
      await sleep(500);
    }
    check(
      "entered value survives reload",
      idhValueAfterReload === "Wildtype",
      `IDH select reads "${idhValueAfterReload}" after reload`,
    );

    await resetPathology();
  }

  // --- 10. Export button (T6.4) exists and is enabled ---------------------
  const exportButtonState = await js(
    `(function(){const b=document.querySelector('[data-testid="clinical-export"]');if(!b)return 'MISSING';return b.disabled ? 'DISABLED' : 'ok';})()`,
  );
  check("export button is present and enabled", exportButtonState === "ok", exportButtonState);

  // --- 11. Save an eyeball screenshot of the twin --------------------------
  // Close the report dialog if this run left it open, and make sure the
  // twin (not the slice grid a fresh mount defaults to) is the active view.
  await js(
    `(function(){const c=document.querySelector('[aria-label="Close report"]');if(c)c.click();return 'ok';})()`,
  );
  await sleep(300);
  const twinReadyForShot = await js(
    `(function(){const b=document.querySelector('[data-testid="clinical-view-twin"]');return b ? !b.disabled : false;})()`,
  );
  if (twinReadyForShot) {
    await js(
      `(function(){const b=document.querySelector('[data-testid="clinical-view-twin"]');if(b && b.getAttribute('aria-pressed')!=='true')b.click();return 'ok';})()`,
    );
    await sleep(1500);
  }
  const shotResponse = await send("Page.captureScreenshot", { format: "png" });
  const shotBase64 = shotResponse.result?.data ?? null;
  check("twin screenshot captured", !!shotBase64, "Page.captureScreenshot returned no data");
  if (shotBase64) {
    const shotDir = process.env.NVX_E2E_SHOT_DIR ?? "e2e/out";
    mkdirSync(shotDir, { recursive: true });
    const shotPath = join(shotDir, `twin-${clinicalViewerJobId.slice(0, 8)}.png`);
    writeFileSync(shotPath, Buffer.from(shotBase64, "base64"));
    console.log(`     saved ${shotPath}`);
  }

  // --- 12. Console hygiene for this section --------------------------------
  check(
    "no console errors from the clinical study viewer's interactions",
    consoleErrors.length === consoleErrorsBeforeClinicalViewer,
    consoleErrors.slice(consoleErrorsBeforeClinicalViewer).join(" | "),
  );
}

console.log(`\n${passed} passed, ${failures.length} failed`);
if (failures.length) {
  console.log("\nFAILURES:");
  for (const f of failures) console.log("  - " + f);
}
ws.close();
chrome.kill();
process.exit(failures.length ? 1 : 0);
