import { useEffect, useState } from "react";
import { AppShell } from "../../components/AppShell";
import { useClinicalJob } from "../../hooks/useClinicalJob";
import { ClinicalJobStatus } from "./ClinicalJobStatus";
import { ClinicalStudyViewer } from "./ClinicalStudyViewer";
import { ClinicalUploadPanel } from "./ClinicalUploadPanel";
import { GatekeeperPanel } from "./GatekeeperPanel";
import { InputQCPanel } from "./InputQCPanel";
import { PipelineStepper } from "./PipelineStepper";
import { PreviousStudies } from "./PreviousStudies";
import { RefusalBanner } from "./RefusalBanner";
import { SeriesMatrix } from "./SeriesMatrix";

/**
 * Reads `?job=<id>` off the current URL, once, for the initial `jobId`
 * state - so a link to a finished study (copied from the address bar,
 * shared in the moments before a demo) reopens straight into it rather than
 * the upload panel. Read directly from `window.location.search` rather than
 * a router, so there is only one source of truth for the URL.
 */
function jobIdFromUrl(): string | null {
  return new URLSearchParams(window.location.search).get("job");
}

// What each gate outcome means, as fixed text (no data).
const GATE_RULES = [
  { dot: "bg-gate-proceed", name: "PROCEED", text: "All enabled signals within range." },
  {
    dot: "bg-gate-caution",
    name: "PROCEED WITH CAUTION",
    text: "A signal is borderline; the mask is shown with a warning.",
  },
  {
    // Hollow ring: refusal is amber but must not look identical to CAUTION.
    dot: "border-2 border-gate-caution",
    name: "REFUSE",
    text: "The study is declined — this is a correct outcome, not an error.",
  },
];

function HowTheGateDecides() {
  return (
    <section className="glass-panel p-5">
      <h2 className="font-heading text-base font-semibold">How the gate decides</h2>
      <ul className="mt-3 flex flex-col gap-3">
        {GATE_RULES.map((r) => (
          <li key={r.name} className="flex items-start gap-3">
            <span className={`mt-1.5 h-2 w-2 shrink-0 rounded-full ${r.dot}`} aria-hidden="true" />
            <div>
              <p className="font-mono text-xs text-text-primary">{r.name}</p>
              <p className="text-xs text-text-secondary">{r.text}</p>
            </div>
          </li>
        ))}
      </ul>
    </section>
  );
}

/**
 * Top-level clinical-upload page: pick a file, watch the job move through
 * the pipeline, then either a refusal explanation or the segmented study.
 *
 * Tracks only the active `job_id` in local state; `useClinicalJob` does all
 * the polling. "New upload" simply drops that id, discarding this component's
 * view of the old job - it does not call `deleteClinicalJob`, so a job a
 * user has moved on from still exists server-side (and can be revisited by
 * its id) until whatever housekeeping the backend does for `NVX_JOB_DIR`
 * reclaims it.
 */
export function ClinicalPage() {
  const [jobId, setJobId] = useState<string | null>(jobIdFromUrl);
  const { job, error } = useClinicalJob(jobId);
  // The done-state "Pipeline & QC details" overlay (see the done branch below).
  const [detailsOpen, setDetailsOpen] = useState(false);

  const isDone = job?.state === "done";

  // replaceState, not pushState: switching jobs (including back to null) is
  // not a navigation the user expects "back" to step through - it just keeps
  // the address bar's ?job= in sync with whatever is on screen.
  useEffect(() => {
    window.history.replaceState(
      {},
      "",
      jobId ? `?job=${encodeURIComponent(jobId)}` : window.location.pathname,
    );
  }, [jobId]);

  const toolbar = job ? (
    <>
      <button type="button" className="btn-secondary !px-3 !py-1 text-xs" onClick={() => setJobId(null)}>
        New upload
      </button>
      <span className="chip">job {job.job_id.slice(0, 8)}</span>
    </>
  ) : undefined;

  // The decision-bearing cards: refusal banner / failure card / gatekeeper
  // panel (also for refused jobs, so every signal is visible).
  const verdictCards = job && (
    <>
      {job.state === "refused" && <RefusalBanner job={job} hideInputQc />}
      {job.state === "failed" && (
        <div className="glass-panel border-gate-refuse/40 p-5">
          <p className="eyebrow">Failed</p>
          <p className="mt-2 font-mono text-xs leading-relaxed text-text-primary">
            {job.error ?? "The job failed for an unspecified reason."}
          </p>
        </div>
      )}
      {(job.state === "done" || job.state === "refused") && job.gatekeeper_decision && (
        <GatekeeperPanel job={job} />
      )}
    </>
  );

  // Everything else that explains the outcome of a job. Shared by both layouts.
  const outcome = job && (
    <>
      {error && <p className="font-mono text-xs text-text-secondary">{error}</p>}
      {job.input_qc_pre && (
        <InputQCPanel title="Input QC — before registration" report={job.input_qc_pre} />
      )}
      {job.input_qc_post && (
        <InputQCPanel title="Input QC — after registration" report={job.input_qc_post} />
      )}
    </>
  );

  if (isDone && job) {
    const decision = job.gatekeeper_decision?.decision ?? null;
    return (
      <AppShell fullHeight toolbar={toolbar}>
        <div className="flex h-full min-h-0 flex-col overflow-y-auto lg:overflow-hidden">
          {/* One slim status row instead of the old tall status/stepper/QC
              stack, so the viewer gets nearly the whole height. The pipeline
              and QC detail is still one click away, in an overlay panel. */}
          <div className="relative z-30 shrink-0 px-2 pt-2">
            <div className="glass-panel flex min-h-12 flex-wrap items-center gap-3 px-4 py-2">
              <span className="h-2 w-2 shrink-0 rounded-full bg-gate-proceed" aria-hidden="true" />
              <span className="font-heading text-sm font-semibold">Study complete</span>
              <span className="chip">job {job.job_id.slice(0, 8)}</span>
              {decision === "proceed" && (
                <span className="chip border-gate-proceed/50 text-gate-proceed">PROCEED</span>
              )}
              {decision === "proceed_with_caution" && (
                <span className="chip border-gate-caution/50 text-gate-caution">
                  PROCEED WITH CAUTION
                </span>
              )}
              <button
                type="button"
                className="btn-secondary ml-auto !px-3 !py-1 text-xs"
                aria-expanded={detailsOpen}
                onClick={() => setDetailsOpen((v) => !v)}
              >
                Pipeline &amp; QC details
              </button>
            </div>
            {detailsOpen && (
              <div className="glass-panel absolute inset-x-2 top-full mt-1 flex max-h-[50vh] flex-col gap-3 overflow-auto p-4 shadow-2xl">
                <ClinicalJobStatus job={job} />
                <PipelineStepper job={job} />
                {verdictCards}
                {outcome}
              </div>
            )}
          </div>
          {/* `flex flex-col` here is what lets the viewer's own `flex-1`
              root actually fill this slot: a flex-1 child of a plain block
              parent only ever gets its content height, which squashed the
              twin canvas to ~150 px on a 1050 px window (seen 2026-09-18 in
              the e2e screenshot, e2e/out/twin-*.png). */}
          <div className="flex flex-none flex-col lg:min-h-0 lg:flex-1">
            <ClinicalStudyViewer jobId={job.job_id} decision={decision} />
          </div>
        </div>
      </AppShell>
    );
  }

  return (
    <AppShell toolbar={toolbar}>
      <div className="bg-grid">
        <div className="mx-auto flex max-w-7xl flex-col gap-6 px-4 py-8 sm:px-6">
          <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
            <div className="max-w-2xl">
              <p className="eyebrow">Clinical study</p>
              <h1 className="mt-2 font-heading text-3xl font-semibold sm:text-4xl">DICOM study ingest</h1>
              <p className="mt-3 text-sm leading-relaxed text-text-secondary">
                Upload one patient's DICOM study as a .zip. It is checked, registered to the SRI24
                atlas, segmented, and then judged by a gate that can refuse it.
              </p>
            </div>
            <div className="glass-panel flex shrink-0 gap-6 px-5 py-3">
              <div>
                <p className="eyebrow">Required series</p>
                <p className="mt-1 font-mono text-sm">T1 · T1CE · T2 · FLAIR</p>
              </div>
              <div>
                <p className="eyebrow">Atlas space</p>
                <p className="mt-1 font-mono text-sm">SRI24</p>
              </div>
            </div>
          </div>

          <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_380px]">
            <div className="flex min-w-0 flex-col gap-6">
              {job ? <ClinicalJobStatus job={job} /> : <ClinicalUploadPanel onJobCreated={setJobId} />}
              {verdictCards}
              {!job && error && <p className="font-mono text-xs text-text-secondary">{error}</p>}
            </div>
            <div className="flex flex-col gap-6">
              <SeriesMatrix job={job ?? null} />
              <HowTheGateDecides />
            </div>
          </div>

          <PipelineStepper job={job ?? null} />

          {outcome}

          {/* Mounts only while jobId === null, so "New upload" remounts it and refetches. */}
          {!jobId && <PreviousStudies onOpen={setJobId} />}
        </div>
      </div>
    </AppShell>
  );
}
