// The clinical pipeline as a vertical timeline beside a real MRI slice. Every
// step below mirrors the pipeline actually built (see
// app/backend/clinical_jobs.py and src/lib/pipelineSteps.ts). The numbering is
// legitimate: it is a real ordered sequence. The slice is shown with its
// GROUND-TRUTH label (BraTS annotation), not a model output; the CursorReveal
// caption says so.
import { CursorReveal } from "./CursorReveal";

const PHASES: { n: string; title: string; body: string }[] = [
  {
    n: "01",
    title: "Ingest and first QC",
    body: "A DICOM zip is read and series are assigned to T1, T1CE, T2 and FLAIR. Input QC runs before anything is registered.",
  },
  {
    n: "02",
    title: "Registration and skull stripping",
    body: "Co-registration, SRI24 atlas registration and HD-BET skull stripping, then input QC a second time.",
  },
  {
    n: "03",
    title: "Segmentation",
    body: "The dual-encoder model predicts the enhancing tumour, tumour core and whole tumour regions.",
  },
  {
    n: "04",
    title: "Gatekeeper",
    body: "Input QC, a predicted-Dice estimate and intended use (adults only) decide. The conformal band and an OOD score are computed and shown, but do not decide.",
  },
  {
    n: "05",
    title: "Decision and outputs",
    body: "PROCEED, CAUTION or REFUSE, then a report, DICOM-SEG and a 3D twin. A refusal is a correct outcome, not an error.",
  },
];

export function PipelineStory() {
  return (
    <section className="mx-auto max-w-[1180px] px-4 py-24 sm:px-6 md:py-32">
      <h2 className="font-heading text-3xl font-semibold text-text-primary md:text-4xl">From DICOM zip to a decision</h2>
      <div className="mt-12 grid gap-12 lg:grid-cols-[minmax(0,1fr)_minmax(0,380px)] lg:gap-16">
        <ol className="relative m-0 list-none p-0">
          {/* The timeline rule: a 1px line behind the step numbers. */}
          <div aria-hidden="true" className="absolute top-2 bottom-2 left-[1.125rem] w-px bg-surface-seam" />
          {PHASES.map((p) => (
            <li key={p.n} className="relative grid grid-cols-[2.25rem_minmax(0,1fr)] gap-x-4 pb-9 last:pb-0">
              <span className="relative z-10 h-6 self-start bg-surface-page text-center font-mono text-sm leading-6 text-brand-primary">
                {p.n}
              </span>
              <div>
                <h3 className="font-heading text-lg font-semibold text-text-primary">{p.title}</h3>
                <p className="mt-1 max-w-[56ch] text-sm leading-relaxed text-text-secondary">{p.body}</p>
              </div>
            </li>
          ))}
        </ol>
        <div className="self-start lg:sticky lg:top-24">
          <CursorReveal />
        </div>
      </div>
    </section>
  );
}
