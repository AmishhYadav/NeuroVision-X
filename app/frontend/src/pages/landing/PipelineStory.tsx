// The clinical pipeline as a vertical timeline beside a real MRI slice. Every
// step below mirrors the pipeline actually built (see
// app/backend/clinical_jobs.py and src/lib/pipelineSteps.ts). The numbering is
// legitimate: it is a real ordered sequence. The slice is shown with its
// GROUND-TRUTH label (BraTS annotation), not a model output; the CursorReveal
// caption says so.
//
// Scroll choreography (scroll-linked, reverses on scroll-up): an ice line
// fills the timeline as the reader moves down it, each step's node and text
// come up as the line reaches them, and a scan line sweeps down the sticky
// slice, uncovering the ground-truth label behind it.
import { useRef } from "react";
import { motion, useTransform } from "motion/react";
import { CursorReveal } from "./CursorReveal";
import { useRevealProgress } from "./scrollHooks";

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

function Step({ phase }: { phase: (typeof PHASES)[number] }) {
  const ref = useRef<HTMLLIElement>(null);
  // Reached when the step's top passes 55% of the viewport, where the
  // timeline's fill line is (see PipelineStory's offset).
  const p = useRevealProgress(ref, ["start 0.75", "start 0.55"]);
  const ring = useTransform(p, [0.6, 1], [0, 1]);
  const numberOpacity = useTransform(p, [0, 1], [0.5, 1]);
  const textOpacity = useTransform(p, [0, 1], [0.45, 1]);
  const textX = useTransform(p, [0, 1], [12, 0]);
  return (
    <li ref={ref} className="relative grid grid-cols-[2.25rem_minmax(0,1fr)] gap-x-4 pb-9 last:pb-0">
      <span className="relative z-10 flex h-9 w-9 items-center justify-center self-start rounded-full border border-surface-seam bg-surface-page">
        <motion.span aria-hidden="true" className="absolute -inset-px rounded-full border border-brand-primary" style={{ opacity: ring }} />
        <motion.span className="font-mono text-[13px] text-brand-primary" style={{ opacity: numberOpacity }}>
          {phase.n}
        </motion.span>
      </span>
      <motion.div className="pt-1" style={{ opacity: textOpacity, x: textX }}>
        <h3 className="font-heading text-lg font-semibold text-text-primary">{phase.title}</h3>
        <p className="mt-1 max-w-[56ch] text-sm leading-relaxed text-text-secondary">{phase.body}</p>
      </motion.div>
    </li>
  );
}

export function PipelineStory() {
  const listRef = useRef<HTMLOListElement>(null);
  const fill = useRevealProgress(listRef, ["start 0.55", "end 0.55"]);
  const scan = useRevealProgress(listRef, ["start 0.7", "end 0.45"]);
  return (
    <section id="pipeline" data-brain-pose="pipeline" className="mx-auto max-w-[1180px] px-4 py-24 sm:px-6 md:py-32">
      <h2 className="font-heading text-3xl font-semibold text-text-primary md:text-4xl">From DICOM zip to a decision</h2>
      <div className="mt-12 grid gap-12 lg:grid-cols-[minmax(0,1fr)_minmax(0,380px)] lg:gap-16">
        <ol ref={listRef} className="relative m-0 list-none p-0">
          {/* The timeline rule (1px, through the node centres), and its fill. */}
          <div aria-hidden="true" className="absolute top-4 bottom-4 left-[1.125rem] w-px bg-surface-seam" />
          <motion.div
            aria-hidden="true"
            className="absolute top-4 bottom-4 left-[1.125rem] w-px origin-top bg-brand-primary"
            style={{ scaleY: fill }}
          />
          {PHASES.map((p) => (
            <Step key={p.n} phase={p} />
          ))}
        </ol>
        <div className="self-start lg:sticky lg:top-24">
          <CursorReveal scan={scan} />
        </div>
      </div>
    </section>
  );
}
