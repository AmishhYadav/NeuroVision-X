// The clinical pipeline as numbered phase cards, beside a real MRI slice.
// Every step below mirrors the pipeline actually built (see
// app/backend/clinical_jobs.py and src/lib/pipelineSteps.ts). The slice is
// shown with its GROUND-TRUTH label (BraTS annotation), not a model output;
// the CursorReveal caption says so.
import { motion, useReducedMotion } from "motion/react";
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
  const reduce = useReducedMotion();
  return (
    <section className="mx-auto max-w-7xl px-4 py-16 sm:px-6">
      <span className="chip">Pipeline</span>
      <h2 className="mt-3 font-heading text-3xl font-semibold text-text-primary md:text-4xl">
        From DICOM zip to a decision
      </h2>
      <div className="mt-8 grid gap-8 lg:grid-cols-[minmax(0,1fr)_minmax(0,380px)]">
        <ol className="grid gap-4 sm:grid-cols-2">
          {PHASES.map((p, i) => (
            <motion.li
              key={p.n}
              initial={reduce ? false : { opacity: 0, y: 16 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: "-40px" }}
              transition={{ duration: 0.35, delay: reduce ? 0 : i * 0.05 }}
              className={`glass-panel flex flex-col gap-2 p-5 ${i === PHASES.length - 1 ? "sm:col-span-2" : ""}`}
            >
              <span className="font-mono text-2xl text-brand-teal">{p.n}</span>
              <h3 className="font-heading text-lg font-semibold text-text-primary">{p.title}</h3>
              <p className="text-sm leading-relaxed text-text-secondary">{p.body}</p>
            </motion.li>
          ))}
        </ol>
        <div className="glass-panel self-start p-4">
          <CursorReveal />
        </div>
      </div>
    </section>
  );
}
