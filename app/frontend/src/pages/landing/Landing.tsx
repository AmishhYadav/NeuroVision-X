// The landing page. Dark Stitch look (see index.css tokens). Copy is limited
// to facts in docs/research_docs/claims_and_evidence.md: this is a research
// prototype, so the page leads with measurements, includes the negative
// results, and ends with what the project does NOT claim. The NOT_CLAIMED
// list quotes the pipeline's own block (src/neurovision/reporting/report.py).
import { lazy, Suspense, type ReactNode } from "react";
import { motion, useReducedMotion } from "motion/react";
import { AppShell } from "../../components/AppShell";
import { navigateTo } from "../../lib/navigate";
import { isPlainLeftClick } from "../../lib/shellStatus";
// three.js / fiber / drei live behind this import so the headline paints first.
const HeroBrain = lazy(() => import("./HeroBrain").then((m) => ({ default: m.HeroBrain })));
import { PipelineStory } from "./PipelineStory";

const NOT_CLAIMED: { what: string; why: string }[] = [
  {
    what: "WHO grade",
    why: "WHO CNS5 grading needs histology plus molecular markers (IDH, 1p/19q, ATRX, TERT, CDKN2A/B) that are not present anywhere in this dataset.",
  },
  {
    what: "Prognosis or outcome",
    why: "This dataset carries no clinical outcomes to validate a prognosis against, so none is computed or implied.",
  },
  {
    what: "Mass effect or midline shift",
    why: "The atlas encodes where a healthy midline sits, not where this patient's own does, and BraTS ships no midline-shift ground truth to validate a displacement estimate against.",
  },
  {
    what: "Any deficit the patient has or will experience",
    why: "A deficit claim is unvalidatable against the outcomes data this project has, so no deficit or functional-loss text is generated anywhere in this artifact.",
  },
];

const STATS = [
  {
    value: "+0.0267",
    label: "Enhancing-tumour Dice over a matched U-Net",
    source: "n = 189 paired test cases, p_holm 1.4e-21",
  },
  {
    value: "85.2%",
    label: "Studies returned with a usable mask, in distribution",
    source: "WT and TC Dice ≥ 0.7, end to end",
  },
  {
    value: "24.2%",
    label: "Same, on the paediatric cohort (PED)",
    source: "Silent-failure rate 49.5%",
  },
  {
    value: "0.64–0.96×",
    label: "Realised miss rate vs. nominal α, in distribution",
    source: "Conformal bound holds on 6/6 cells; breaks under shift",
  },
];

type Tone = "holds" | "caution" | "neutral";
const FINDINGS: { tone: Tone; tag: string; text: string }[] = [
  { tone: "holds", tag: "Holds", text: "Conformal bound holds in distribution for both models (6/6 cells)." },
  {
    tone: "caution",
    tag: "Breaks",
    text: "Under shift the bound fails in proportion to the shift — SSA whole tumour ~1.1× nominal, PED whole tumour 1.4–1.9×; PED tumour core is worse still, partly from a label-definition mismatch.",
  },
  {
    tone: "caution",
    tag: "Negative",
    text: "The accuracy gain does not transfer to the African (SSA) or paediatric (PED) cohorts.",
  },
  {
    tone: "caution",
    tag: "Negative",
    text: "The founding hypothesis — disagreement-conditioned gating — is a pre-registered null.",
  },
  {
    tone: "neutral",
    tag: "Baseline wins",
    text: "Free single-pass predictive entropy is undefeated as an error localiser; MC-dropout is equivalent to it.",
  },
  {
    tone: "neutral",
    tag: "Gate",
    text: "The refusal gate has decent precision but poor recall, and is blind to cohort-level shift.",
  },
];
const DOT: Record<Tone, string> = {
  holds: "bg-gate-proceed",
  caution: "bg-gate-caution",
  neutral: "bg-text-dim",
};

const STATUS_WORD: Record<string, string> = {
  Holds: "holds",
  Breaks: "breaks",
  Negative: "negative",
  "Baseline wins": "baseline",
  Gate: "gate",
};

/** Real <a href> that does pushState navigation on a plain left-click only. */
function NavLink({ href, className, children }: { href: string; className: string; children: ReactNode }) {
  return (
    <a
      href={href}
      className={className}
      onClick={(e) => {
        if (isPlainLeftClick(e)) {
          e.preventDefault();
          navigateTo(href);
        }
      }}
    >
      {children}
    </a>
  );
}

function CtaPair() {
  return (
    <div className="flex flex-wrap gap-3">
      <NavLink href="/clinical" className="btn-primary">
        Open a clinical study
      </NavLink>
      <NavLink href="/app" className="btn-secondary">
        Browse test cases
      </NavLink>
    </div>
  );
}

const WRAP = "mx-auto max-w-[1180px] px-4 sm:px-6";
const EASE_OUT_QUART = [0.25, 1, 0.5, 1] as const;

/** One orchestrated load: children fade and rise 12px, 60ms apart. Content is
 * in the DOM and visible by default under reduced motion. */
function Hero() {
  const reduce = useReducedMotion();
  const item = {
    hidden: reduce ? { opacity: 1, y: 0 } : { opacity: 0, y: 12 },
    show: { opacity: 1, y: 0, transition: { duration: 0.5, ease: EASE_OUT_QUART } },
  };
  return (
    <section className={`${WRAP} grid min-h-[calc(100svh-4rem)] items-center gap-12 py-12 lg:grid-cols-[minmax(0,1fr)_minmax(0,600px)] lg:gap-8`}>
      <motion.div
        className="flex flex-col items-start"
        initial="hidden"
        animate="show"
        variants={{ show: { transition: { staggerChildren: reduce ? 0 : 0.06 } } }}
      >
        <motion.h1
          variants={item}
          className="max-w-[18ch] font-heading font-bold text-text-primary lg:max-w-[15ch]"
          style={{ fontSize: "clamp(2.4rem, 4.6vw, 4.25rem)", letterSpacing: "-0.02em", lineHeight: 1.05, textWrap: "balance" }}
        >
          Where a tumour segmentation model can be trusted — and where it breaks.
        </motion.h1>
        <motion.p variants={item} className="mt-6 max-w-[58ch] text-[1.05rem] leading-[1.6] text-text-secondary">
          A 3D brain-tumour segmentation model wrapped in a distribution-free error bound and a refusal gate. The
          bound holds in distribution, fails under distribution shift in proportion to the shift, and the gate
          cannot see cohort-level shift. This project measures where it breaks, and what a new site needs to restore
          it.
        </motion.p>
        <motion.div variants={item} className="mt-8">
          <CtaPair />
        </motion.div>
        <motion.p variants={item} className="mt-4 text-sm text-text-dim">
          Research prototype on BraTS 2021 — not for clinical use.
        </motion.p>
      </motion.div>
      <motion.div
        initial={{ opacity: reduce ? 1 : 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.9, delay: reduce ? 0 : 0.2, ease: EASE_OUT_QUART }}
      >
        <Suspense
          fallback={
            <div className="mx-auto flex aspect-square w-full max-w-[640px] items-center justify-center text-sm text-text-dim">
              Loading brain surface…
            </div>
          }
        >
          <HeroBrain />
        </Suspense>
      </motion.div>
    </section>
  );
}

/** One ledger: the four measurements, then the findings. Hairline rows, no boxes. */
function Ledger() {
  const row = "grid gap-1 border-t border-surface-seam py-5 sm:grid-cols-[12.5rem_minmax(0,1fr)] sm:gap-8";
  return (
    <section className={`${WRAP} py-24 md:py-32`}>
      <h2 className="font-heading text-3xl font-semibold text-text-primary md:text-4xl">What the measurements say</h2>
      <div className="mt-10">
        {STATS.map((s) => (
          <div key={s.value} className={row}>
            <p className="whitespace-nowrap font-mono text-2xl text-text-primary">{s.value}</p>
            <div>
              <p className="text-base text-text-primary">{s.label}</p>
              <p className="mt-1 text-sm text-text-dim">{s.source}</p>
            </div>
          </div>
        ))}
        {FINDINGS.map((f) => (
          <div key={f.text} className={row}>
            <p className="flex items-center gap-2 self-start font-mono text-sm leading-7 text-text-secondary">
              <span className={`h-[6px] w-[6px] shrink-0 rounded-full ${DOT[f.tone]}`} aria-hidden="true" />
              {STATUS_WORD[f.tag] ?? f.tag}
            </p>
            <p className="text-base text-text-primary">{f.text}</p>
          </div>
        ))}
        <div className="border-t border-surface-seam" />
      </div>
    </section>
  );
}

const NODE_LABEL = "fill-text-primary text-[14px] font-semibold";
const NODE_SUB = "fill-text-dim text-[13px]";

/** Horizontal data-flow diagram (lg and up). 1px strokes, no boxes. */
function FlowWide() {
  return (
    <svg viewBox="0 0 1060 290" className="hidden h-auto w-full text-text-dim lg:block" role="img" aria-label="4 MRI channels split into a 3D CNN encoder and a Swin Transformer encoder, merge in gated cross-attention fusion, pass through a U-Net decoder, and produce segmentation, confidence and boundary outputs.">
      <g fill="none" stroke="currentColor" strokeWidth="1" vectorEffect="non-scaling-stroke">
        <path d="M200 165 H250 M250 80 V250 M250 80 H290 M250 250 H290" />
        <path d="M500 80 H540 V250 H500 M540 165 H590 M590 165 H820" />
        <path d="M820 165 H890 M890 80 V250 M890 80 H925 M890 165 H925 M890 250 H925" />
      </g>
      <g fill="currentColor">
        {[[200, 165], [290, 80], [290, 250], [590, 165], [820, 165], [925, 80], [925, 165], [925, 250]].map(([x, y]) => (
          <circle key={`${x}-${y}`} cx={x} cy={y} r="3" />
        ))}
      </g>
      <text x="0" y="160" className={NODE_LABEL}>4 MRI channels</text>
      <text x="0" y="180" className={NODE_SUB}>(T1, T1CE, T2, FLAIR)</text>
      <text x="306" y="76" className={NODE_LABEL}>3D CNN encoder</text>
      <text x="306" y="96" className={NODE_SUB}>local texture and boundaries</text>
      <text x="306" y="246" className={NODE_LABEL}>Swin Transformer encoder</text>
      <text x="306" y="266" className={NODE_SUB}>long-range context</text>
      <text x="604" y="150" className={NODE_LABEL}>Gated cross-attention fusion</text>
      <text x="820" y="196" textAnchor="middle" className={NODE_LABEL}>U-Net decoder</text>
      <text x="941" y="85" className={NODE_LABEL}>segmentation</text>
      <text x="941" y="170" className={NODE_LABEL}>confidence</text>
      <text x="941" y="255" className={NODE_LABEL}>boundary</text>
    </svg>
  );
}

/** Vertical version for narrow screens; fits a 360px viewport without scaling past 1:1. */
function FlowNarrow() {
  return (
    <svg viewBox="0 0 328 440" className="mx-auto block h-auto w-full max-w-[420px] text-text-dim lg:hidden" role="img" aria-label="4 MRI channels split into a 3D CNN encoder and a Swin Transformer encoder, merge in gated cross-attention fusion, pass through a U-Net decoder, and produce segmentation, confidence and boundary outputs.">
      <g fill="none" stroke="currentColor" strokeWidth="1" vectorEffect="non-scaling-stroke">
        <path d="M164 50 V70 M78 70 H250 M78 70 V96 M250 70 V96" />
        <path d="M78 96 V180 H164 M250 96 V180 H164 M164 180 V240 M164 280 V305" />
        <path d="M164 305 V345 M50 345 H278 M50 345 V385 M164 345 V385 M278 345 V385" />
      </g>
      <g fill="currentColor">
        {[[164, 50], [78, 96], [250, 96], [164, 240], [164, 305], [50, 385], [164, 385], [278, 385]].map(([x, y]) => (
          <circle key={`${x}-${y}`} cx={x} cy={y} r="3" />
        ))}
      </g>
      <g textAnchor="middle">
        <text x="164" y="18" className={NODE_LABEL}>4 MRI channels</text>
        <text x="164" y="36" className={NODE_SUB}>(T1, T1CE, T2, FLAIR)</text>
        <text x="78" y="118" className={NODE_LABEL}>3D CNN encoder</text>
        <text x="78" y="136" className={NODE_SUB}>local texture and</text>
        <text x="78" y="152" className={NODE_SUB}>boundaries</text>
        <text x="250" y="118" className={NODE_LABEL}>Swin Transformer</text>
        <text x="250" y="136" className={NODE_LABEL}>encoder</text>
        <text x="250" y="154" className={NODE_SUB}>long-range context</text>
        <text x="164" y="266" className={NODE_LABEL}>Gated cross-attention fusion</text>
        <text x="180" y="309" textAnchor="start" className={NODE_LABEL}>U-Net decoder</text>
        <text x="50" y="410" className={NODE_LABEL}>segmentation</text>
        <text x="164" y="410" className={NODE_LABEL}>confidence</text>
        <text x="278" y="410" className={NODE_LABEL}>boundary</text>
      </g>
    </svg>
  );
}

function Architecture() {
  return (
    <section className={`${WRAP} pb-24 md:pb-32`}>
      <h2 className="font-heading text-3xl font-semibold text-text-primary md:text-4xl">Dual encoder, gated fusion</h2>
      <div className="mt-10">
        <FlowWide />
        <FlowNarrow />
      </div>
      <p className="mt-8 max-w-[70ch] font-mono text-sm text-text-secondary">
        34.91M parameters. Most of the ET gain is architectural (+0.0211), not width (+0.0055) — measured against a
        width-matched control.
      </p>
    </section>
  );
}

function NotClaimedSection() {
  return (
    <section className={`${WRAP} py-24 md:py-32`}>
      <h2 className="font-heading text-3xl font-semibold text-text-primary md:text-4xl">What this does not claim</h2>
      <p className="mt-3 max-w-[58ch] text-base text-text-secondary">
        Four things a reader might assume, each left out on purpose, with the reason.
      </p>
      <dl className="mt-10 grid md:grid-cols-2 md:gap-x-12">
        {NOT_CLAIMED.map((item) => (
          <div key={item.what} className="border-t border-surface-seam py-5">
            <dt className="font-semibold text-text-primary">{item.what}</dt>
            <dd className="mt-1.5 text-sm leading-relaxed text-text-secondary">{item.why}</dd>
          </div>
        ))}
      </dl>
    </section>
  );
}

function Closing() {
  return (
    <section className={`${WRAP} pt-12 pb-28 md:pt-20 md:pb-36`}>
      <h2 className="font-heading text-3xl font-semibold text-text-primary">Run a study through the pipeline</h2>
      <div className="mt-6">
        <CtaPair />
      </div>
    </section>
  );
}

export function Landing() {
  return (
    <AppShell>
      <Hero />
      <Ledger />
      <Architecture />
      <PipelineStory />
      <NotClaimedSection />
      <Closing />
    </AppShell>
  );
}
