// "What the measurements say": the four headline measurements, then the
// findings, as one hairline ledger. Every figure and sentence is from
// docs/research_docs/claims_and_evidence.md (unchanged from the previous
// landing page; only the motion is new).
//
// Scroll choreography, all scroll-linked (it reverses on scroll-up):
//  - the heading's words light up in reading order;
//  - each row's hairline draws left to right behind a short ice "scan head";
//  - each figure counts up to its published value (tickValue keeps sign,
//    unit and decimals, and lands on the exact published string);
//  - the two usable-mask shares fill a bar, and the conformal miss-rate range
//    grows on an axis with the nominal rate marked, so "below 1x = holds" is
//    visible, not just stated;
//  - each finding's status dot lights as its row arrives.
import { useRef } from "react";
import { motion, useTransform, type MotionValue } from "motion/react";
import { easeOutCubic, tickValue } from "../../lib/scrollScene";
import { useRevealProgress } from "./scrollHooks";

type Viz =
  | { kind: "share"; share: number } // a proportion of studies, 0..1
  | { kind: "range"; lo: number; hi: number; axisMax: number }; // x nominal

const STATS: { value: string; label: string; source: string; viz?: Viz }[] = [
  {
    value: "+0.0267",
    label: "Enhancing-tumour Dice over a matched U-Net",
    source: "n = 189 paired test cases, p_holm 1.4e-21",
  },
  {
    value: "85.2%",
    label: "Studies returned with a usable mask, in distribution",
    source: "WT and TC Dice ≥ 0.7, end to end",
    viz: { kind: "share", share: 0.852 },
  },
  {
    value: "24.2%",
    label: "Same, on the paediatric cohort (PED)",
    source: "Silent-failure rate 49.5%",
    viz: { kind: "share", share: 0.242 },
  },
  {
    value: "0.64–0.96×",
    label: "Realised miss rate vs. nominal α, in distribution",
    source: "Conformal bound holds on 6/6 cells; breaks under shift",
    viz: { kind: "range", lo: 0.64, hi: 0.96, axisMax: 1.5 },
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

const ROW = "relative grid gap-1 py-5 sm:grid-cols-[12.5rem_minmax(0,1fr)] sm:gap-8";

/** The row's top hairline, drawn by `p`, with a short ice segment riding its tip. */
function DrawnRule({ p }: { p: MotionValue<number> }) {
  const tip = useTransform(p, (v) => `${v * 100}%`);
  const tipOpacity = useTransform(p, [0, 0.08, 0.9, 1], [0, 1, 1, 0]);
  // overflow-hidden matters: the tip's layer is translated by up to 100% of
  // the row's width, and unclipped it widened the whole page (a horizontal
  // scrollbar on desktop, a zoomed-out layout on phones).
  return (
    <span aria-hidden="true" className="absolute inset-x-0 top-0 h-px overflow-hidden">
      <motion.span className="absolute inset-0 origin-left bg-surface-seam" style={{ scaleX: p }} />
      <motion.span className="absolute inset-0" style={{ x: tip, opacity: tipOpacity }}>
        <span className="absolute top-0 right-full h-px w-10 bg-linear-to-r from-transparent to-brand-primary" />
      </motion.span>
    </span>
  );
}

function ShareBar({ p, share }: { p: MotionValue<number>; share: number }) {
  const fill = useTransform(p, (v) => easeOutCubic(v) * share);
  return (
    <span aria-hidden="true" className="mt-2.5 block h-[3px] w-44 overflow-hidden bg-surface-raised">
      <motion.span className="block h-full origin-left bg-brand-primary" style={{ scaleX: fill }} />
    </span>
  );
}

function RangeAxis({ p, lo, hi, axisMax }: { p: MotionValue<number>; lo: number; hi: number; axisMax: number }) {
  const grow = useTransform(p, (v) => easeOutCubic(v));
  const pct = (v: number) => `${(v / axisMax) * 100}%`;
  return (
    <span aria-hidden="true" className="mt-2.5 block w-44">
      <span className="relative block h-[3px] bg-surface-raised">
        <motion.span
          className="absolute inset-y-0 origin-left bg-brand-primary"
          style={{ left: pct(lo), width: pct(hi - lo), scaleX: grow }}
        />
        <span className="absolute -top-1 h-[11px] w-px bg-text-secondary" style={{ left: pct(1) }} />
      </span>
      <span className="relative mt-1 block h-4 font-mono text-[11px] text-text-dim">
        <span className="absolute -translate-x-1/2 whitespace-nowrap" style={{ left: pct(1) }}>
          1× nominal
        </span>
      </span>
    </span>
  );
}

function StatRow({ stat }: { stat: (typeof STATS)[number] }) {
  const ref = useRef<HTMLDivElement>(null);
  const p = useRevealProgress(ref, ["start 0.95", "start 0.55"]);
  const shown = useTransform(p, (v) => tickValue(stat.value, easeOutCubic(v)));
  const textOpacity = useTransform(p, [0.15, 1], [0.35, 1]);
  const textX = useTransform(p, [0, 1], [16, 0]);
  return (
    <div ref={ref} className={ROW}>
      <DrawnRule p={p} />
      <div>
        <p className="font-mono text-2xl whitespace-nowrap text-text-primary">
          <span className="sr-only">{stat.value}</span>
          <motion.span aria-hidden="true">{shown}</motion.span>
        </p>
        {stat.viz?.kind === "share" && <ShareBar p={p} share={stat.viz.share} />}
        {stat.viz?.kind === "range" && <RangeAxis p={p} lo={stat.viz.lo} hi={stat.viz.hi} axisMax={stat.viz.axisMax} />}
      </div>
      <motion.div style={{ opacity: textOpacity, x: textX }}>
        <p className="text-base text-text-primary">{stat.label}</p>
        <p className="mt-1 text-sm text-text-dim">{stat.source}</p>
      </motion.div>
    </div>
  );
}

function FindingRow({ finding }: { finding: (typeof FINDINGS)[number] }) {
  const ref = useRef<HTMLDivElement>(null);
  const p = useRevealProgress(ref, ["start 0.92", "start 0.6"]);
  const dot = useTransform(p, [0.45, 0.8], [0, 1]);
  const halo = useTransform(p, [0.6, 0.8, 1], [0, 0.55, 0]);
  const haloScale = useTransform(p, [0.6, 1], [1, 3.2]);
  const textOpacity = useTransform(p, [0.15, 1], [0.35, 1]);
  const textX = useTransform(p, [0, 1], [16, 0]);
  return (
    <div ref={ref} className={ROW}>
      <DrawnRule p={p} />
      <p className="flex items-center gap-2 self-start font-mono text-sm leading-7 text-text-secondary">
        <span className="relative h-[6px] w-[6px] shrink-0" aria-hidden="true">
          <motion.span className={`absolute inset-0 rounded-full ${DOT[finding.tone]}`} style={{ opacity: halo, scale: haloScale }} />
          <motion.span className={`absolute inset-0 rounded-full ${DOT[finding.tone]}`} style={{ scale: dot }} />
        </span>
        {STATUS_WORD[finding.tag] ?? finding.tag}
      </p>
      <motion.p className="text-base text-text-primary" style={{ opacity: textOpacity, x: textX }}>
        {finding.text}
      </motion.p>
    </div>
  );
}

function Word({ p, i, n, children }: { p: MotionValue<number>; i: number; n: number; children: string }) {
  const opacity = useTransform(p, [i / n, (i + 1) / n], [0.22, 1]);
  return (
    <motion.span aria-hidden="true" style={{ opacity }}>
      {children}
      {i < n - 1 ? " " : ""}
    </motion.span>
  );
}

/** The heading lights up word by word, like a line being read. */
function ReadingHeading({ text }: { text: string }) {
  const ref = useRef<HTMLHeadingElement>(null);
  const p = useRevealProgress(ref, ["start 0.9", "start 0.45"]);
  const words = text.split(" ");
  return (
    <h2 ref={ref} aria-label={text} className="font-heading text-3xl font-semibold text-text-primary md:text-4xl">
      {words.map((w, i) => (
        <Word key={i} p={p} i={i} n={words.length}>
          {w}
        </Word>
      ))}
    </h2>
  );
}

function ClosingRule() {
  const ref = useRef<HTMLDivElement>(null);
  const p = useRevealProgress(ref, ["start 0.95", "start 0.6"]);
  return (
    <div ref={ref} className="relative h-px">
      <DrawnRule p={p} />
    </div>
  );
}

export function Ledger({ wrap }: { wrap: string }) {
  return (
    <section id="measurements" data-brain-pose="measurements" className={`${wrap} py-24 md:py-32`}>
      <ReadingHeading text="What the measurements say" />
      <div className="mt-10">
        {STATS.map((s) => (
          <StatRow key={s.value} stat={s} />
        ))}
        {FINDINGS.map((f) => (
          <FindingRow key={f.text} finding={f} />
        ))}
        <ClosingRule />
      </div>
    </section>
  );
}
