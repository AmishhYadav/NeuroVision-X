// "What this does not claim". The NOT_CLAIMED list quotes the pipeline's own
// block (src/neurovision/reporting/report.py); unchanged from before.
//
// Scroll choreography (scroll-linked): each item's hairline draws, then a
// strike line is drawn through the thing not claimed, and the reason comes up
// beside it. The strike is the point: these are claims the project leaves out
// on purpose. It is a background line (background-size grows), so it follows
// the text across line breaks, and it is decoration only - screen readers get
// the plain term.
import { useRef } from "react";
import { motion, useMotionTemplate, useTransform } from "motion/react";
import { useRevealProgress } from "./scrollHooks";

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

function Item({ item }: { item: (typeof NOT_CLAIMED)[number] }) {
  const ref = useRef<HTMLDivElement>(null);
  const p = useRevealProgress(ref, ["start 0.92", "start 0.58"]);
  const strike = useTransform(p, [0.35, 0.85], [0, 100]);
  const strikeSize = useMotionTemplate`${strike}% 1px`;
  const whyOpacity = useTransform(p, [0.3, 1], [0.4, 1]);
  const whyY = useTransform(p, [0, 1], [10, 0]);
  return (
    <div ref={ref} className="relative py-5">
      <motion.span aria-hidden="true" className="absolute inset-x-0 top-0 h-px origin-left bg-surface-seam" style={{ scaleX: p }} />
      <dt className="font-semibold text-text-primary">
        <motion.span
          className="bg-no-repeat [box-decoration-break:clone] [-webkit-box-decoration-break:clone]"
          style={{
            backgroundImage: "linear-gradient(var(--color-text-dim), var(--color-text-dim))",
            backgroundPosition: "0 58%",
            backgroundSize: strikeSize,
          }}
        >
          {item.what}
        </motion.span>
      </dt>
      <motion.dd className="mt-1.5 text-sm leading-relaxed text-text-secondary" style={{ opacity: whyOpacity, y: whyY }}>
        {item.why}
      </motion.dd>
    </div>
  );
}

export function NotClaimedSection({ wrap }: { wrap: string }) {
  return (
    <section id="not-claimed" data-brain-pose="notclaimed" className={`${wrap} py-24 md:py-32`}>
      <h2 className="font-heading text-3xl font-semibold text-text-primary md:text-4xl">What this does not claim</h2>
      <p className="mt-3 max-w-[58ch] text-base text-text-secondary">
        Four things a reader might assume, each left out on purpose, with the reason.
      </p>
      <dl className="mt-10 grid md:grid-cols-2 md:gap-x-12">
        {NOT_CLAIMED.map((item) => (
          <Item key={item.what} item={item} />
        ))}
      </dl>
    </section>
  );
}
