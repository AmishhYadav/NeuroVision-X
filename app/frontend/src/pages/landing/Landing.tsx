// The landing page. "Reading room" design system (see index.css tokens). Copy
// is limited to facts in docs/research_docs/claims_and_evidence.md: this is a
// research prototype, so the page leads with measurements, includes the
// negative results, and ends with what the project does NOT claim.
//
// Scroll is the scan. One hologram of the SRI24 cortex (ScrollBrain) is fixed
// behind the whole page. It starts in the hero, then follows the reader down:
// it recedes behind each section, turns as the page moves, and a scan plane
// sweeps through it from crown to base, finishing beside the closing call to
// action. Each section's own motion is scroll-linked too, and fits what it
// shows: the ledger's figures count up and its rules draw, the architecture's
// data flow draws stage by stage, the pipeline's timeline fills while a scan
// line uncovers the ground-truth label, and the not-claimed items are struck
// through. A section rail on wide screens tracks where the reader is.
//
// Under prefers-reduced-motion none of this runs: the hero keeps the still
// HeroBrain in its column and every section renders in its final state.
//
// Layering: ScrollBrain is `fixed z-0`; all page content sits in a
// `relative z-10` wrapper above it (and AppShell's header and footer are
// z-10 too), so the brain is always behind the words.
import { lazy, Suspense, useCallback, useRef, useState, type ReactNode, type RefObject } from "react";
import { motion, useMotionValue, useReducedMotion, useScroll, useTransform } from "motion/react";
import { AppShell } from "../../components/AppShell";
import { navigateTo } from "../../lib/navigate";
import { isPlainLeftClick } from "../../lib/shellStatus";
import { Architecture } from "./ArchitectureFlow";
import { Ledger } from "./Ledger";
import { NotClaimedSection } from "./NotClaimed";
import { PipelineStory } from "./PipelineStory";
import { useRevealProgress } from "./scrollHooks";
import { SectionRail } from "./SectionRail";
// three.js / fiber live behind these imports so the headline paints first.
const HeroBrain = lazy(() => import("./HeroBrain").then((m) => ({ default: m.HeroBrain })));
const ScrollBrain = lazy(() => import("./ScrollBrain").then((m) => ({ default: m.ScrollBrain })));

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

interface HeroProps {
  reduce: boolean;
  anchorRef: RefObject<HTMLDivElement | null>;
  caption: string | null;
}

/** Load: children fade and rise 12px, 60ms apart. Scroll: the three text
 * layers drift up at different speeds (headline fastest) and fade, so the
 * brain behind is left on stage. Each layer is two elements - the outer one
 * takes the load entrance, the inner one the scroll offset - so the two
 * animations never fight over the same transform. */
function Hero({ reduce, anchorRef, caption }: HeroProps) {
  const ref = useRef<HTMLElement>(null);
  const { scrollYProgress } = useScroll({ target: ref, offset: ["start start", "end start"] });
  const still = useMotionValue(0);
  const p = reduce ? still : scrollYProgress;
  const headY = useTransform(p, [0, 1], [0, -170]);
  const bodyY = useTransform(p, [0, 1], [0, -110]);
  const ctaY = useTransform(p, [0, 1], [0, -60]);
  // Every range here spans the whole 0..1. This offset pair is one motion
  // hands to the browser's native ScrollTimeline, and there a partial range
  // like [0, 0.6] -> [1, 0] gets an implicit final keyframe at the element's
  // own value, so opacity drifted back up to 1 by the end of the hero.
  const fade = useTransform(p, [0, 0.6, 1], [1, 0, 0]);
  const cueFade = useTransform(p, [0, 0.1, 1], [1, 0, 0]);

  const item = {
    hidden: reduce ? { opacity: 1, y: 0 } : { opacity: 0, y: 12 },
    show: { opacity: 1, y: 0, transition: { duration: 0.5, ease: EASE_OUT_QUART } },
  };
  return (
    <section
      id="overview"
      ref={ref}
      className={`${WRAP} relative grid min-h-[calc(100svh-4rem)] items-center gap-12 py-12 lg:grid-cols-[minmax(0,1fr)_minmax(0,600px)] lg:gap-8`}
    >
      <motion.div
        className="flex flex-col items-start"
        initial="hidden"
        animate="show"
        variants={{ show: { transition: { staggerChildren: reduce ? 0 : 0.06 } } }}
      >
        <motion.div variants={item}>
          <motion.h1
            className="max-w-[18ch] font-heading font-bold text-text-primary lg:max-w-[15ch]"
            style={{
              fontSize: "clamp(2.4rem, 4.6vw, 4.25rem)",
              letterSpacing: "-0.02em",
              lineHeight: 1.05,
              textWrap: "balance",
              y: headY,
              opacity: fade,
            }}
          >
            Where a tumour segmentation model can be trusted — and where it breaks.
          </motion.h1>
        </motion.div>
        <motion.div variants={item}>
          <motion.p
            className="mt-6 max-w-[58ch] text-[1.05rem] leading-[1.6] text-text-secondary"
            style={{ y: bodyY, opacity: fade }}
          >
            A 3D brain-tumour segmentation model wrapped in a distribution-free error bound and a refusal gate. The
            bound holds in distribution, fails under distribution shift in proportion to the shift, and the gate
            cannot see cohort-level shift. This project measures where it breaks, and what a new site needs to
            restore it.
          </motion.p>
        </motion.div>
        <motion.div variants={item} className="mt-8">
          <motion.div style={{ y: ctaY, opacity: fade }}>
            <CtaPair />
          </motion.div>
        </motion.div>
        <motion.div variants={item}>
          <motion.p className="mt-4 text-sm text-text-dim" style={{ y: ctaY, opacity: fade }}>
            Research prototype on BraTS 2021 — not for clinical use.
          </motion.p>
        </motion.div>
      </motion.div>

      {reduce ? (
        <Suspense
          fallback={
            <div className="mx-auto flex aspect-square w-full max-w-[640px] items-center justify-center text-sm text-text-dim">
              Loading brain surface…
            </div>
          }
        >
          <HeroBrain />
        </Suspense>
      ) : (
        // The brain itself is drawn by ScrollBrain, fixed behind the page; this
        // box is where it starts, and where it can be dragged to spin it.
        <motion.figure className="m-0" style={{ opacity: fade }}>
          <div
            ref={anchorRef}
            role="img"
            aria-label="Hologram of a brain's cortical surface. Drag sideways to turn it."
            className="relative mx-auto aspect-square w-full max-w-[640px] cursor-grab touch-pan-y select-none active:cursor-grabbing"
          >
            {!caption && (
              <div className="absolute inset-0 flex items-center justify-center">
                <span className="text-sm text-text-dim">Loading brain surface…</span>
              </div>
            )}
          </div>
          <figcaption className="mt-2 text-xs text-text-dim">{caption ?? " "}</figcaption>
        </motion.figure>
      )}

      {!reduce && (
        <motion.div
          aria-hidden="true"
          className="pointer-events-none absolute bottom-6 left-1/2 hidden -translate-x-1/2 flex-col items-center gap-2 md:flex"
          style={{ opacity: cueFade }}
        >
          <span className="font-mono text-[11px] text-text-dim">Scroll to scan</span>
          <span className="relative h-10 w-px overflow-hidden bg-surface-seam">
            <span className="scroll-cue absolute inset-x-0 top-0 h-3 bg-brand-primary" />
          </span>
        </motion.div>
      )}
    </section>
  );
}

/** The bookend: the scan has finished and the brain comes back to full
 * brightness on the right, so this section is tall and its words sit left. */
function Closing() {
  const ref = useRef<HTMLDivElement>(null);
  const p = useRevealProgress(ref, ["start 0.95", "start 0.55"]);
  const y = useTransform(p, [0, 1], [40, 0]);
  const opacity = useTransform(p, [0, 1], [0.3, 1]);
  return (
    <section id="run" data-brain-pose="closing" className={`${WRAP} flex min-h-[85svh] items-center py-24`}>
      <motion.div ref={ref} className="max-w-[30rem]" style={{ y, opacity }}>
        <h2 className="font-heading text-4xl font-semibold text-text-primary md:text-5xl">
          Run a study through the pipeline
        </h2>
        <div className="mt-8">
          <CtaPair />
        </div>
      </motion.div>
    </section>
  );
}

export function Landing() {
  const reduce = useReducedMotion() ?? false;
  const anchorRef = useRef<HTMLDivElement>(null);
  const [caption, setCaption] = useState<string | null>(null);
  const onCaption = useCallback((c: string) => setCaption(c), []);
  return (
    <AppShell>
      {!reduce && (
        <Suspense fallback={null}>
          <ScrollBrain anchorRef={anchorRef} onCaption={onCaption} />
        </Suspense>
      )}
      <div className="relative z-10">
        <Hero reduce={reduce} anchorRef={anchorRef} caption={caption} />
        <Ledger wrap={WRAP} />
        <Architecture wrap={WRAP} />
        <PipelineStory />
        <NotClaimedSection wrap={WRAP} />
        <Closing />
      </div>
      <SectionRail />
    </AppShell>
  );
}
