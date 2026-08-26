// The landing page. Light theme, deliberately separate from the tool's dark
// clinical chrome (see index.css's --color-landing-* tokens) - a marketing
// surface and a clinical instrument are allowed different registers.
//
// The one exception to "light theme throughout" is PipelineStory, a
// full-bleed dark scroll section: the taste skill's Page Theme Lock allows
// exactly one deliberate full theme-block moment per page (not random
// alternation), and a cinematic dark scroll story is that moment here.
//
// Copy is scoped to what is actually built as of this writing - checked
// against CLAUDE.md's "do not write" list and docs/paper/claims_and_evidence.md
// so the page never claims more than the pipeline does. The "what this does
// not claim" section below quotes the pipeline's own NOT_CLAIMED block
// (src/neurovision/reporting/report.py) rather than writing new limitations
// copy from scratch.
import { useReducedMotion } from "motion/react";
import { HeroBrain } from "./HeroBrain";
import { PipelineStory } from "./PipelineStory";

const DISCLAIMER =
  "This tool is a research and educational decision-support artifact. It is not a diagnostic tool and must not be used, alone or together with any other information, to make or support a clinical decision about any patient.";

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

function openViewer() {
  window.history.pushState({}, "", "/app");
  window.dispatchEvent(new PopStateEvent("popstate"));
}

function ViewerLink({
  className,
  variant = "ghost",
  children,
}: {
  className?: string;
  variant?: "ghost" | "solid";
  children: React.ReactNode;
}) {
  const base =
    variant === "solid"
      ? "bg-landing-accent text-landing-accent-ink hover:opacity-90"
      : "border border-landing-seam text-landing-text hover:border-landing-text-dim";
  return (
    <a
      href="/app"
      onClick={(e) => {
        e.preventDefault();
        openViewer();
      }}
      className={`inline-flex items-center justify-center rounded-full px-6 py-2.5 font-mono text-xs font-semibold transition-all duration-[120ms] hover:scale-[1.02] active:scale-[0.98] ${base} ${className ?? ""}`}
    >
      {children}
    </a>
  );
}

function Nav() {
  return (
    <nav className="fixed top-0 right-0 left-0 z-30 flex justify-center px-6 py-5">
      <div className="liquid-glass-light flex w-full max-w-4xl items-center justify-between rounded-full border border-landing-seam px-5 py-2.5">
        <span className="font-condensed text-sm font-semibold tracking-[0.12em] text-landing-text uppercase">
          NeuroVision-X
        </span>
        <ViewerLink variant="ghost" className="!px-4 !py-1.5">
          Open the viewer
        </ViewerLink>
      </div>
    </nav>
  );
}

function Hero() {
  const reduceMotion = useReducedMotion();
  const rise = (delayMs: number) =>
    reduceMotion ? undefined : ({ animationDelay: `${delayMs}ms` } as const);

  return (
    <section className="relative grid min-h-[100dvh] grid-cols-1 items-center gap-8 px-6 pt-28 pb-12 lg:grid-cols-[minmax(0,560px)_1fr] lg:gap-4 lg:px-16">
      <div className="hero-rise flex flex-col gap-6" style={rise(150)}>
        <span className="font-mono text-xs tracking-[0.14em] text-landing-accent uppercase">
          Brain tumour segmentation, MICCAI BraTS 2021
        </span>
        <h1 className="font-condensed text-5xl leading-[1.02] tracking-tight text-landing-text md:text-7xl">
          Segmentation that refuses what it cannot handle.
        </h1>
        <p className="max-w-[48ch] text-base leading-relaxed text-landing-text-secondary md:text-lg">
          Dual-encoder CNN and Swin Transformer segmentation for multi-modal MRI, with calibrated
          input gating and an atlas-based anatomical report.
        </p>
        <div className="mt-2">
          <ViewerLink variant="solid">Open the viewer</ViewerLink>
        </div>
      </div>

      <div className="hero-rise h-[56vh] min-h-[380px] lg:h-[78vh]" style={rise(320)}>
        <HeroBrain />
      </div>
    </section>
  );
}

function StatSection() {
  return (
    <section className="mx-auto max-w-3xl px-6 py-20 text-center">
      <p className="tabular font-condensed text-6xl text-landing-accent md:text-7xl">+0.0267</p>
      <p className="mx-auto mt-4 max-w-md text-base leading-relaxed text-landing-text-secondary">
        Dice on the enhancing-tumour region, against a parameter- and width-matched U-Net
        baseline. p = 1.4×10⁻²¹, n = 189 held-out test cases.
      </p>
    </section>
  );
}

function NotClaimedSection() {
  return (
    <section className="mx-auto max-w-3xl px-6 py-20">
      <span className="font-mono text-xs tracking-[0.14em] text-landing-text-dim uppercase">
        What this does not claim
      </span>
      <h2 className="mt-3 font-condensed text-3xl text-landing-text">
        Every pipeline has a boundary. Here is this one's.
      </h2>
      <dl className="mt-8 flex flex-col gap-6 border-t border-landing-seam pt-8">
        {NOT_CLAIMED.map((item) => (
          <div key={item.what}>
            <dt className="font-mono text-sm font-semibold text-landing-text">{item.what}</dt>
            <dd className="mt-1.5 text-sm leading-relaxed text-landing-text-secondary">{item.why}</dd>
          </div>
        ))}
      </dl>
    </section>
  );
}

function Footer() {
  return (
    <footer className="border-t border-landing-seam bg-landing-bg-raised px-6 py-14">
      <div className="mx-auto flex max-w-3xl flex-col items-center gap-6 text-center">
        <p className="max-w-md font-mono text-xs leading-relaxed text-landing-text-dim">{DISCLAIMER}</p>
        <ViewerLink variant="solid">Open the viewer</ViewerLink>
      </div>
    </footer>
  );
}

export function Landing() {
  return (
    <div className="min-h-screen bg-landing-bg text-landing-text">
      <Nav />
      <Hero />
      <StatSection />
      <PipelineStory />
      <NotClaimedSection />
      <Footer />
    </div>
  );
}
