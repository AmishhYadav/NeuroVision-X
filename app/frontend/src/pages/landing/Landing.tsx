// The landing page. Dark Stitch look (see index.css tokens). Copy is limited
// to facts in docs/research_docs/claims_and_evidence.md: this is a research
// prototype, so the page leads with measurements, includes the negative
// results, and ends with what the project does NOT claim. The NOT_CLAIMED
// list quotes the pipeline's own block (src/neurovision/reporting/report.py).
import type { ReactNode } from "react";
import { AppShell } from "../../components/AppShell";
import { navigateTo } from "../../lib/navigate";
import { isPlainLeftClick } from "../../lib/shellStatus";
import { HeroBrain } from "./HeroBrain";
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

function SectionHead({ chip, title }: { chip: string; title: string }) {
  return (
    <>
      <span className="chip">{chip}</span>
      <h2 className="mt-3 font-heading text-3xl font-semibold text-text-primary md:text-4xl">{title}</h2>
    </>
  );
}

function Hero() {
  return (
    <section className="bg-grid">
      <div className="mx-auto grid max-w-7xl items-center gap-8 px-4 py-14 sm:px-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,480px)] lg:py-20">
        <div className="flex flex-col items-start gap-6">
          <span className="chip">Research prototype · BraTS 2021</span>
          <h1 className="font-heading text-4xl leading-[1.05] font-bold text-text-primary md:text-6xl">
            Where a tumour segmentation model can be trusted — and where it breaks.
          </h1>
          <p className="max-w-[60ch] text-base leading-relaxed text-text-secondary md:text-lg">
            A 3D brain-tumour segmentation model wrapped in a distribution-free error bound and a refusal gate.
            The bound holds in distribution, fails under distribution shift in proportion to the shift, and the
            gate cannot see cohort-level shift. This project measures where it breaks, and what a new site needs
            to restore it.
          </p>
          <CtaPair />
        </div>
        <figure className="m-0">
          <div className="glass-panel h-[360px] overflow-hidden sm:h-[440px]">
            <HeroBrain />
          </div>
          <figcaption className="mt-2 font-mono text-[11px] text-text-dim">
            Brain surface reconstructed from one BraTS case (BraTS2021_00000) · no tumour shown
          </figcaption>
        </figure>
      </div>
    </section>
  );
}

function StatsRow() {
  return (
    <section className="mx-auto max-w-7xl px-4 py-10 sm:px-6">
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {STATS.map((s) => (
          <div key={s.value} className="glass-panel flex flex-col gap-2 p-5">
            <p className="tabular font-mono text-3xl font-bold text-brand-teal">{s.value}</p>
            <p className="text-sm leading-snug text-text-primary">{s.label}</p>
            <p className="eyebrow !normal-case !tracking-normal">{s.source}</p>
          </div>
        ))}
      </div>
    </section>
  );
}

function Architecture() {
  const cards = [
    { title: "3D CNN encoder", body: "Local texture and boundaries." },
    { title: "Swin Transformer encoder", body: "Long-range context." },
    {
      title: "Gated cross-attention fusion → U-Net decoder",
      body: "Fuses both encoders. Three heads: segmentation, confidence, boundary.",
    },
  ];
  return (
    <section className="mx-auto max-w-7xl px-4 py-16 sm:px-6">
      <SectionHead chip="Model" title="Dual encoder, gated fusion" />
      <div className="mt-8 grid gap-4 md:grid-cols-3">
        {cards.map((c) => (
          <div key={c.title} className="glass-panel flex flex-col gap-2 p-5">
            <h3 className="font-heading text-lg font-semibold text-text-primary">{c.title}</h3>
            <p className="text-sm leading-relaxed text-text-secondary">{c.body}</p>
          </div>
        ))}
      </div>
      <p className="mt-6 font-mono text-sm text-text-secondary">
        34.91M parameters. Most of the ET gain is architectural (+0.0211), not width (+0.0055) — measured against
        a width-matched control.
      </p>
    </section>
  );
}

function Findings() {
  return (
    <section className="mx-auto max-w-7xl px-4 py-16 sm:px-6">
      <SectionHead chip="Results" title="What it found, including the negatives" />
      <ul className="mt-8 grid gap-3 md:grid-cols-2">
        {FINDINGS.map((f) => (
          <li key={f.text} className="glass-panel flex items-start gap-3 p-4">
            <span className={`mt-1.5 h-2.5 w-2.5 shrink-0 rounded-full ${DOT[f.tone]}`} aria-hidden="true" />
            <div>
              <p className="eyebrow">{f.tag}</p>
              <p className="mt-1 text-sm leading-relaxed text-text-primary">{f.text}</p>
            </div>
          </li>
        ))}
      </ul>
    </section>
  );
}

function NotClaimedSection() {
  return (
    <section className="mx-auto max-w-7xl px-4 py-16 sm:px-6">
      <SectionHead chip="Limits" title="What this does not claim" />
      <div className="mt-8 grid gap-4 md:grid-cols-2">
        {NOT_CLAIMED.map((item) => (
          <div key={item.what} className="glass-panel p-5">
            <h3 className="font-heading text-base font-semibold text-text-primary">{item.what}</h3>
            <p className="mt-2 text-sm leading-relaxed text-text-secondary">{item.why}</p>
          </div>
        ))}
      </div>
    </section>
  );
}

function ClosingBand() {
  return (
    <section className="mx-auto max-w-7xl px-4 pt-6 pb-20 sm:px-6">
      <div className="glass-panel flex flex-col items-start gap-5 p-8 md:p-10">
        <h2 className="font-heading text-3xl font-semibold text-text-primary">Run a study through the pipeline</h2>
        <CtaPair />
      </div>
    </section>
  );
}

export function Landing() {
  return (
    <AppShell>
      <Hero />
      <StatsRow />
      <Architecture />
      <PipelineStory />
      <Findings />
      <NotClaimedSection />
      <ClosingBand />
    </AppShell>
  );
}
