// A slim section index fixed to the right edge on wide screens: one tick per
// section, an ice fill that tracks how far down the page the reader is, and
// the current section's name beside its tick. It appears once the hero has
// scrolled away. Each tick is a real link to its section, so it doubles as
// navigation. Hidden below 1440px, where it would crowd the 1180px column.
import { useEffect, useState } from "react";
import { motion, useMotionValueEvent, useReducedMotion, useScroll, useTransform } from "motion/react";
import { clamp01 } from "../../lib/scrollScene";

const SECTIONS: { id: string; label: string }[] = [
  { id: "overview", label: "Overview" },
  { id: "measurements", label: "Measurements" },
  { id: "architecture", label: "Architecture" },
  { id: "pipeline", label: "Pipeline" },
  { id: "not-claimed", label: "Not claimed" },
  { id: "run", label: "Run a study" },
];

/** The section whose top has passed 45% of the viewport most recently. */
function activeIndex(scrollY: number): number {
  const line = scrollY + window.innerHeight * 0.45;
  let idx = 0;
  SECTIONS.forEach((s, i) => {
    const el = document.getElementById(s.id);
    if (el && el.getBoundingClientRect().top + scrollY <= line) idx = i;
  });
  return idx;
}

export function SectionRail() {
  const reduce = useReducedMotion();
  const { scrollY, scrollYProgress } = useScroll();
  const [active, setActive] = useState(0);

  const shown = useTransform(scrollY, (y) => clamp01((y - window.innerHeight * 0.35) / (window.innerHeight * 0.3)));
  // Fully hidden (not just transparent) at the top, so it is not tabbable there.
  const visibility = useTransform(shown, (o) => (o < 0.02 ? "hidden" : "visible"));

  useMotionValueEvent(scrollY, "change", (y) => {
    const idx = activeIndex(y);
    setActive((prev) => (prev === idx ? prev : idx));
  });
  useEffect(() => {
    setActive(activeIndex(window.scrollY));
  }, []);

  return (
    <motion.nav
      aria-label="Page sections"
      className="fixed top-1/2 right-6 z-20 hidden -translate-y-1/2 min-[1440px]:block"
      style={{ opacity: shown, visibility }}
    >
      <ol className="relative m-0 flex list-none flex-col gap-5 p-0">
        <span aria-hidden="true" className="absolute top-1 right-[3px] bottom-1 w-px bg-surface-seam" />
        <motion.span
          aria-hidden="true"
          className="absolute top-1 right-[3px] bottom-1 w-px origin-top bg-brand-primary"
          style={{ scaleY: scrollYProgress }}
        />
        {SECTIONS.map((s, i) => {
          const isActive = i === active;
          return (
            <li key={s.id}>
              <a
                href={`#${s.id}`}
                aria-current={isActive ? "location" : undefined}
                onClick={(e) => {
                  e.preventDefault();
                  const behavior = reduce ? "auto" : "smooth";
                  if (i === 0) window.scrollTo({ top: 0, behavior });
                  else document.getElementById(s.id)?.scrollIntoView({ behavior, block: "start" });
                }}
                className="group flex items-center justify-end gap-3 font-mono text-[11px] leading-none"
              >
                <span
                  className={`transition-opacity duration-200 ${
                    isActive
                      ? "text-text-secondary opacity-100"
                      : "text-text-dim opacity-0 group-hover:opacity-100 group-focus-visible:opacity-100"
                  }`}
                >
                  {s.label}
                </span>
                <span
                  aria-hidden="true"
                  className={`relative h-[7px] w-[7px] rounded-full border transition-colors duration-200 ${
                    i <= active ? "border-brand-primary bg-brand-primary" : "border-text-dim bg-surface-page"
                  }`}
                />
              </a>
            </li>
          );
        })}
      </ol>
    </motion.nav>
  );
}
