// Real scroll-scrubbed storytelling, not discrete fade-ins: the real MRI
// slice sits pinned full-bleed in the background and grows from a small
// centred frame out past the edges of the viewport as the page scrolls
// through this section, while four stages of the real pipeline cross-fade
// over it in sequence. One continuous scroll progress value
// (useScroll + useTransform, Motion only - see the taste skill's warning
// against mixing GSAP and Motion in one tree) drives every layer, each at
// its own rate, the same idea as a classic multi-layer parallax reveal.
import { motion, useReducedMotion, useScroll, useTransform } from "motion/react";
import { useRef } from "react";
import { useHeroSliceImages } from "./heroSliceImages";

interface Stage {
  title: string;
  body: string;
  range: [number, number, number, number]; // fade-in start, hold start, hold end, fade-out end
}

const STAGES: Stage[] = [
  {
    title: "Four sequences, one frame",
    body: "T1, T1CE, T2 and FLAIR, co-registered, skull-stripped and atlas-aligned before the model ever sees them.",
    range: [0.04, 0.1, 0.2, 0.26],
  },
  {
    title: "Dual encoder, gated fusion",
    body: "A 3D CNN and a Swin Transformer read the same volume in parallel; an adaptive gated cross-attention block decides how much of each to trust, per location.",
    range: [0.28, 0.34, 0.44, 0.5],
  },
  {
    title: "Three heads, not one mask",
    body: "The decoder outputs a segmentation, a confidence estimate and a boundary map together, not a single class prediction bolted onto uncertainty after the fact.",
    range: [0.52, 0.58, 0.68, 0.74],
  },
  {
    title: "A gate that can say no",
    body: "An input quality check and a calibrated refusal gate run before the segmentation is trusted - every threshold measured against held-out data, not picked by hand.",
    range: [0.76, 0.82, 0.94, 0.99],
  },
];

function StageCaption({ stage, progress }: { stage: Stage; progress: import("motion/react").MotionValue<number> }) {
  const [a, b, c, d] = stage.range;
  const opacity = useTransform(progress, [a, b, c, d], [0, 1, 1, 0]);
  const y = useTransform(progress, [a, b, c, d], [24, 0, 0, -24]);
  return (
    <motion.div style={{ opacity, y }} className="pointer-events-none absolute inset-x-0 bottom-[14%] px-6 text-center">
      <span className="font-mono text-[11px] text-white/50">{stage.title}</span>
      <p className="mx-auto mt-2 max-w-lg text-base leading-relaxed text-white md:text-lg">{stage.body}</p>
    </motion.div>
  );
}

export function PipelineStory() {
  const sectionRef = useRef<HTMLDivElement>(null);
  const reduceMotion = useReducedMotion();
  const { baseUrl, revealUrl } = useHeroSliceImages();
  const { scrollYProgress } = useScroll({
    target: sectionRef,
    offset: ["start start", "end end"],
  });

  const imageScale = useTransform(scrollYProgress, [0, 1], reduceMotion ? [1, 1] : [0.42, 1.22]);
  const overlayOpacity = useTransform(scrollYProgress, [0.18, 0.32], [0, 1]);
  const vignetteOpacity = useTransform(scrollYProgress, [0, 0.08, 0.92, 1], [1, 0, 0, 1]);
  // The image layer and the caption layer drift at slightly different rates
  // - the multi-speed-layers idea, done with two useTransform outputs off
  // the same progress value instead of two separately-clocked animations.
  const imageY = useTransform(scrollYProgress, [0, 1], reduceMotion ? ["0%", "0%"] : ["6%", "-6%"]);

  return (
    <section ref={sectionRef} className="relative h-[420vh]">
      <div className="sticky top-0 h-screen overflow-hidden bg-black">
        <motion.div style={{ scale: imageScale, y: imageY }} className="absolute inset-0 flex items-center justify-center">
          {baseUrl && (
            <img
              src={baseUrl}
              alt="Raw MRI slice"
              className="h-full w-full object-cover"
              style={{ imageRendering: "pixelated" }}
            />
          )}
          {revealUrl && (
            <motion.img
              src={revealUrl}
              alt="Same slice with the real segmentation overlay"
              style={{ opacity: overlayOpacity, imageRendering: "pixelated" }}
              className="absolute inset-0 h-full w-full object-cover"
            />
          )}
        </motion.div>

        <motion.div style={{ opacity: vignetteOpacity }} className="pointer-events-none absolute inset-0 bg-black" />
        <div className="pointer-events-none absolute inset-0 bg-gradient-to-t from-black/85 via-black/10 to-black/60" />

        {STAGES.map((stage) => (
          <StageCaption key={stage.title} stage={stage} progress={scrollYProgress} />
        ))}
      </div>
    </section>
  );
}
