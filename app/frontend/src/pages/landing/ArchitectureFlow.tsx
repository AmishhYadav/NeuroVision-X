// "Dual encoder, gated fusion": the model's data flow, drawn by scroll.
//
// The diagram (same geometry and labels as before) is always present as a
// dim "blueprint". Over it, an ice line draws the flow stage by stage as the
// reader scrolls: the four MRI channels split into the two encoders, merge in
// the fusion block, pass the decoder, and fan out to the three heads. Each
// node lights as the line reaches it and its label comes up to full strength.
// Scroll-linked, so scrolling back up un-draws it.
//
// Two layouts: a horizontal diagram from lg up, a vertical one below, which
// fits a 360px viewport without scaling past 1:1.
import { useRef } from "react";
import { motion, useTransform, type MotionValue } from "motion/react";
import { useRevealProgress } from "./scrollHooks";

/** A stroke drawn while progress runs from `from` to `to`. */
interface Seg {
  d: string;
  from: number;
  to: number;
}
/** A node that lights at progress `at`. */
interface Node {
  x: number;
  y: number;
  at: number;
}
/** A label that comes up to full strength at progress `at`. */
interface Label {
  x: number;
  y: number;
  text: string;
  at: number;
  sub?: boolean;
  anchor?: "start" | "middle";
}
interface Flow {
  viewBox: string;
  segs: Seg[];
  nodes: Node[];
  labels: Label[];
}

const ARIA =
  "4 MRI channels split into a 3D CNN encoder and a Swin Transformer encoder, merge in gated cross-attention fusion, pass through a U-Net decoder, and produce segmentation, confidence and boundary outputs.";

const WIDE: Flow = {
  viewBox: "0 0 1060 290",
  segs: [
    { d: "M200 165 H250", from: 0.0, to: 0.1 },
    { d: "M250 165 V80 H290", from: 0.1, to: 0.28 },
    { d: "M250 165 V250 H290", from: 0.1, to: 0.28 },
    { d: "M500 80 H540 V165", from: 0.36, to: 0.5 },
    { d: "M500 250 H540 V165", from: 0.36, to: 0.5 },
    { d: "M540 165 H590", from: 0.5, to: 0.56 },
    { d: "M590 165 H820", from: 0.56, to: 0.74 },
    { d: "M820 165 H890", from: 0.74, to: 0.8 },
    { d: "M890 165 V80 H925", from: 0.8, to: 0.94 },
    { d: "M890 165 H925", from: 0.8, to: 0.94 },
    { d: "M890 165 V250 H925", from: 0.8, to: 0.94 },
  ],
  nodes: [
    { x: 200, y: 165, at: 0.02 },
    { x: 290, y: 80, at: 0.28 },
    { x: 290, y: 250, at: 0.28 },
    { x: 590, y: 165, at: 0.56 },
    { x: 820, y: 165, at: 0.74 },
    { x: 925, y: 80, at: 0.94 },
    { x: 925, y: 165, at: 0.94 },
    { x: 925, y: 250, at: 0.94 },
  ],
  labels: [
    { x: 0, y: 160, text: "4 MRI channels", at: 0.02 },
    { x: 0, y: 180, text: "(T1, T1CE, T2, FLAIR)", at: 0.02, sub: true },
    { x: 306, y: 76, text: "3D CNN encoder", at: 0.3 },
    { x: 306, y: 96, text: "local texture and boundaries", at: 0.3, sub: true },
    { x: 306, y: 246, text: "Swin Transformer encoder", at: 0.3 },
    { x: 306, y: 266, text: "long-range context", at: 0.3, sub: true },
    { x: 604, y: 150, text: "Gated cross-attention fusion", at: 0.58 },
    { x: 820, y: 196, text: "U-Net decoder", at: 0.76, anchor: "middle" },
    { x: 941, y: 85, text: "segmentation", at: 0.96 },
    { x: 941, y: 170, text: "confidence", at: 0.96 },
    { x: 941, y: 255, text: "boundary", at: 0.96 },
  ],
};

const NARROW: Flow = {
  viewBox: "0 0 328 440",
  segs: [
    { d: "M164 50 V70", from: 0.0, to: 0.08 },
    { d: "M164 70 H78 V96", from: 0.08, to: 0.24 },
    { d: "M164 70 H250 V96", from: 0.08, to: 0.24 },
    { d: "M78 96 V180 H164", from: 0.3, to: 0.5 },
    { d: "M250 96 V180 H164", from: 0.3, to: 0.5 },
    { d: "M164 180 V240", from: 0.5, to: 0.58 },
    { d: "M164 280 V305", from: 0.62, to: 0.68 },
    { d: "M164 305 V345", from: 0.68, to: 0.74 },
    { d: "M164 345 H50 V385", from: 0.74, to: 0.92 },
    { d: "M164 345 V385", from: 0.74, to: 0.92 },
    { d: "M164 345 H278 V385", from: 0.74, to: 0.92 },
  ],
  nodes: [
    { x: 164, y: 50, at: 0.02 },
    { x: 78, y: 96, at: 0.24 },
    { x: 250, y: 96, at: 0.24 },
    { x: 164, y: 240, at: 0.58 },
    { x: 164, y: 305, at: 0.68 },
    { x: 50, y: 385, at: 0.92 },
    { x: 164, y: 385, at: 0.92 },
    { x: 278, y: 385, at: 0.92 },
  ],
  labels: [
    { x: 164, y: 18, text: "4 MRI channels", at: 0.02, anchor: "middle" },
    { x: 164, y: 36, text: "(T1, T1CE, T2, FLAIR)", at: 0.02, sub: true, anchor: "middle" },
    { x: 78, y: 118, text: "3D CNN encoder", at: 0.26, anchor: "middle" },
    { x: 78, y: 136, text: "local texture and", at: 0.26, sub: true, anchor: "middle" },
    { x: 78, y: 152, text: "boundaries", at: 0.26, sub: true, anchor: "middle" },
    { x: 250, y: 118, text: "Swin Transformer", at: 0.26, anchor: "middle" },
    { x: 250, y: 136, text: "encoder", at: 0.26, anchor: "middle" },
    { x: 250, y: 154, text: "long-range context", at: 0.26, sub: true, anchor: "middle" },
    { x: 164, y: 266, text: "Gated cross-attention fusion", at: 0.6, anchor: "middle" },
    { x: 180, y: 309, text: "U-Net decoder", at: 0.7 },
    { x: 50, y: 410, text: "segmentation", at: 0.94, anchor: "middle" },
    { x: 164, y: 410, text: "confidence", at: 0.94, anchor: "middle" },
    { x: 278, y: 410, text: "boundary", at: 0.94, anchor: "middle" },
  ],
};

const NODE_LABEL = "fill-text-primary text-[14px] font-semibold";
const NODE_SUB = "fill-text-dim text-[13px]";

function DrawnSeg({ p, seg }: { p: MotionValue<number>; seg: Seg }) {
  const pathLength = useTransform(p, [seg.from, seg.to], [0, 1]);
  // Hidden at length 0, or the dash trick leaves a dot at the start.
  const opacity = useTransform(p, [seg.from, seg.from + 0.005], [0, 1]);
  return <motion.path d={seg.d} style={{ pathLength, opacity }} />;
}

function LitNode({ p, node }: { p: MotionValue<number>; node: Node }) {
  const scale = useTransform(p, [node.at - 0.03, node.at], [0, 1]);
  const halo = useTransform(p, [node.at - 0.01, node.at + 0.03, node.at + 0.1], [0, 0.5, 0]);
  const haloScale = useTransform(p, [node.at, node.at + 0.1], [1, 2.6]);
  return (
    <>
      <motion.circle cx={node.x} cy={node.y} r="7" className="fill-brand-primary" style={{ opacity: halo, scale: haloScale }} />
      <motion.circle cx={node.x} cy={node.y} r="3.5" className="fill-brand-primary" style={{ scale }} />
    </>
  );
}

function FlowLabel({ p, label }: { p: MotionValue<number>; label: Label }) {
  const opacity = useTransform(p, [label.at - 0.04, label.at], [0.35, 1]);
  return (
    <motion.text
      x={label.x}
      y={label.y}
      textAnchor={label.anchor ?? "start"}
      className={label.sub ? NODE_SUB : NODE_LABEL}
      style={{ opacity }}
    >
      {label.text}
    </motion.text>
  );
}

function FlowDiagram({ flow, className, p }: { flow: Flow; className: string; p: MotionValue<number> }) {
  return (
    <svg viewBox={flow.viewBox} className={className} role="img" aria-label={ARIA}>
      {/* Blueprint: the whole diagram, dim, always visible. */}
      <g fill="none" stroke="currentColor" strokeWidth="1" vectorEffect="non-scaling-stroke" opacity="0.45">
        {flow.segs.map((s) => (
          <path key={s.d} d={s.d} />
        ))}
      </g>
      <g fill="currentColor" opacity="0.6">
        {flow.nodes.map((n) => (
          <circle key={`${n.x}-${n.y}`} cx={n.x} cy={n.y} r="3" />
        ))}
      </g>
      {/* The flow, drawn by scroll. */}
      <g fill="none" className="stroke-brand-primary" strokeWidth="1.5" strokeLinecap="round">
        {flow.segs.map((s) => (
          <DrawnSeg key={s.d} p={p} seg={s} />
        ))}
      </g>
      {flow.nodes.map((n) => (
        <LitNode key={`${n.x}-${n.y}`} p={p} node={n} />
      ))}
      {flow.labels.map((l) => (
        <FlowLabel key={`${l.x}-${l.y}`} p={p} label={l} />
      ))}
    </svg>
  );
}

export function Architecture({ wrap }: { wrap: string }) {
  const diagramRef = useRef<HTMLDivElement>(null);
  const p = useRevealProgress(diagramRef, ["start 0.85", "end 0.5"]);
  const noteOpacity = useTransform(p, [0.9, 1], [0.35, 1]);
  return (
    <section id="architecture" data-brain-pose="architecture" className={`${wrap} pb-24 md:pb-32`}>
      <h2 className="font-heading text-3xl font-semibold text-text-primary md:text-4xl">Dual encoder, gated fusion</h2>
      <div ref={diagramRef} className="mt-10 text-text-dim">
        <FlowDiagram flow={WIDE} p={p} className="hidden h-auto w-full lg:block" />
        <FlowDiagram flow={NARROW} p={p} className="mx-auto block h-auto w-full max-w-[420px] lg:hidden" />
      </div>
      <motion.p className="mt-8 max-w-[70ch] font-mono text-sm text-text-secondary" style={{ opacity: noteOpacity }}>
        34.91M parameters. Most of the ET gain is architectural (+0.0211), not width (+0.0055) — measured against a
        width-matched control.
      </motion.p>
    </section>
  );
}
