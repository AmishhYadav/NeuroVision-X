// One hook every scroll-linked landing section uses.
import { useMotionValue, useReducedMotion, useScroll, type MotionValue } from "motion/react";
import type { RefObject } from "react";

type ScrollOffset = NonNullable<Parameters<typeof useScroll>[0]>["offset"];

/** 0 -> 1 as `target` scrolls through `offset` (motion's "<target edge>
 * <viewport edge>" pairs, e.g. ["start 0.9", "start 0.5"]: from the target's
 * top at 90% down the viewport to its top at 50%). Scroll-linked, so it runs
 * backwards when the reader scrolls back up.
 *
 * Under prefers-reduced-motion it is a constant 1: every section renders in
 * its final, settled state and nothing moves. */
export function useRevealProgress(target: RefObject<HTMLElement | null>, offset: ScrollOffset): MotionValue<number> {
  const reduce = useReducedMotion();
  const { scrollYProgress } = useScroll({ target, offset });
  const settled = useMotionValue(1);
  return reduce ? settled : scrollYProgress;
}
