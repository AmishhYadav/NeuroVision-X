import { useEffect, useState } from "react";
import type { LayoutMode } from "../components/ViewportGrid";

const STACK_BREAKPOINT = 700;
const PANEL_BREAKPOINT = 1100;

/**
 * Structural (not just visual) responsive breakpoints:
 * - < 700px:  a single viewport with a plane switcher
 * - < 1100px: three viewports stacked, case list collapsed behind a toggle
 * - >= 1100px: three-up viewport row, case list shown inline
 */
export function useResponsiveLayout(): { layout: LayoutMode; isPanelWidth: boolean } {
  const [width, setWidth] = useState<number>(() =>
    typeof window === "undefined" ? PANEL_BREAKPOINT : window.innerWidth,
  );

  useEffect(() => {
    const onResize = () => setWidth(window.innerWidth);
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []);

  const layout: LayoutMode =
    width < STACK_BREAKPOINT ? "single" : width < PANEL_BREAKPOINT ? "stack" : "grid";

  return { layout, isPanelWidth: width >= PANEL_BREAKPOINT };
}
