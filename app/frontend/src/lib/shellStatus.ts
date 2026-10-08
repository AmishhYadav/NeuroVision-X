/**
 * Pure helpers for AppShell: which nav item is active, how to describe the
 * API health, and whether a click is a plain left click. No React here so
 * everything can be unit-tested in the node environment.
 */
import type { HealthResponse } from "../api";

export type NavKey = "home" | "clinical" | "viewer" | "report";

export const NAV_ITEMS: { key: NavKey; label: string; href: string }[] = [
  { key: "home", label: "Overview", href: "/" },
  { key: "clinical", label: "Clinical study", href: "/clinical" },
  { key: "viewer", label: "Case viewer", href: "/app" },
];

/** Exact path matching (not substring): "/application" is not the viewer. */
export function navKeyForPath(pathname: string): NavKey {
  if (pathname === "/clinical") return "clinical";
  if (pathname === "/app") return "viewer";
  if (pathname.startsWith("/report/")) return "report";
  return "home";
}

export type ApiStatus = {
  tone: "online" | "degraded" | "offline" | "connecting";
  label: string;
  detail: string | null;
};

/** reachable: null = no answer yet, false = request failed, true = answered. */
export function apiStatus(health: HealthResponse | null, reachable: boolean | null): ApiStatus {
  if (reachable === null) return { tone: "connecting", label: "Connecting…", detail: null };
  if (reachable === false || health === null) {
    return { tone: "offline", label: "API offline", detail: null };
  }
  if (!health.checkpoint_present) {
    return { tone: "degraded", label: "API online · no checkpoint", detail: health.experiment };
  }
  const n = health.case_count;
  return {
    tone: "online",
    label: "API online",
    detail: `${health.experiment} · ${n} ${n === 1 ? "case" : "cases"}`,
  };
}

/** True only for an unmodified left click, so cmd/ctrl-click still opens a new tab. */
export function isPlainLeftClick(e: {
  button: number;
  metaKey: boolean;
  ctrlKey: boolean;
  shiftKey: boolean;
  altKey: boolean;
}): boolean {
  return e.button === 0 && !e.metaKey && !e.ctrlKey && !e.shiftKey && !e.altKey;
}
