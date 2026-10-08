/**
 * AppShell: the header / footer frame shared by every page.
 *
 * It shows the brand, the nav, an optional page toolbar, and a live API
 * status chip (polled from /health every 30 s). Data rule: every word shown
 * here is either fixed project text or returned by the API; nothing invented.
 */
import { Brain } from "lucide-react";
import { useEffect, useState, type ReactNode } from "react";
import { getHealth, type HealthResponse } from "../api";
import { DISCLAIMER } from "../lib/disclaimer";
import { navigateTo } from "../lib/navigate";
import { NAV_ITEMS, apiStatus, isPlainLeftClick, navKeyForPath } from "../lib/shellStatus";

const POLL_MS = 30_000;

const DOT_CLASS = {
  online: "bg-brand-teal",
  degraded: "bg-gate-caution",
  offline: "bg-gate-refuse",
  connecting: "bg-text-dim animate-pulse",
} as const;

export function AppShell({
  children,
  toolbar,
  fullHeight = false,
}: {
  children: ReactNode;
  toolbar?: ReactNode;
  fullHeight?: boolean;
}) {
  const active = navKeyForPath(window.location.pathname);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [reachable, setReachable] = useState<boolean | null>(null);

  // Poll /health on mount and every 30 s. The effect cleanup aborts any
  // in-flight request and stops the timer when the shell unmounts.
  useEffect(() => {
    const controller = new AbortController();
    const poll = async () => {
      try {
        const h = await getHealth(controller.signal);
        if (controller.signal.aborted) return;
        setHealth(h);
        setReachable(true);
      } catch {
        if (controller.signal.aborted) return;
        setReachable(false);
      }
    };
    void poll();
    const timer = window.setInterval(() => void poll(), POLL_MS);
    return () => {
      controller.abort();
      window.clearInterval(timer);
    };
  }, []);

  const status = apiStatus(health, reachable);

  return (
    <div
      className={
        fullHeight
          ? "flex h-full flex-col overflow-hidden bg-surface-page"
          : "flex min-h-full flex-col bg-surface-page"
      }
    >
      <header className="flex h-14 shrink-0 items-center gap-3 border-b border-surface-seam bg-surface-panel/80 px-4 backdrop-blur sm:px-6">
        <a
          href="/"
          onClick={(e) => {
            if (isPlainLeftClick(e)) {
              e.preventDefault();
              navigateTo("/");
            }
          }}
          className="flex shrink-0 items-center gap-3"
        >
          <span className="flex h-8 w-8 items-center justify-center rounded-lg border border-brand-primary/40 bg-brand-primary/10">
            <Brain className="h-4 w-4 text-brand-primary" aria-hidden="true" />
          </span>
          <span className="flex flex-col leading-tight">
            <span className="font-heading font-semibold whitespace-nowrap">NeuroVision-X</span>
            <span className="eyebrow hidden md:block">Research prototype · not for clinical use</span>
          </span>
        </a>

        <nav aria-label="Primary" className="flex min-w-0 flex-1 items-center gap-5 overflow-x-auto px-2 md:px-6">
          {NAV_ITEMS.map((item) => {
            const isActive = item.key === active;
            return (
              <a
                key={item.key}
                href={item.href}
                aria-current={isActive ? "page" : undefined}
                onClick={(e) => {
                  if (isPlainLeftClick(e)) {
                    e.preventDefault();
                    navigateTo(item.href);
                  }
                }}
                className={`relative shrink-0 py-4 text-sm whitespace-nowrap ${
                  isActive ? "text-text-primary" : "text-text-secondary hover:text-text-primary"
                }`}
              >
                {item.label}
                {isActive && (
                  <span className="absolute inset-x-0 bottom-0 h-0.5 bg-brand-primary" aria-hidden="true" />
                )}
              </a>
            );
          })}
        </nav>

        <div className="flex shrink-0 items-center gap-3">
          {toolbar}
          <span className="chip" title={status.detail ?? undefined}>
            <span className={`h-1.5 w-1.5 rounded-full ${DOT_CLASS[status.tone]}`} aria-hidden="true" />
            <span>{status.label}</span>
            {status.detail && <span className="hidden text-text-dim md:inline">{status.detail}</span>}
          </span>
        </div>
      </header>

      <main className={fullHeight ? "min-h-0 flex-1" : "flex-1"}>{children}</main>

      {fullHeight ? (
        <div className="shrink-0 border-t border-surface-seam px-4 py-1 text-[11px] text-text-dim">
          {DISCLAIMER}
        </div>
      ) : (
        <footer className="border-t border-surface-seam">
          <div className="mx-auto flex max-w-7xl flex-col gap-4 px-4 py-6 sm:px-6 md:flex-row md:items-start md:justify-between">
            <div className="flex flex-col gap-1">
              <span className="font-heading text-sm">NeuroVision-X</span>
              <span className="eyebrow">BraTS 2021 · 3D tumour segmentation research</span>
            </div>
            <p className="max-w-xl text-xs text-text-dim">{DISCLAIMER}</p>
          </div>
        </footer>
      )}
    </div>
  );
}
