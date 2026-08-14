import type { HealthResponse } from "../api";

interface HeaderProps {
  health: HealthResponse | null;
  reachable: boolean;
  onToggleCaseList: () => void;
  showCaseListToggle: boolean;
}

export function Header({ health, reachable, onToggleCaseList, showCaseListToggle }: HeaderProps) {
  return (
    <header className="flex h-12 shrink-0 items-center gap-4 border-b border-surface-seam bg-surface-panel px-4">
      {showCaseListToggle && (
        <button
          type="button"
          onClick={onToggleCaseList}
          className="rounded-sm border border-surface-seam px-2 py-1 font-condensed text-[11px] tracking-[0.12em] text-text-secondary uppercase transition-colors duration-[120ms] hover:border-text-dim hover:text-text-primary"
        >
          Cases
        </button>
      )}
      <h1 className="font-condensed text-sm font-semibold tracking-[0.12em] text-text-primary uppercase">
        NeuroVision-X
      </h1>
      <div className="font-mono text-xs text-text-secondary">
        {health ? (
          <span>
            {health.experiment} · test split · {health.case_count} cases
          </span>
        ) : (
          <span className="text-text-dim">connecting…</span>
        )}
      </div>
      <div className="ml-auto flex items-center gap-2 font-mono text-xs text-text-secondary">
        <span
          className={`h-2 w-2 rounded-full ${reachable ? "bg-data-oedema" : "bg-data-enhancing"}`}
          aria-hidden="true"
        />
        <span>{reachable ? "online" : "offline"}</span>
      </div>
    </header>
  );
}
