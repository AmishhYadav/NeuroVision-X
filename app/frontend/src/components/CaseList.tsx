import { PanelLeftClose } from "lucide-react";
import type { CaseSummary } from "../api";

interface CaseListProps {
  cases: CaseSummary[];
  selectedCaseId: string | null;
  onSelect: (caseId: string) => void;
  /** Show the collapse control in the header (only when the sidebar can itself be collapsed - see App.tsx's caseListCollapsed). */
  collapsible?: boolean;
  onCollapse?: () => void;
}

function formatDice(v: number | null): string {
  return v === null ? "—" : v.toFixed(2);
}

export function CaseList({
  cases,
  selectedCaseId,
  onSelect,
  collapsible,
  onCollapse,
}: CaseListProps) {
  return (
    <div className="flex h-full flex-col overflow-hidden">
      <div className="flex shrink-0 items-center px-3 pt-3 pb-2">
        <div className="eyebrow">Cases</div>
        {collapsible && (
          <button
            type="button"
            onClick={onCollapse}
            aria-label="Hide cases"
            title="Hide cases"
            className="ml-auto rounded-sm p-1 text-text-secondary transition-colors duration-[120ms] hover:text-text-primary"
          >
            <PanelLeftClose size={14} aria-hidden="true" />
          </button>
        )}
      </div>
      <ul className="min-h-0 flex-1 overflow-y-auto">
        {cases.map((c) => {
          const active = c.case_id === selectedCaseId;
          return (
            <li key={c.case_id}>
              <button
                type="button"
                onClick={() => onSelect(c.case_id)}
                className={`flex w-full items-center gap-2 border-l-2 px-3 py-1.5 text-left font-mono text-xs transition-colors duration-[120ms] ${
                  active
                    ? "border-text-primary bg-surface-raised text-text-primary"
                    : "border-transparent text-text-secondary hover:bg-surface-raised/60 hover:text-text-primary"
                }`}
                aria-current={active ? "true" : undefined}
              >
                <span aria-hidden="true" className="text-text-dim">
                  {active ? "▸" : " "}
                </span>
                <span className="truncate">{c.case_id}</span>
                <span className="ml-auto tabular text-text-secondary">
                  {formatDice(c.dice_mean)}
                </span>
              </button>
            </li>
          );
        })}
        {cases.length === 0 && (
          <li className="px-3 py-2 font-mono text-xs text-text-dim">No cases available.</li>
        )}
      </ul>
    </div>
  );
}
