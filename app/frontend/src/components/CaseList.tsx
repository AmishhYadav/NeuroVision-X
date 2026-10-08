import { PanelLeftClose } from "lucide-react";
import type { CaseSummary } from "../api";

interface CaseListProps {
  cases: CaseSummary[];
  selectedCaseId: string | null;
  onSelect: (caseId: string) => void;
  /** Server-reported size of the full test set (showcase response only). */
  total?: number;
  /** The server's own description of how `cases` were chosen, shown verbatim. */
  selection?: string;
  /** A case opened by deep link that is not among `cases`; shown as an extra "linked" row. */
  linkedCase?: CaseSummary | null;
  /** Show the collapse control in the header (only when the sidebar can itself be collapsed - see App.tsx's caseListCollapsed). */
  collapsible?: boolean;
  onCollapse?: () => void;
}

function formatDice(v: number | null): string {
  return v === null ? "—" : v.toFixed(2);
}

function formatVolume(v: number | undefined): string {
  return v === undefined ? "—" : `${v.toFixed(1)} mL`;
}

export function CaseList({
  cases,
  selectedCaseId,
  onSelect,
  total,
  selection,
  linkedCase,
  collapsible,
  onCollapse,
}: CaseListProps) {
  return (
    <div className="glass-panel flex h-full flex-col overflow-hidden">
      <div className="flex shrink-0 items-center px-3 pt-3 pb-2">
        <div className="eyebrow">Cases</div>
        {collapsible && (
          <button
            type="button"
            onClick={onCollapse}
            aria-label="Hide cases"
            title="Hide cases"
            className="ml-auto rounded-md border border-surface-seam p-1 text-text-secondary transition-colors duration-[120ms] hover:border-brand-primary/60 hover:text-text-primary"
          >
            <PanelLeftClose size={14} aria-hidden="true" />
          </button>
        )}
      </div>
      {/* Columns: case id, ground-truth WT volume (wt_volume_ml), CaseSummary.dice_mean. */}
      <div className="shrink-0 border-b border-surface-seam px-3 pb-1.5">
        <span className="eyebrow">Case · GT tumour volume · Mean Dice</span>
      </div>
      <ul className="min-h-0 flex-1 overflow-y-auto">
        {[...cases, ...(linkedCase ? [linkedCase] : [])].map((c) => {
          const active = c.case_id === selectedCaseId;
          const linked = linkedCase !== null && linkedCase !== undefined && c === linkedCase;
          return (
            <li key={c.case_id}>
              <button
                type="button"
                onClick={() => onSelect(c.case_id)}
                className={`flex w-full items-center gap-2 px-3 py-1.5 text-left font-mono text-xs transition-colors duration-[120ms] ${
                  active
                    ? "bg-brand-primary/12 text-text-primary ring-1 ring-inset ring-brand-primary/35"
                    : "text-text-secondary hover:bg-surface-raised/50 hover:text-text-primary"
                }`}
                aria-current={active ? "true" : undefined}
              >
                <span aria-hidden="true" className="text-text-dim">
                  {active ? "▸" : " "}
                </span>
                <span className="min-w-0 flex-1 truncate">
                  {c.case_id}
                  {linked && <span className="ml-1 text-text-dim">(linked)</span>}
                </span>
                <span className="tabular text-text-secondary">{formatVolume(c.wt_volume_ml)}</span>
                <span className="tabular w-8 text-right text-text-secondary">
                  {formatDice(c.dice_mean)}
                </span>
              </button>
            </li>
          );
        })}
        {cases.length === 0 && (
          <li className="px-3 py-2 text-xs text-text-dim">No cases available.</li>
        )}
      </ul>
      {total !== undefined && selection !== undefined && (
        <p className="shrink-0 border-t border-surface-seam px-3 py-2 text-xs text-text-dim">
          {`Showing ${cases.length} of ${total} test cases, chosen by ${selection}.`}
        </p>
      )}
    </div>
  );
}
