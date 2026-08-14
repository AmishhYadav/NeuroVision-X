import type { HealthResponse } from "../api";

/**
 * `eval_dir` is an arbitrary path (`NVX_EVAL_DIR`), and this project
 * routinely evaluates on val as well as test - so the split shown here must
 * come from the directory name actually in use, never be hardcoded.
 */
function splitLabel(evalDir: string): string {
  const segments = evalDir.split(/[\\/]/).filter(Boolean);
  const basename = segments[segments.length - 1] ?? evalDir;

  // Match whole WORDS, never substrings. The default directory is
  // `eval_test_baseline_unet3d`, and "eval" contains "val" -- a substring
  // test labels the test split as "val split", which is a false statement
  // about which data is on screen. Same trap as the BraTS modality suffixes,
  // where `"_t1" in name` is also true for `_t1ce`.
  const words = basename.toLowerCase().split(/[^a-z0-9]+/).filter(Boolean);
  if (words.includes("val")) return "val split";
  if (words.includes("test")) return "test split";
  return basename;
}

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
      {/* The wordmark never wraps: at ~600px it otherwise breaks mid-word into
          "NEUROVISION-" / "X" and pushes the header to two lines. The meta
          strip truncates instead of wrapping, for the same reason. */}
      <h1 className="font-condensed shrink-0 text-sm font-semibold tracking-[0.12em] whitespace-nowrap text-text-primary uppercase">
        NeuroVision-X
      </h1>
      <div className="min-w-0 truncate font-mono text-xs text-text-secondary">
        {health ? (
          <span>
            {health.experiment} · {splitLabel(health.eval_dir)} · {health.case_count} cases
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
