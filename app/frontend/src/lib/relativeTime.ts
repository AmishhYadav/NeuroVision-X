// Pure relative-time formatter for `PreviousStudies`' job list ("3 min ago",
// "2 h ago", ...). Takes `nowMs` as a parameter (rather than reading
// `Date.now()` internally) so the vitest suite can assert exact boundaries
// without faking global time.

const SECOND = 1000;
const MINUTE = 60 * SECOND;
const HOUR = 60 * MINUTE;
const DAY = 24 * HOUR;

/**
 * Formats a job's `created_at` (seconds since the Unix epoch, as the backend
 * sends it) relative to `nowMs` (milliseconds since the epoch, defaulting to
 * `Date.now()`).
 *
 * Falls back to a plain UTC calendar date once a job is a day or more old,
 * rather than "N days ago" - a demo audience reading a list of finished
 * studies cares which day, not a rolling count. UTC (not
 * `toLocaleDateString`) so the same job renders the same date regardless of
 * the viewer's timezone or locale, which matters for a screenshot taken on
 * one machine and shown on another.
 */
export function formatRelativeTime(epochSeconds: number, nowMs: number = Date.now()): string {
  const diffMs = nowMs - epochSeconds * 1000;

  if (diffMs < MINUTE) return "just now";
  if (diffMs < HOUR) return `${Math.floor(diffMs / MINUTE)} min ago`;
  if (diffMs < DAY) return `${Math.floor(diffMs / HOUR)} h ago`;

  return new Date(epochSeconds * 1000).toISOString().slice(0, 10);
}
