#!/usr/bin/env bash
# Sourced helper (not run directly) shared by run.sbatch and nnunet.sh.
#
# nvx_check_wall_limit <hours> <variable-name>
#   Fails (return 1) unless <hours> is at least 0.5 h below the SLURM wall
#   limit. The limit comes from $SLURM_JOB_END_TIME (epoch seconds), else from
#   `squeue -h -j $SLURM_JOB_ID -o %L` (time left, [D-]HH:MM:SS), else it only
#   WARNS (returns 0) because the limit is unknown -- never silent.
#   Why 0.5 h: preflight checks, an epoch overshooting its estimate and the
#   final checkpoint scan all happen after the self-imposed time budget.
nvx_check_wall_limit() {
    local hours="$1" name="$2" remaining="" left=""
    if [ -n "${SLURM_JOB_END_TIME:-}" ] && [ "$SLURM_JOB_END_TIME" -gt 0 ] 2>/dev/null; then
        remaining="$(awk -v e="$SLURM_JOB_END_TIME" -v n="$(date +%s)" 'BEGIN { print (e - n) / 3600 }')"
    elif [ -n "${SLURM_JOB_ID:-}" ] && command -v squeue >/dev/null 2>&1; then
        left="$(squeue -h -j "$SLURM_JOB_ID" -o %L 2>/dev/null | head -n1 | tr -d ' ' || true)"
        case "$left" in
            "" | UNLIMITED | INVALID | NOT_SET) ;;
            *)
                remaining="$(echo "$left" | awk '{
                    d = 0; t = $0
                    if (index(t, "-")) { split(t, a, "-"); d = a[1]; t = a[2] }
                    n = split(t, p, ":"); s = 0
                    for (i = 1; i <= n; i++) s = s * 60 + p[i]
                    print (d * 86400 + s) / 3600 }')"
                ;;
        esac
    fi
    if [ -z "$remaining" ]; then
        echo "warning: could not read the SLURM wall limit; make sure $name <= the job time limit minus 0.5 h." >&2
        return 0
    fi
    if ! awk -v m="$hours" -v r="$remaining" 'BEGIN { exit !(m <= r - 0.5) }'; then
        echo "error: $name=$hours leaves less than 0.5 h before the SLURM wall limit" >&2
        echo "(~$remaining h left). The job would be killed mid-epoch; lower $name or raise --time." >&2
        return 1
    fi
    return 0
}
