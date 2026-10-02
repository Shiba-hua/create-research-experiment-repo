#!/usr/bin/env bash
# Submit a whole matrix at once: one sbatch job per series + one CPU matrix watchdog.
# The scheduler's per-user quota queues the excess, so everything that can run in parallel does.
#
#   submit_matrix.sh <matrix_name> [task filter regex] [extra KEY=VAL exported to every job ...]
#   e.g. submit_matrix.sh main-20261003 '^taskA$' PRESET=general MTP=0
#
# Edit the four paths below for the project. Run on the login node.
set -euo pipefail
PROJECT=/path/to/project                 # code root on the cluster (deploy target)
DATA=/path/to/data/runs                  # raw records root on the cluster
SERIES_SCRIPT="$PROJECT/scripts/series.sbatch"
WATCHDOG_SCRIPT="$PROJECT/scripts/watchdog.sbatch"

NAME=${1:?matrix name}; FILTER=${2:-.}; shift $(( $# >= 2 ? 2 : $# ))
EXTRA=""; for kv in "$@"; do EXTRA+=",$kv"; done
MDIR="$DATA/matrices/$NAME"
[[ -e "$MDIR" ]] && { echo "matrix $NAME already exists: $MDIR" >&2; exit 2; }
mkdir -p "$MDIR"
cp "$PROJECT/scripts/matrix.tsv" "$MDIR/matrix.tsv"
cat "$PROJECT/.source-commit" >"$MDIR/source-commit" 2>/dev/null || echo unknown >"$MDIR/source-commit"
grep -q dirty "$MDIR/source-commit" && echo "WARNING: deployed code is dirty; commit and redeploy before a formal run" >&2
printf '# series_id\tjob_id\ttask\thw_type\tnode_args\textra\tnote\n' >"$MDIR/registry.tsv"

grep -v '^#' "$PROJECT/scripts/matrix.tsv" | sort -t$'\t' -k5,5n |
while IFS=$'\t' read -r SID TASK HW NODEARGS PRIO; do
  [[ -n "$SID" && "$TASK" =~ $FILTER ]] || continue
  SID="$SID-$NAME"
  mkdir -p "$DATA/series/$SID"
  JOB=$(sbatch --parsable --job-name "x-$SID" --output "$DATA/series/$SID/slurm-%j.out" \
        --export "ALL,TASK=$TASK,HW_TYPE=$HW,SERIES_ID=$SID$EXTRA" $NODEARGS "$SERIES_SCRIPT")
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$SID" "${JOB%%;*}" "$TASK" "$HW" "$NODEARGS" "${EXTRA#,}" initial >>"$MDIR/registry.tsv"
  echo "submitted $SID -> ${JOB%%;*}"
done

WJOB=$(sbatch --parsable --job-name "x-watchdog-$NAME" --output "$MDIR/watchdog-%j.out" \
       --export "ALL,MATRIX_DIR=$MDIR,SERIES_ROOT=$DATA/series,SERIES_SCRIPT=$SERIES_SCRIPT" "$WATCHDOG_SCRIPT")
echo "watchdog -> ${WJOB%%;*}"
echo "status: $PROJECT/scripts/status.sh $NAME"
