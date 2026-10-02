#!/usr/bin/env bash
#SBATCH --partition=CHANGE_ME
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
# One series = one hardware type x one task; levels measured in increasing order in this job.
# Submitted by submit_matrix.sh with TASK, HW_TYPE, SERIES_ID (+ project extras) exported.
#
# Project hooks to fill in (keep the contract in references/orchestration-patterns.md §4):
#   detect_hw_type      -> prints the hardware type of this node (to catch wrong placement)
#   start_service       -> starts a long-lived server if the task needs one; echoes its pid
#   RUN_POINT_CMD       -> measures one level; writes into $POINT_DIR; honours SIGTERM; exit codes 0/10/11/12
set -uo pipefail
PROJECT=${PROJECT:-/path/to/project}
DATA=${DATA:-/path/to/data/runs}
: "${TASK:?}" "${HW_TYPE:?}" "${SERIES_ID:?}"
LEVELS=${LEVELS:-"4 8 16 32 64 128 256"}
START_LEVEL=${START_LEVEL:-${LEVELS%% *}}
RUN="$DATA/series/$SERIES_ID/job-$SLURM_JOB_ID"
mkdir -p "$RUN"
exec >>"$RUN/series.log" 2>&1
log() { echo "[$(date -u +%FT%TZ)] $*"; }
status() { echo "{\"utc\":\"$(date -u +%FT%TZ)\",\"job\":\"$SLURM_JOB_ID\",$1}" >>"$RUN/series_status.jsonl"; }

detect_hw_type() { echo "$HW_TYPE"; }          # TODO project: map hostname/nvidia-smi to a type
start_service() { echo ""; }                     # TODO project: start server, wait for health, echo pid
RUN_POINT_CMD=${RUN_POINT_CMD:-"echo TODO-run-one-point"}   # TODO project

BG=(); SERVICE_PID=""
JOB_TMP="/tmp/$(id -u)-$SLURM_JOB_ID"
cleanup() {
  log cleanup
  [[ -n "$SERVICE_PID" ]] && kill "$SERVICE_PID" 2>/dev/null
  for p in "${BG[@]}"; do kill "$p" 2>/dev/null; done
  if [[ -d "$JOB_TMP" ]]; then chmod -R u+rwX "$JOB_TMP" 2>/dev/null; rm -rf "$JOB_TMP"; fi
  touch "$RUN/DONE"
}
trap cleanup EXIT
trap 'log "SIGTERM"; status "\"event\":\"terminated\""; exit 143' TERM

[[ "$(detect_hw_type)" == "$HW_TYPE" ]] || { status "\"event\":\"wrong_hw\""; exit 2; }
log "series=$SERIES_ID task=$TASK hw=$HW_TYPE node=$(hostname -s) levels=[$LEVELS] start=$START_LEVEL"
( while true; do date -u +%FT%TZ >"$RUN/heartbeat"; sleep 30; done ) & BG+=($!)
command -v nvidia-smi >/dev/null && { nvidia-smi --query-gpu=timestamp,uuid,utilization.gpu,memory.used,power.draw \
  --format=csv -l 15 >"$RUN/gpu.csv" 2>/dev/null & BG+=($!); }

SERVICE_PID=$(start_service) || { status "\"event\":\"service_start_failed\""; exit 4; }

export TASK HW_TYPE SERIES_ID LEVELS START_LEVEL RUN PROJECT
python3 - <<'PY'
import json, os
e = os.environ
src = os.environ.get("PROJECT", "/path/to/project") + "/.source-commit"
json.dump({k.lower(): e.get(k) for k in ("SERIES_ID", "SLURM_JOB_ID", "TASK", "HW_TYPE", "LEVELS", "START_LEVEL")} |
          {"node": os.uname().nodename,
           "source_commit": open(src).read().strip() if os.path.exists(src) else None},
          open(e["RUN"] + "/run.json", "w"), indent=2)
PY

[[ -n "$SERVICE_PID" ]] && { python3 "$PROJECT/scripts/job_watchdog.py" --run "$RUN" --server-pid "$SERVICE_PID" & BG+=($!); }

STOP=""
for L in $LEVELS; do
  (( L < START_LEVEL )) && continue
  if [[ -n "$STOP" ]]; then status "\"level\":$L,\"event\":\"not_measured\",\"reason\":\"$STOP\""; continue; fi
  echo "$L" >"$RUN/current_level"
  export POINT_DIR="$RUN/p-$L" LEVEL="$L"
  mkdir -p "$POINT_DIR"
  status "\"level\":$L,\"event\":\"point_start\""
  bash -c "$RUN_POINT_CMD" >"$POINT_DIR.stdout" 2>"$POINT_DIR.stderr" &
  P=$!; echo "$P" >"$POINT_DIR/point.pid"; wait "$P"; rc=$?
  if [[ -n "$SERVICE_PID" ]] && ! kill -0 "$SERVICE_PID" 2>/dev/null; then
    STOP="service_died_at_$L"   # TODO project: grep the service log for OOM phrases -> capacity stop
  fi
  case $rc in
    0) status "\"level\":$L,\"event\":\"point_done\"" ;;
    10) STOP="capacity_at_$L"; status "\"level\":$L,\"event\":\"capacity_stop\"" ;;
    *) [[ -z "$STOP" ]] && STOP="failure_rc${rc}_at_$L"; status "\"level\":$L,\"event\":\"point_failed\",\"rc\":$rc" ;;
  esac
  rm -f "$RUN/current_level"
done
status "\"event\":\"series_end\",\"stop\":\"$STOP\""
log "series end stop=${STOP:-none}"
