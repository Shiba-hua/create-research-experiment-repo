#!/usr/bin/env bash
#SBATCH --partition=CHANGE_ME
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=7-00:00:00
# Matrix watchdog (CPU only). submit_matrix.sh exports MATRIX_DIR, SERIES_ROOT, SERIES_SCRIPT.
set -u
: "${MATRIX_DIR:?}" "${SERIES_ROOT:?}" "${SERIES_SCRIPT:?}"
PROJECT=/path/to/project
exec python3 "$PROJECT/scripts/matrix_watchdog.py" --matrix-dir "$MATRIX_DIR" --series-root "$SERIES_ROOT" \
  --series-script "$SERIES_SCRIPT" --cleanup-script "$PROJECT/scripts/cleanup_node.sbatch" \
  --levels "${LEVELS:-4 8 16 32 64 128 256}"
