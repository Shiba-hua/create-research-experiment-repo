#!/usr/bin/env bash
# Read-only status for humans on the login node:  status.sh <matrix_name>
D=/path/to/data/runs/matrices/${1:?matrix name}
cat "$D/status.md" 2>/dev/null || echo "no status board yet: $D"
echo; echo "== recent alerts =="; grep ALERT "$D/events.log" 2>/dev/null | tail -10
echo; [[ -f "$D/DONE" ]] && echo "matrix finished at $(cat "$D/DONE")" || squeue -u "$USER" -o '%.8i %.32j %.8T %.10M %R'
