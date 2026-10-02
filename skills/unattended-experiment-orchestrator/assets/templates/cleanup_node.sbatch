#!/usr/bin/env bash
#SBATCH --partition=CHANGE_ME
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
# Same-node cleanup of one finished job's private scratch (submitted by the matrix watchdog
# with --nodelist=<node> and TARGET_JOB exported). Container image layers are often unpacked
# read-only, so make them writable before deleting; unmount any leftover FUSE mounts first.
set -u
: "${TARGET_JOB:?}"
D="/tmp/$(id -u)-$TARGET_JOB"
echo "node=$(hostname -s) target=$D exists=$([[ -e $D ]] && echo yes || echo no)"
[[ -e "$D" ]] || exit 0
grep " $D" /proc/mounts | awk '{print $2}' | sort -r | while read -r m; do fusermount -u "$m" 2>/dev/null || true; done
chmod -R u+rwX "$D" 2>/dev/null
rm -rf "$D" && echo removed || echo "partial removal; leftover: $(du -sh "$D" 2>/dev/null)"
