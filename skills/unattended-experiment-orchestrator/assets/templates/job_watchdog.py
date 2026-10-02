"""In-job watchdog: runs in the background of series.sbatch. Writes ALERT lines to <run>/alerts.log.

  STALL               no file under the current point dir changed for --stall-s -> SIGTERM the point program
  NO_COMPLETION_30MIN the point started 30 min ago and events.jsonl has no completed datapoint (alert only)
  SERVER_DOWN         the service pid is gone (alert; the point program should exit on its own)

Progress is judged from file activity so the watchdog stays task-agnostic. If the project has a
better progress signal (e.g. a token counter in metrics.jsonl), replace `last_activity`.

  python3 job_watchdog.py --run RUN [--server-pid PID] [--stall-s 600]
"""
import argparse
import json
import os
import signal
import time
from pathlib import Path


def alert(run: Path, kind: str, **detail):
    rec = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "job": os.environ.get("SLURM_JOB_ID"), "kind": kind, **detail}
    with (run / "alerts.log").open("a") as fh:
        fh.write(json.dumps(rec) + "\n")


def last_activity(point: Path) -> float:
    times = [p.stat().st_mtime for p in point.rglob("*") if p.is_file()]
    return max(times) if times else point.stat().st_mtime


def has_completion(point: Path) -> bool:
    ev = point / "events.jsonl"
    if not ev.is_file():
        return False
    with ev.open() as fh:
        return any('"terminal"' in line for line in fh)


def check(run: Path, server_pid, stall_s: float, state: dict):
    if server_pid:
        try:
            os.kill(server_pid, 0)
        except ProcessLookupError:
            if not state.get("server_down"):
                alert(run, "SERVER_DOWN", server_pid=server_pid)
                state["server_down"] = True
    cur = run / "current_level"
    if not cur.is_file():
        return
    level = cur.read_text().strip()
    point = run / f"p-{level}"
    if not point.is_dir():
        return
    now = time.time()
    started = point.stat().st_ctime
    if now - last_activity(point) > stall_s and state.get("stalled") != level:
        alert(run, "STALL", level=level, idle_seconds=int(now - last_activity(point)))
        state["stalled"] = level
        pid_file = point / "point.pid"
        if pid_file.is_file():
            try:
                os.kill(int(pid_file.read_text()), signal.SIGTERM)
            except (ProcessLookupError, ValueError):
                pass
    if now - started > 1800 and not has_completion(point) and state.get("no_completion") != level:
        alert(run, "NO_COMPLETION_30MIN", level=level)
        state["no_completion"] = level


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, type=Path)
    p.add_argument("--server-pid", type=int)
    p.add_argument("--stall-s", type=float, default=600)
    p.add_argument("--interval-s", type=float, default=60)
    a = p.parse_args()
    state = {}
    while not (a.run / "DONE").exists():
        check(a.run, a.server_pid, a.stall_s, state)
        time.sleep(a.interval_s)


if __name__ == "__main__":
    main()
