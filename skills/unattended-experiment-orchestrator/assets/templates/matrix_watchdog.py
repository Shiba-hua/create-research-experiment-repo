"""Matrix watchdog: one long CPU job that follows every series job of a matrix (Slurm).

Uses squeue/scontrol (sacct is often unavailable on compute nodes) plus the files series jobs write.
Every --interval-s seconds: rewrite status.md, append new ALERTs/suspicious flags to events.log,
resubmit a series once from its first unfinished level if it vanished without series_end, submit a
same-node cleanup job for finished jobs, and write DONE when every series has ended.

  python3 matrix_watchdog.py --matrix-dir DIR --series-root DIR --series-script PATH --cleanup-script PATH
"""
import argparse
import json
import subprocess
import time
from pathlib import Path


def sh(*argv):
    p = subprocess.run(argv, capture_output=True, text=True)
    return p.returncode, p.stdout.strip()


def jsonl(path: Path):
    out = []
    if path.is_file():
        for line in path.read_text().splitlines():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def queue_state(job):
    rc, out = sh("squeue", "-h", "-j", job, "-o", "%T|%N")
    if rc == 0 and out:
        return tuple(out.split("|", 1))
    rc, out = sh("scontrol", "show", "job", job, "-o")
    if rc == 0 and "JobState=" in out:
        f = dict(kv.split("=", 1) for kv in out.split() if "=" in kv)
        return f.get("JobState", "UNKNOWN"), f.get("NodeList", "")
    return "GONE", ""


class Watchdog:
    def __init__(self, a):
        self.a, self.dir = a, a.matrix_dir
        self.state_path = self.dir / "watchdog_state.json"
        self.state = (json.loads(self.state_path.read_text()) if self.state_path.is_file()
                      else {"resubmitted": {}, "cleaned": [], "seen": {}})

    def log(self, kind, **d):
        with (self.dir / "events.log").open("a") as fh:
            fh.write(json.dumps({"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "kind": kind, **d},
                                ensure_ascii=False) + "\n")

    def series(self):
        rows = {}
        for line in (self.dir / "registry.tsv").read_text().splitlines():
            if line.strip() and not line.startswith("#"):
                sid, job, task, hw, node_args, extra, *_ = line.split("\t")
                rows.setdefault(sid, []).append(dict(sid=sid, job=job, task=task, hw=hw,
                                                     node_args=node_args, extra=extra))
        return rows

    def resubmit(self, s, start_level):
        export = f"ALL,TASK={s['task']},HW_TYPE={s['hw']},SERIES_ID={s['sid']},START_LEVEL={start_level}"
        if s["extra"]:
            export += "," + s["extra"]
        rc, out = sh("sbatch", "--parsable", "--job-name", f"x-{s['sid']}",
                     "--output", str(self.a.series_root / s["sid"] / "slurm-%j.out"),
                     "--export", export, *s["node_args"].split(), str(self.a.series_script))
        if rc == 0:
            with (self.dir / "registry.tsv").open("a") as fh:
                fh.write("\t".join([s["sid"], out.split(";")[0], s["task"], s["hw"], s["node_args"], s["extra"],
                                    f"resubmit_from_{start_level}"]) + "\n")
        return rc, out

    def step(self):
        lines = ["| series | job | state | node | level | heartbeat age | done | not measured | flags | alerts |",
                 "|---|---|---|---|---|---|---|---|---|---|"]
        all_ended = True
        for sid, jobs in self.series().items():
            s = jobs[-1]
            state, node = queue_state(s["job"])
            run = self.a.series_root / sid / f"job-{s['job']}"
            st = jsonl(run / "series_status.jsonl")
            done = sorted({e["level"] for j in jobs
                           for e in jsonl(self.a.series_root / sid / f"job-{j['job']}" / "series_status.jsonl")
                           if e.get("event") == "point_done"})
            not_measured = [e["level"] for e in st if e.get("event") == "not_measured"]
            ended = any(e.get("event") == "series_end" for e in st)
            hb = run / "heartbeat"
            hb_age = int(time.time() - hb.stat().st_mtime) if hb.is_file() else None
            cur = (run / "current_level").read_text().strip() if (run / "current_level").is_file() else ""
            flags = []
            for summ in sorted(run.glob("p-*/point_summary.json")):
                for flag in json.loads(summ.read_text()).get("flags") or []:
                    flags.append(f"{summ.parent.name}:{flag}")
                    key = f"{sid}:{summ.parent.name}:{flag}"
                    if flag.startswith("suspicious") and key not in self.state["seen"]:
                        self.state["seen"][key] = 1
                        self.log("ALERT_SUSPICIOUS_POINT", series=sid, point=summ.parent.name, flag=flag)
            alerts = jsonl(run / "alerts.log")
            for al in alerts:
                key = f"{sid}:{al.get('utc')}:{al.get('kind')}"
                if key not in self.state["seen"]:
                    self.state["seen"][key] = 1
                    self.log("ALERT_JOB", series=sid, **al)
            if state == "RUNNING" and hb_age is not None and hb_age > 300:
                self.log("ALERT_HEARTBEAT_STALE", series=sid, job=s["job"], age=hb_age)
            finished = state not in ("PENDING", "RUNNING", "CONFIGURING", "COMPLETING")
            if finished:
                if s["job"] not in self.state["cleaned"] and node:
                    rc, out = sh("sbatch", "--parsable", "--nodelist", node, "--cpus-per-task=1", "--mem=1G",
                                 "--time=00:10:00", "--output", str(self.dir / f"cleanup-{s['job']}.out"),
                                 "--export", f"ALL,TARGET_JOB={s['job']}", str(self.a.cleanup_script))
                    self.state["cleaned"].append(s["job"])
                    self.log("cleanup_submitted", job=s["job"], node=node, rc=rc)
                if not ended and sid not in self.state["resubmitted"]:
                    levels = self.a.levels.split()
                    start = next((lv for lv in levels if int(lv) not in done), None)
                    if start is not None:
                        rc, out = self.resubmit(s, start)
                        self.state["resubmitted"][sid] = out
                        self.log("resubmitted", series=sid, old_job=s["job"], start_level=start, rc=rc, new=out)
                elif not ended:
                    self.log("ALERT_SERIES_ABANDONED", series=sid, job=s["job"], state=state)
                    ended = True
            all_ended &= ended and finished
            lines.append(f"| {sid} | {s['job']} | {state} | {node} | {cur} | {hb_age} | {done} | {not_measured} | "
                         f"{'; '.join(flags) or '-'} | {len(alerts)} |")
        (self.dir / "status.md").write_text(
            f"# Matrix status ({time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())})\n\n" + "\n".join(lines) + "\n")
        self.state_path.write_text(json.dumps(self.state, indent=1))
        return all_ended


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--matrix-dir", required=True, type=Path)
    p.add_argument("--series-root", required=True, type=Path)
    p.add_argument("--series-script", required=True, type=Path)
    p.add_argument("--cleanup-script", required=True, type=Path)
    p.add_argument("--levels", default="4 8 16 32 64 128 256")
    p.add_argument("--interval-s", type=float, default=60)
    a = p.parse_args()
    w = Watchdog(a)
    w.log("watchdog_started")
    while not w.step():
        time.sleep(a.interval_s)
    (a.matrix_dir / "DONE").write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ\n", time.gmtime()))
    w.log("matrix_done")


if __name__ == "__main__":
    main()
