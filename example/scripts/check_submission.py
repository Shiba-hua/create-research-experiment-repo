#!/usr/bin/env python3
"""Read-only branch/commit gate; this does not approve scientific claims or a merge."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess


CANONICAL = "experiment"
ARCHIVE = "experiment-0913"
TARGET_REFS = {
    CANONICAL: "refs/heads/experiment",
    "origin/experiment": "refs/remotes/origin/experiment",
}


def git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo), *args], text=True, capture_output=True, check=False
    )


def inspect(args: argparse.Namespace) -> dict:
    result = {
        "schema_version": 1,
        "status": "FAIL",
        "source_branch": args.source,
        "target_branch": args.target,
        "source_sha": None,
        "target_sha": None,
        "freeze_required": args.require_frozen,
        "clean_required": args.require_clean,
        "checks": [],
        "scope": "Read-only Git gate; not reviewer identity, scientific acceptance, or server protection.",
    }

    def check(name: str, passed: bool, detail: str) -> bool:
        result["checks"].append({"id": name, "passed": bool(passed), "detail": detail})
        return bool(passed)

    repo = Path(args.repo).resolve()
    inside = git(repo, "rev-parse", "--is-inside-work-tree")
    if not check("worktree", inside.returncode == 0 and inside.stdout.strip() == "true",
                 "A non-bare Git worktree is required."):
        return result

    source_ref = f"refs/heads/{args.source}"
    source_valid = (
        args.source not in {CANONICAL, ARCHIVE, "HEAD"}
        and not args.source.startswith(("-", "refs/"))
        and git(repo, "check-ref-format", source_ref).returncode == 0
    )
    if not check("source_is_topic", source_valid,
                 "Source must be a named local topic branch, never experiment or experiment-0913."):
        return result
    if not check("target_is_experiment", args.target in TARGET_REFS,
                 "Target must be experiment or its origin/experiment tracking ref."):
        return result
    target_ref = TARGET_REFS[args.target]
    result.update(source_ref=source_ref, target_ref=target_ref)

    def resolve(ref: str) -> str | None:
        # An exact local ref avoids DWIM resolution of tags, HEAD, or a supplied revision expression.
        resolved = git(repo, "show-ref", "--verify", "--hash", ref)
        if resolved.returncode != 0:
            return None
        sha = resolved.stdout.strip()
        if git(repo, "cat-file", "-t", sha).stdout.strip() != "commit":
            return None
        return sha

    source_sha, target_sha = resolve(source_ref), resolve(target_ref)
    result.update(source_sha=source_sha, target_sha=target_sha)
    if not check("refs_exist", source_sha is not None and target_sha is not None,
                 "Both exact refs must exist locally and point to commits; no fetch is performed."):
        return result

    frozen = args.expected_source_sha is not None and args.expected_target_sha is not None
    one_frozen = (args.expected_source_sha is None) != (args.expected_target_sha is None)
    check("freeze_pair", not one_frozen and (frozen or not args.require_frozen),
          "Supply both expected SHAs; --require-frozen requires the pair for a reviewed submission.")
    if frozen:
        full_shas = all(re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", sha) is not None
                        for sha in (args.expected_source_sha, args.expected_target_sha))
        check("full_frozen_shas", full_shas, "Frozen revisions must be full lowercase Git object IDs.")
        check("source_sha_unchanged", args.expected_source_sha == source_sha,
              "The source tip must equal the submitted or approved frozen source SHA.")
        check("target_sha_unchanged", args.expected_target_sha == target_sha,
              "The target tip must equal the submitted or approved frozen base SHA; advancement "
              "invalidates that handoff or approval.")

    check("has_topic_commits", source_sha != target_sha, "A submission must contain topic commits.")
    ancestry = git(repo, "merge-base", "--is-ancestor", target_sha, source_sha)
    check("contains_current_target", ancestry.returncode == 0,
          "The current target must be an ancestor of source. Merge/rebase the updated target into "
          "the topic, then freeze both tips and review again if this fails.")
    if args.require_clean:
        state = git(repo, "status", "--porcelain=v1", "--untracked-files=normal")
        check("clean_worktree", state.returncode == 0 and not state.stdout,
              "Tracked and untracked changes must be absent in the inspected worktree.")
    check("refs_stable_during_check", resolve(source_ref) == source_sha and resolve(target_ref) == target_sha,
          "Neither ref may change while this check runs; rerun immediately before merge.")
    result["status"] = "PASS" if all(item["passed"] for item in result["checks"]) else "FAIL"
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=".", help="Git worktree to inspect (default: current directory)")
    parser.add_argument("--source", required=True, help="Exact short name of a local topic branch")
    parser.add_argument("--target", required=True, help="experiment or origin/experiment (already fetched)")
    parser.add_argument("--expected-source-sha", help="Full source commit frozen in the handoff or main review")
    parser.add_argument("--expected-target-sha", help="Full target/base commit frozen in the handoff or main review")
    parser.add_argument("--require-frozen", action="store_true", help="Reject a missing expected SHA pair")
    parser.add_argument("--require-clean", action="store_true", help="Also reject tracked/untracked worktree edits")
    args = parser.parse_args()
    try:
        result = inspect(args)
    except (OSError, subprocess.SubprocessError) as exc:
        result = {"schema_version": 1, "status": "FAIL", "error": str(exc)}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
