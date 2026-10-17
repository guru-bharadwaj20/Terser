"""Row 11.4.1: duration and pass rate of the real GitHub Actions runs of `.github/workflows/ci.yml`.

Fetches every run of the workflow and the jobs of each run with the `gh` CLI (which must be
authenticated), then splits the runs into

* **executed**: at least one job was picked up by a runner (non-empty ``runner_name`` and steps);
* **not started**: no job reached a runner. On this repository these are the runs GitHub refused
  with "The job was not started because recent account payments have failed or your spending
  limit needs to be increased" (an account billing block, not a test failure).

Run duration = ``updated_at - run_started_at`` of the run (all four matrix jobs run in parallel,
so this is the wall time a PR author waits). Pass rate = success / (success + failure) over
executed, non-cancelled runs; runs cancelled by the workflow's ``concurrency`` group are counted
separately. Per-step medians come from the jobs of executed successful runs.

Usage (stdlib only; works on Windows or Linux)::

    python scripts/ci_run_stats.py --repo guru-bharadwaj20/Terser --out reports/p11_ci.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def gh_api(path: str) -> Any:
    out = subprocess.run(
        ["gh", "api", "--paginate", "--slurp", path],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout
    return json.loads(out)


def ts(s: str | None) -> datetime | None:
    return datetime.fromisoformat(s.replace("Z", "+00:00")) if s else None


def seconds(a: str | None, b: str | None) -> float | None:
    ta, tb = ts(a), ts(b)
    return (tb - ta).total_seconds() if ta and tb else None


def median(xs: list[float]) -> float | None:
    return statistics.median(xs) if xs else None


def collect(repo: str, workflow: str) -> list[dict[str, Any]]:
    pages = gh_api(f"repos/{repo}/actions/workflows/{workflow}/runs?per_page=100")
    runs = [r for page in pages for r in page["workflow_runs"]]
    out = []
    for r in runs:
        jpages = gh_api(f"repos/{repo}/actions/runs/{r['id']}/jobs?per_page=100")
        jobs = [j for page in jpages for j in page["jobs"]]
        out.append(
            {
                "id": r["id"],
                "event": r["event"],
                "branch": r["head_branch"],
                "sha": r["head_sha"],
                "conclusion": r["conclusion"],
                "created_at": r["created_at"],
                "run_started_at": r.get("run_started_at") or r["created_at"],
                "updated_at": r["updated_at"],
                "jobs": [
                    {
                        "name": j["name"],
                        "conclusion": j["conclusion"],
                        "runner": j.get("runner_name") or "",
                        "seconds": seconds(j["started_at"], j["completed_at"]),
                        "steps": [
                            {
                                "name": s["name"],
                                "conclusion": s["conclusion"],
                                "seconds": seconds(s.get("started_at"), s.get("completed_at")),
                            }
                            for s in j.get("steps") or []
                        ],
                    }
                    for j in jobs
                ],
            }
        )
    return out


def summarise(runs: list[dict[str, Any]], recent: int) -> dict[str, Any]:
    for r in runs:
        r["executed"] = any(j["runner"] and j["steps"] for j in r["jobs"])
        r["seconds"] = seconds(r["run_started_at"], r["updated_at"])
    executed = [r for r in runs if r["executed"]]
    finished = [r for r in executed if r["conclusion"] in ("success", "failure")]
    ok = [r for r in finished if r["conclusion"] == "success"]
    step_secs: dict[str, list[float]] = defaultdict(list)
    job_secs: dict[str, list[float]] = defaultdict(list)
    for r in ok:
        for j in r["jobs"]:
            if j["seconds"] is not None:
                job_secs[j["name"]].append(j["seconds"])
            for s in j["steps"]:
                if s["seconds"] is not None:
                    step_secs[s["name"]].append(s["seconds"])
    latest = runs[:recent]
    return {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "runs_total": len(runs),
        "first_run": runs[-1]["created_at"] if runs else None,
        "last_run": runs[0]["created_at"] if runs else None,
        "by_conclusion": dict(Counter(r["conclusion"] for r in runs)),
        "by_event": dict(Counter(r["event"] for r in runs)),
        "executed": len(executed),
        "executed_by_conclusion": dict(Counter(r["conclusion"] for r in executed)),
        "not_started": len(runs) - len(executed),
        "last_executed_run": executed[0]["created_at"] if executed else None,
        "first_not_started_run": min(
            (r["created_at"] for r in runs if not r["executed"]), default=None
        ),
        "pass_rate_executed": len(ok) / len(finished) if finished else None,
        "median_seconds_executed_finished": median([r["seconds"] for r in finished]),
        "median_seconds_success": median([r["seconds"] for r in ok]),
        "max_seconds_success": max((r["seconds"] for r in ok), default=None),
        "success_seconds": sorted(r["seconds"] for r in ok),
        "median_job_seconds_success": {k: median(v) for k, v in sorted(job_secs.items())},
        "median_step_seconds_success": {k: median(v) for k, v in step_secs.items()},
        "recent": {
            "n": len(latest),
            "executed": sum(r["executed"] for r in latest),
            "by_conclusion": dict(Counter(r["conclusion"] for r in latest)),
        },
        "steps_seen_in_executed_runs": sorted(
            {s["name"] for r in executed for j in r["jobs"] for s in j["steps"]}
        ),
        "executed_runs": [
            {
                "id": r["id"],
                "created_at": r["created_at"],
                "sha": r["sha"][:7],
                "conclusion": r["conclusion"],
                "seconds": r["seconds"],
                "failed_steps": sorted(
                    {
                        f"{j['name']}: {s['name']}"
                        for j in r["jobs"]
                        for s in j["steps"]
                        if s["conclusion"] == "failure"
                    }
                ),
                "full_workflow": any(
                    s["name"] == "DVC pipeline dry run" for j in r["jobs"] for s in j["steps"]
                ),
            }
            for r in executed
        ],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--repo", default="guru-bharadwaj20/Terser")
    ap.add_argument("--workflow", default="ci.yml")
    ap.add_argument("--recent", type=int, default=50, help="size of the 'latest runs' window")
    ap.add_argument("--out", type=Path, default=Path("reports/p11_ci.json"))
    args = ap.parse_args()
    runs = collect(args.repo, args.workflow)
    summary = summarise(runs, args.recent)
    summary["repo"] = args.repo
    summary["workflow"] = args.workflow
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    keys = ("runs_total", "executed", "not_started", "pass_rate_executed", "median_seconds_success")
    print(json.dumps({k: summary[k] for k in keys}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
