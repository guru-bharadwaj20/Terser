"""CLI for the Prefect flows (rows 11.6.1, 11.6.2): list, plan, deploy, run, serve, demo."""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from terser.flows.flows import FLOWS, SCHEDULES, module_path, plan
from terser.flows.steps import StepResult


def _print_plan(name: str, results: Sequence[StepResult]) -> None:
    print(f"{name}:")
    for i, r in enumerate(results, 1):
        bad = "FIRED" if r.kind == "trigger" else "REFUSED"  # a trigger that fires is not ok
        status = "planned" if not r.executed else ("ok" if r.ok else bad)
        print(f"  {i}. [{r.kind}] {r.name}: {r.command}  ({status})")
        if r.executed and r.output:
            print(f"     {r.output.splitlines()[0][:200]}")


def _list() -> int:
    from terser.flows.registry import TAGS

    for name, f in FLOWS.items():
        params = ", ".join(inspect.signature(f.fn).parameters)
        print(f"{name}  [{module_path(name)}]")
        print(f"    {(f.description or '').strip().splitlines()[0]}")
        print(f"    schedule: {SCHEDULES.get(name, 'on demand')}; tags: {', '.join(TAGS[name])}")
        print(f"    parameters: {params}")
    return 0


def _parse_params(items: Sequence[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for item in items:
        key, sep, raw = item.partition("=")
        if not sep or not key:
            raise SystemExit(f"--param expects key=value, got {item!r}")
        try:
            out[key] = json.loads(raw)
        except json.JSONDecodeError:
            out[key] = raw
    return out


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point; returns the process exit code."""
    ap = argparse.ArgumentParser(prog="python -m terser.flows", description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="list the registered flows")
    p_plan = sub.add_parser("plan", help="dry-run flows and print their command plans")
    p_plan.add_argument("flows", nargs="*", metavar="FLOW", help=f"default: all of {list(FLOWS)}")
    sub.add_parser("deploy", help="register every flow as a deployment with the Prefect API")
    sub.add_parser("serve", help="serve the deployments (long-running; runs schedules)")
    p_run = sub.add_parser("run", help="run one flow once")
    p_run.add_argument("flow", choices=list(FLOWS))
    p_run.add_argument("--dry-run", action="store_true")
    p_run.add_argument("--param", action="append", default=[], help="key=JSON value")
    p_demo = sub.add_parser("demo", help="registration, plans and injected failures -> report")
    p_demo.add_argument("--out", type=Path, default=Path("reports/p11_flows.md"))
    args = ap.parse_args(argv)

    if args.command == "list":
        return _list()
    if args.command == "plan":
        unknown = sorted(set(args.flows) - set(FLOWS))
        if unknown:
            ap.error(f"unknown flow(s) {unknown}; choose from {list(FLOWS)}")
        for name in args.flows or list(FLOWS):
            _print_plan(name, plan(name))
        return 0
    if args.command == "deploy":
        from terser.flows.registry import register_deployments, registered_triggers

        rows = register_deployments()
        for r in rows:
            print(f"{r['deployment']:45} {r['schedule']:12} {r['entrypoint']}  id={r['id']}")
        print(f"{len(rows)} deployment(s) registered")
        for t in registered_triggers():
            print(f"trigger {t['automation']}: on {t['expect']} [{t['match_related']}]")
            print(f"    -> {t['deployment']} with {t['parameters']}")
        return 0 if len(rows) == len(FLOWS) else 1
    if args.command == "serve":  # pragma: no cover - long-running
        from prefect import serve

        from terser.flows.registry import build_deployments

        serve(*build_deployments())
        return 0
    if args.command == "run":
        params = _parse_params(args.param)
        if args.dry_run:
            _print_plan(args.flow, plan(args.flow, **params))
            return 0
        state = FLOWS[args.flow](**params, return_state=True)
        if state.is_completed():
            _print_plan(args.flow, state.result())
            return 0
        print(f"{args.flow}: {state.name}: {state.message}", file=sys.stderr)
        return 1
    from terser.flows.demo import run_demo

    text = run_demo(args.out)
    print(f"wrote {args.out} ({len(text.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
