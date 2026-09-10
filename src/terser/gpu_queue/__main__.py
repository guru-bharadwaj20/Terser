"""CLI: ``python -m terser.gpu_queue validate|dry-run|run [--dir gpu_queue]``."""

from __future__ import annotations

import argparse
import sys

from terser.gpu_queue.queue import QueueError, dry_run, format_dry_run, run, validate


def main(argv: list[str] | None = None) -> int:
    """Entry point; returns the process exit code."""
    parser = argparse.ArgumentParser(prog="python -m terser.gpu_queue", description=__doc__)
    parser.add_argument("command", choices=["validate", "dry-run", "run"])
    parser.add_argument("--dir", default="gpu_queue", help="queue directory (default: gpu_queue)")
    parser.add_argument("--root", default=None, help="repo root (default: current directory)")
    args = parser.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):  # messages contain non-ASCII (em dash) on Windows
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

    try:
        if args.command == "validate":
            errors = validate(args.dir, args.root)
            for e in errors:
                print(f"ERROR {e}", file=sys.stderr)
            print(f"{len(errors)} error(s)" if errors else "queue OK")
            return 1 if errors else 0
        if args.command == "dry-run":
            print(format_dry_run(dry_run(args.dir, args.root)))
            return 0
        state = run(args.dir, args.root)
        return 1 if any(s["status"] == "failed" for s in state.values()) else 0
    except QueueError as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
