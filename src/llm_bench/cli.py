"""`llm-bench` command-line entrypoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _add_run(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("run", help="run a benchmark suite")
    p.add_argument("--suite", required=True, help="suite name from suites.yaml")
    p.add_argument("--models", required=True, help="comma-separated model names from models.yaml")
    p.add_argument("--limit", type=int, default=None, help="override sample limit per task")
    p.add_argument("--epochs", type=int, default=None, help="override epochs per task")
    p.add_argument("--log-dir", type=Path, default=None)
    p.add_argument("--dry-run", action="store_true", help="resolve and print, run nothing")
    p.add_argument("--force", action="store_true", help="run despite preflight errors")


def _add_report(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("report", help="summarize a log directory")
    p.add_argument("log_dir", type=Path)


def _add_dashboard(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("dashboard", help="render a log directory as an interactive HTML dashboard")
    p.add_argument("log_dir", type=Path)
    p.add_argument("--output", type=Path, default=None, help="default: <log_dir>/dashboard.html")
    p.add_argument("--no-open", action="store_true", help="write the file but don't open a browser")


def _add_profile(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("profile", help="latency @ concurrency=1 and a throughput sweep")
    p.add_argument("--models", required=True, help="comma-separated model names")
    p.add_argument("--concurrencies", default="1,2,4,8,16")
    p.add_argument("--samples-per-level", type=int, default=4)


def _add_resume(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "resume",
        help="resume incomplete/errored runs in a log dir (e.g. after a connectivity drop)",
    )
    p.add_argument("log_dir", type=Path)


def _add_list(sub: argparse._SubParsersAction) -> None:
    sub.add_parser("list-models", help="show configured models")
    sub.add_parser("list-suites", help="show configured suites")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="llm-bench")
    sub = parser.add_subparsers(dest="command", required=True)
    _add_run(sub)
    _add_report(sub)
    _add_dashboard(sub)
    _add_profile(sub)
    _add_resume(sub)
    _add_list(sub)

    args = parser.parse_args(argv)

    if args.command == "run":
        from llm_bench.run import run_suite

        run_suite(
            suite=args.suite,
            model_names=[m.strip() for m in args.models.split(",")],
            limit=args.limit,
            epochs=args.epochs,
            log_dir=args.log_dir,
            dry_run=args.dry_run,
            force=args.force,
        )
        return 0

    if args.command == "report":
        from llm_bench.report import report

        report(args.log_dir)
        return 0

    if args.command == "dashboard":
        from llm_bench.dashboard import dashboard

        dashboard(args.log_dir, output=args.output, open_browser=not args.no_open)
        return 0

    if args.command == "profile":
        from llm_bench.profile import run_profile

        run_profile(
            [m.strip() for m in args.models.split(",")],
            concurrencies=tuple(int(x) for x in args.concurrencies.split(",")),
            samples_per_level=args.samples_per_level,
        )
        return 0

    if args.command == "resume":
        from llm_bench.run import resume_incomplete

        resume_incomplete(args.log_dir)
        return 0

    if args.command in ("list-models", "list-suites"):
        from llm_bench.registry import Registry

        registry = Registry.load()
        if args.command == "list-models":
            for name, m in sorted(registry.models.items()):
                print(f"{name:24s} {m.spec}")
        else:
            for name, s in sorted(registry.suites.items()):
                print(f"{name:12s} {s.get('description', '')}")
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
