"""Portable commands: python -m credit_risk {all,experiment,report,verify}."""

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Beyond AUC reproducible research")
    parser.add_argument("command", choices=["all", "experiment", "report", "verify"])
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Repository root")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.command in {"all", "experiment"}:
        from credit_risk.experiment import run_experiment

        run_experiment(root)
    if args.command in {"all", "report"}:
        from credit_risk.plotting import generate_figures
        from credit_risk.report import build_report

        generate_figures(root)
        build_report(root)
    if args.command in {"all", "report", "verify"}:
        from credit_risk.report import verify_artifacts

        print(verify_artifacts(root))


if __name__ == "__main__":
    main()
