#!/usr/bin/env python3
"""Role E: compare every saved model run. Install first: pip install -e ."""

import argparse

from compress_ecg.audit import write_scorecard


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", default="runs/dummy", help="Use runs in Phase 2")
    parser.add_argument("--output-dir", default="results/dummy_audit", help="Use results/audit in Phase 2")
    args = parser.parse_args()
    print(write_scorecard(args.runs_root, args.output_dir))


if __name__ == "__main__":
    main()
