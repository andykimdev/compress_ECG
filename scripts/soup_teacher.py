#!/usr/bin/env python3
"""
Average the weights of several finished Teachers into one Teacher checkpoint.

Purpose:
- Write a checkpoint that scripts/run_teacher.py --checkpoint can score.

Special Notes:
- Install first: pip install -e .
- The Teachers must share one layout and one starting point, such as seeds of one setting.
"""

import argparse
from pathlib import Path

import torch

from compress_ecg.teacher import average_checkpoints


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoints", nargs="+", required=True, help="best.pt files written by run_teacher.py")
    parser.add_argument("--output", required=True, help="Path of the averaged checkpoint. It must not exist yet")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"{output} already exists. Choose a new path")
    state_dict = average_checkpoints(args.checkpoints)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": state_dict, "sources": args.checkpoints}, output)
    print(f"Averaged {len(args.checkpoints)} checkpoints into {output}")


if __name__ == "__main__":
    main()
