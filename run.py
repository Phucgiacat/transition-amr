#!/usr/bin/env python3
"""
run.py - Batch AMR parsing entry point.

Usage examples:
  # Run from scratch (no progress tracking):
  python3 run.py --input "train.en(2).jsonl" --output "train-amr.txt"

  # Run with progress tracking (resume on GPU disconnect):
  python3 run.py --input "train.en(2).jsonl" --output "train-amr.txt" --process "process.json"

  # Use a different model:
  python3 run.py --input "train.en(2).jsonl" --output "train-amr.txt" --process "process.json" --model "AMR2-structbart-L"

  # Start fresh, ignoring an existing progress file:
  python3 run.py --input "train.en(2).jsonl" --output "train-amr.txt" --process "process.json" --no-resume
"""

import argparse
import os
import sys


def _build_parser():
    p = argparse.ArgumentParser(
        prog="run.py",
        description="Batch AMR parsing with resume support (IBM transition-amr-parser).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--input",
        required=True,
        metavar="FILE",
        help='Input JSONL file - each line: {"sent": "sentence text"}',
    )
    p.add_argument(
        "--output",
        required=True,
        metavar="FILE",
        help="Output AMR text file.",
    )
    p.add_argument(
        "--process",
        default=None,
        metavar="FILE",
        help=(
            "Progress tracking JSON file.  "
            "Omit this flag (or pass null) to run from scratch without checkpointing.  "
            "If the file already exists the run will resume automatically."
        ),
    )
    p.add_argument(
        "--model",
        default="AMR3-structbart-L",
        metavar="MODEL",
        help=(
            "Pre-trained model name (default: AMR3-structbart-L).  "
            "Other option: AMR2-structbart-L."
        ),
    )
    p.add_argument(
        "--no-resume",
        dest="no_resume",
        action="store_true",
        help="Start from the beginning even if a progress file already exists.",
    )
    p.add_argument(
        "--batch-size",
        dest="batch_size",
        type=int,
        default=8,
        metavar="N",
        help="Number of sentences per parsing batch (default: 8).",
    )
    return p


def main():
    args = _build_parser().parse_args()

    # Basic validation
    if not os.path.isfile(args.input):
        print(f"[ERROR] Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    if args.batch_size < 1:
        print("[ERROR] --batch-size must be >= 1", file=sys.stderr)
        sys.exit(1)

    # Ensure output directory exists
    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)

    # Delegate to core inference module
    from inference import run_batch_inference  # noqa: PLC0415

    run_batch_inference(
        input_file=args.input,
        output_file=args.output,
        process_file=args.process,          # None when --process is omitted
        model_name=args.model,
        no_resume=args.no_resume,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
