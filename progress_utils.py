#!/usr/bin/env python3
"""
progress_utils.py - Check and manage AMR parsing progress files.

Usage:
  # Show current progress:
  python3 progress_utils.py --show process.json

  # Reset progress (backs up the existing file first):
  python3 progress_utils.py --reset process.json
"""

import argparse
import json
import os
import shutil
import sys
from datetime import datetime


# ---------------------------------------------------------------------------
# Show
# ---------------------------------------------------------------------------

def show_progress(process_file):
    """Print a human-readable progress report."""
    if not os.path.exists(process_file):
        print(f"[INFO] Progress file not found: {process_file}")
        return

    try:
        with open(process_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError) as exc:
        print(f"[ERROR] Could not read progress file: {exc}", file=sys.stderr)
        return

    processed    = data.get("processed_count", 0)
    total        = data.get("total_count", 0)
    completed    = data.get("completed", False)
    model        = data.get("model", "unknown")
    last_updated = data.get("last_updated", "unknown")
    input_file   = data.get("input_file", "unknown")
    output_file  = data.get("output_file", "unknown")
    last_error   = data.get("last_error")

    pct = (processed / total * 100) if total > 0 else 0
    bar_len = 30
    filled  = int(bar_len * (processed / total)) if total > 0 else 0
    bar     = "#" * filled + "-" * (bar_len - filled)

    status = "COMPLETED" if completed else "IN PROGRESS / PAUSED"

    print()
    print("=" * 56)
    print("  AMR Parsing Progress")
    print("=" * 56)
    print(f"  Input  : {input_file}")
    print(f"  Output : {output_file}")
    print(f"  Model  : {model}")
    print(f"  Status : {status}")
    print(f"  [{bar}] {processed}/{total} ({pct:.1f}%)")
    print(f"  Last updated : {last_updated}")
    if last_error:
        print(f"  Last error   : {last_error}")
    print("=" * 56)

    if not completed and total > 0:
        remaining = total - processed
        print(f"\n  Remaining : {remaining} sentences")
        print(
            f"\n  To resume :\n"
            f"    python3 run.py \\\n"
            f"      --input  {input_file!r} \\\n"
            f"      --output {output_file!r} \\\n"
            f"      --process {process_file!r}"
        )
    print()


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

def reset_progress(process_file):
    """Back up and delete the progress file so the next run starts fresh."""
    if not os.path.exists(process_file):
        print(f"[INFO] Progress file not found: {process_file}")
        return

    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{process_file}.backup_{timestamp}"

    try:
        shutil.copy2(process_file, backup_path)
        os.remove(process_file)
        print(f"[INFO] Backed up -> {backup_path}")
        print(f"[INFO] Removed   -> {process_file}")
        print("[INFO] The next run will start from scratch.")
    except IOError as exc:
        print(f"[ERROR] Reset failed: {exc}", file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        prog="progress_utils.py",
        description="Check and manage AMR parsing progress files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--show",
        metavar="FILE",
        help="Show progress from FILE.",
    )
    group.add_argument(
        "--reset",
        metavar="FILE",
        help="Reset progress - backs up FILE and removes it.",
    )

    args = p.parse_args()

    if args.show:
        show_progress(args.show)
    elif args.reset:
        reset_progress(args.reset)


if __name__ == "__main__":
    main()
