#!/usr/bin/env python3
"""
Simplified wrapper script for AMR parsing
Supports standard command-line interface
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path


def _build_env_with_src(script_dir: Path) -> dict:
    """Return environment with repo src/ prepended to PYTHONPATH."""
    env = os.environ.copy()
    src_dir = (script_dir / 'src').resolve()
    if src_dir.is_dir():
        sep = os.pathsep
        current = env.get('PYTHONPATH', '')
        env['PYTHONPATH'] = f"{src_dir}{sep}{current}" if current else str(src_dir)
    return env


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Run AMR Parser on input file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Quick Start:
  # Run from start
  python3 run.py --input "train.en(2).jsonl" --output "train-amr.txt"
  
  # Resume from checkpoint
  python3 run.py --input "train.en(2).jsonl" --output "train-amr.txt" --process "process.json"
  
  # Start fresh (ignore progress file)
  python3 run.py --input "train.en(2).jsonl" --output "train-amr.txt" --process "process.json" --no-resume
        """
    )
    
    parser.add_argument(
        '--input',
        required=True,
        help='Input JSONL file (e.g., train.en(2).jsonl)'
    )
    
    parser.add_argument(
        '--output',
        required=True,
        help='Output file (e.g., train-amr.txt)'
    )
    
    parser.add_argument(
        '--process',
        default=None,
        help='Progress file to track/resume (if null = no checkpointing)'
    )
    
    parser.add_argument(
        '--model',
        default='AMR3-structbart-L',
        help='Model name (default: AMR3-structbart-L)'
    )
    
    parser.add_argument(
        '--no-resume',
        action='store_true',
        help='Start fresh, ignore existing progress file'
    )
    
    args = parser.parse_args()
    
    # Build inference.py command
    script_dir = Path(__file__).parent
    inference_script = script_dir / 'inference.py'
    
    if not inference_script.exists():
        print(f"Error: inference.py not found at {inference_script}")
        sys.exit(1)
    
    # Build command
    cmd = [
        sys.executable,
        str(inference_script),
        '--input', args.input,
        '--output', args.output,
        '--model', args.model,
    ]
    
    # Add optional arguments
    if args.process:
        cmd.extend(['--process', args.process])
    
    if args.no_resume:
        cmd.append('--no-resume')
    
    # Run
    print(f"Running: {' '.join(cmd)}\n")
    try:
        result = subprocess.run(
            cmd,
            check=False,
            env=_build_env_with_src(script_dir),
        )
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(130)


if __name__ == '__main__':
    main()
