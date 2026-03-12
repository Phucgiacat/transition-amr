#!/usr/bin/env python3
"""
Inference script for AMR parsing using IBM Transition AMR Parser
Supports batch processing with progress tracking and recovery
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from datetime import datetime
import traceback

# ── Ensure src/ is in sys.path so fairseq_ext subpackages are importable ──────
# uv's editable install does not always propagate `package_dir={'': 'src'}`
# into the running interpreter's sys.path. Setting the path here is the
# safest fallback and is idempotent when the .pth file already did the job.
_REPO_ROOT = Path(__file__).parent.resolve()
for _candidate in (
    _REPO_ROOT / 'src',
    _REPO_ROOT.parent / 'src',
    Path.cwd() / 'src',
    Path.cwd().parent / 'src',
):
    _candidate = _candidate.resolve()
    if _candidate.is_dir() and str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))
        break
# ──────────────────────────────────────────────────────────────────────────────

try:
    from transition_amr_parser.parse import AMRParser
    from penman import decode, encode
except ImportError as e:
    print(f"Error: {e}")
    print("Please install transition_amr_parser first:")
    print("  pip install -e .")
    sys.exit(1)


class ProgressTracker:
    """Track processing progress and enable recovery"""
    
    def __init__(self, progress_file: Optional[str] = None):
        self.progress_file = progress_file
        self.progress = self._load_progress()
    
    def _load_progress(self) -> Dict:
        """Load progress from file if exists"""
        if self.progress_file and os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Failed to load progress file: {e}")
                return self._create_empty_progress()
        return self._create_empty_progress()
    
    def _create_empty_progress(self) -> Dict:
        """Create new progress tracker"""
        return {
            'start_time': datetime.now().isoformat(),
            'last_update': datetime.now().isoformat(),
            'total_processed': 0,
            'total_failed': 0,
            'current_index': 0,
            'processed_indices': [],
            'failed_indices': [],
            'errors': []
        }
    
    def save(self):
        """Save progress to file"""
        if self.progress_file:
            self.progress['last_update'] = datetime.now().isoformat()
            os.makedirs(os.path.dirname(self.progress_file) or '.', exist_ok=True)
            with open(self.progress_file, 'w', encoding='utf-8') as f:
                json.dump(self.progress, f, indent=2, ensure_ascii=False)
    
    def mark_processed(self, index: int, error: Optional[str] = None):
        """Mark an index as processed"""
        self.progress['current_index'] = index + 1
        if error:
            self.progress['total_failed'] += 1
            self.progress['failed_indices'].append(index)
            self.progress['errors'].append({
                'index': index,
                'error': error,
                'timestamp': datetime.now().isoformat()
            })
        else:
            self.progress['total_processed'] += 1
            self.progress['processed_indices'].append(index)
        self.save()
    
    def get_last_index(self) -> int:
        """Get the last successfully processed index"""
        return self.progress.get('current_index', 0)
    
    def get_stats(self) -> str:
        """Get progress statistics"""
        total = self.progress['total_processed'] + self.progress['total_failed']
        stats = f"\n===== Progress Stats =====\n"
        stats += f"Total Processed: {self.progress['total_processed']}\n"
        stats += f"Total Failed: {self.progress['total_failed']}\n"
        stats += f"Total: {total}\n"
        stats += f"Last Update: {self.progress['last_update']}\n"
        stats += f"========================\n"
        return stats


class AMRInferencer:
    """Inference engine for AMR parsing"""
    
    def __init__(self, 
                 input_file: str, 
                 output_file: str,
                 progress_file: Optional[str] = None,
                 model_name: str = 'AMR3-structbart-L'):
        """
        Initialize inferencer
        
        Args:
            input_file: Path to input JSONL file
            output_file: Path to output file
            progress_file: Path to progress file (optional)
            model_name: AMR parser model name
        """
        self.input_file = input_file
        self.output_file = output_file
        self.progress_file = progress_file
        self.model_name = model_name
        
        # Validate input file
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")
        
        # Initialize components
        self.progress_tracker = ProgressTracker(progress_file)
        self.parser = None
        self.output_handle = None
        self._initialize_parser()
    
    def _initialize_parser(self):
        """Initialize AMR parser model"""
        print(f"Loading model: {self.model_name}...")
        try:
            self.parser = AMRParser.from_pretrained(self.model_name)
            print(f"✓ Model loaded successfully")
        except Exception as e:
            print(f"✗ Failed to load model: {e}")
            traceback.print_exc()
            raise
    
    def _open_output(self):
        """Open output file for writing"""
        os.makedirs(os.path.dirname(self.output_file) or '.', exist_ok=True)
        # Open in append mode if resume, write mode if new
        mode = 'a' if self.progress_tracker.get_last_index() > 0 else 'w'
        self.output_handle = open(self.output_file, mode, encoding='utf-8')
    
    def _close_output(self):
        """Close output file"""
        if self.output_handle:
            self.output_handle.close()
    
    def _read_jsonl(self) -> List[Tuple[int, Dict]]:
        """Read JSONL file and return (index, data) pairs"""
        data = []
        try:
            with open(self.input_file, 'r', encoding='utf-8') as f:
                for idx, line in enumerate(f):
                    if line.strip():
                        try:
                            obj = json.loads(line)
                            data.append((idx, obj))
                        except json.JSONDecodeError as e:
                            print(f"Warning: Failed to parse JSON at line {idx}: {e}")
        except Exception as e:
            print(f"Error reading input file: {e}")
            raise
        return data
    
    def _parse_sentence(self, sentence: str, index: int) -> Optional[str]:
        """
        Parse single sentence and return AMR in Penman format
        
        Returns:
            AMR in Penman format or None if parsing failed
        """
        try:
            tokens, positions = self.parser.tokenize(sentence)
            annotations, machines = self.parser.parse_sentence(tokens)
            amr = machines.get_amr()
            return amr.to_penman(jamr=False, isi=True)
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            print(f"  Error parsing sentence {index}: {error_msg}")
            return None
    
    def _write_result(self, index: int, sentence: str, amr_output: Optional[str]):
        """Write parsing result to output file"""
        self.output_handle.write(f"# ::id id{index}\n")
        self.output_handle.write(f"# ::annotator bart-amr\n")
        self.output_handle.write(f"# ::snt {sentence}\n")
        
        if amr_output:
            self.output_handle.write(amr_output)
        else:
            # Write empty AMR or error indicator
            self.output_handle.write("(x / PARSE_ERROR)\n")
        
        self.output_handle.write("\n")
        self.output_handle.flush()
    
    def run(self, resume: bool = True):
        """
        Run inference on all sentences
        
        Args:
            resume: Whether to resume from last checkpoint
        """
        self._open_output()
        
        try:
            # Read all data
            print(f"\nReading input file: {self.input_file}")
            data = self._read_jsonl()
            total_sentences = len(data)
            print(f"✓ Loaded {total_sentences} sentences")
            
            # Determine starting index
            start_index = 0
            if resume:
                start_index = self.progress_tracker.get_last_index()
                if start_index > 0:
                    print(f"\nResuming from index: {start_index}")
                    print(self.progress_tracker.get_stats())
            
            # Process sentences
            print(f"\nProcessing sentences {start_index} to {total_sentences}...\n")
            
            for idx, (line_idx, obj) in enumerate(data):
                if line_idx < start_index:
                    continue
                
                # Extract sentence
                sentence = obj.get('sent') or obj.get('sentence') or str(obj)
                
                # Parse
                progress_pct = (line_idx / total_sentences * 100)
                print(f"[{line_idx:5d}/{total_sentences}] ({progress_pct:5.1f}%) ", end='', flush=True)
                
                amr_output = self._parse_sentence(sentence, line_idx)
                
                if amr_output:
                    print("✓", flush=True)
                    self._write_result(line_idx, sentence, amr_output)
                    self.progress_tracker.mark_processed(line_idx)
                else:
                    print("✗", flush=True)
                    self.progress_tracker.mark_processed(
                        line_idx, 
                        error="Parsing failed"
                    )
            
            # Final stats
            print(self.progress_tracker.get_stats())
            print(f"Output file: {self.output_file}")
            
        except KeyboardInterrupt:
            print("\n\n⚠ Process interrupted by user")
            print(self.progress_tracker.get_stats())
            print("Run again without --no-resume to continue from checkpoint")
            self.progress_tracker.save()
        except Exception as e:
            print(f"\n\n✗ Error during processing: {e}")
            traceback.print_exc()
            print(self.progress_tracker.get_stats())
            self.progress_tracker.save()
            raise
        finally:
            self._close_output()


def main():
    parser = argparse.ArgumentParser(
        description='AMR Parser Inference Engine',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage - process from start
  python3 inference.py --input train.en.jsonl --output train-amr.txt
  
  # Resume from checkpoint
  python3 inference.py --input train.en.jsonl --output train-amr.txt --process progress.json
  
  # Start fresh (ignore existing progress)
  python3 inference.py --input train.en.jsonl --output train-amr.txt --no-resume
  
  # Custom model
  python3 inference.py --input train.en.jsonl --output train-amr.txt --model AMR2-structbart-L
        """
    )
    
    parser.add_argument(
        '--input',
        required=True,
        help='Input JSONL file with sentences (e.g., train.en.jsonl)'
    )
    
    parser.add_argument(
        '--output',
        required=True,
        help='Output file for AMR results (e.g., train-amr.txt)'
    )
    
    parser.add_argument(
        '--process',
        default=None,
        help='Progress file for tracking (default: null = start from beginning)'
    )
    
    parser.add_argument(
        '--model',
        default='AMR3-structbart-L',
        help='AMR model name (default: AMR3-structbart-L)'
    )
    
    parser.add_argument(
        '--no-resume',
        action='store_true',
        help='Start from beginning even if progress file exists'
    )
    
    args = parser.parse_args()
    
    # Create and run inferencer
    try:
        inferencer = AMRInferencer(
            input_file=args.input,
            output_file=args.output,
            progress_file=args.process,
            model_name=args.model
        )
        inferencer.run(resume=not args.no_resume)
    except Exception as e:
        print(f"\n✗ Fatal error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
