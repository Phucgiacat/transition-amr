"""
inference.py - Core batch AMR inference with resume support.

Handles:
- Loading sentences from JSONL
- Tracking progress to a JSON file (atomic writes)
- Resuming after GPU disconnect / process kill
- Writing AMR output incrementally and flush-safe
"""

import json
import os
import signal
import sys
import tempfile
from datetime import datetime

DEFAULT_BATCH_SIZE = 8


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_sentences(input_file):
    """Load sentences from a JSONL file.

    Accepts lines with key "sent" or "sentence".  Falls back to the first
    value in the dict if neither key is present.
    """
    sentences = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if "sent" in data:
                    sentences.append(data["sent"])
                elif "sentence" in data:
                    sentences.append(data["sentence"])
                else:
                    sentences.append(next(iter(data.values())))
            except json.JSONDecodeError as exc:
                print(
                    f"[WARNING] Skipping malformed line {line_no}: {exc}",
                    file=sys.stderr,
                )
    return sentences


# ---------------------------------------------------------------------------
# Progress helpers
# ---------------------------------------------------------------------------

def load_progress(process_file):
    """Return progress dict from *process_file*, or a fresh default."""
    if process_file is None or not os.path.exists(process_file):
        return {"processed_count": 0, "completed": False}
    try:
        with open(process_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as exc:
        print(
            f"[WARNING] Could not read progress file '{process_file}': {exc}",
            file=sys.stderr,
        )
        return {"processed_count": 0, "completed": False}


def save_progress(process_file, data):
    """Atomically write *data* to *process_file* (temp-then-rename)."""
    if process_file is None:
        return
    data = dict(data)
    data["last_updated"] = datetime.now().isoformat()
    dir_name = os.path.dirname(os.path.abspath(process_file)) or "."
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            dir=dir_name,
            suffix=".tmp",
            delete=False,
            encoding="utf-8",
        ) as tf:
            json.dump(data, tf, indent=2, ensure_ascii=False)
            tmp_path = tf.name
        os.replace(tmp_path, process_file)
    except Exception as exc:  # noqa: BLE001
        print(f"[WARNING] Could not save progress: {exc}", file=sys.stderr)
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def format_amr_entry(idx, sentence, annotation_str, model_name):
    """Return a formatted AMR corpus entry (string) for one sentence."""
    annotation_str = (annotation_str or "").strip()

    # Strip any pre-existing comment header lines from the parser output
    # so we don't duplicate them.
    graph_lines = [
        line for line in annotation_str.split("\n")
        if not line.startswith("# ")
    ]
    graph_str = "\n".join(graph_lines).strip()

    return (
        f"# ::id {idx}\n"
        f"# ::annotator {model_name}\n"
        f"# ::snt {sentence}\n"
        f"{graph_str}\n\n"
    )


def count_entries_in_output(output_file):
    """Count completed AMR entries (# ::id lines) in an existing output file."""
    if not os.path.exists(output_file):
        return 0
    count = 0
    try:
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("# ::id "):
                    count += 1
    except IOError:
        pass
    return count


# ---------------------------------------------------------------------------
# Main inference runner
# ---------------------------------------------------------------------------

def run_batch_inference(
    input_file,
    output_file,
    process_file=None,
    model_name="AMR3-structbart-L",
    no_resume=False,
    batch_size=DEFAULT_BATCH_SIZE,
):
    """Run batch AMR parsing with incremental output and GPU-resume support.

    Args:
        input_file:   Path to input JSONL file.
        output_file:  Path to output AMR text file.
        process_file: Path to progress JSON file (None = no checkpointing).
        model_name:   AMR model name.
        no_resume:    If True start from the beginning, ignoring progress file.
        batch_size:   Sentences per batch.
    """

    # ------------------------------------------------------------------ #
    # 1. Load sentences                                                    #
    # ------------------------------------------------------------------ #
    print(f"[INFO] Loading sentences from: {input_file}")
    sentences = load_sentences(input_file)
    total = len(sentences)
    print(f"[INFO] Total sentences: {total}")
    if total == 0:
        print("[INFO] No sentences found. Exiting.")
        return

    # ------------------------------------------------------------------ #
    # 2. Determine start index (resume support)                           #
    # ------------------------------------------------------------------ #
    start_idx = 0
    if not no_resume and process_file is not None:
        progress = load_progress(process_file)
        if progress.get("completed", False):
            print(f"[INFO] Already completed ({total} sentences). Nothing to do.")
            return
        saved_count = progress.get("processed_count", 0)
        # Cross-check with actual output file to guard against corruption
        actual_in_output = count_entries_in_output(output_file)
        start_idx = min(saved_count, actual_in_output)
        if start_idx != saved_count:
            print(
                f"[WARNING] Progress file says {saved_count} done, but "
                f"output file has {actual_in_output} entries. "
                f"Resuming from {start_idx}."
            )
        if start_idx > 0:
            print(f"[INFO] Resuming from sentence {start_idx}/{total}")

    if start_idx >= total:
        print("[INFO] All sentences already processed. Done.")
        return

    # ------------------------------------------------------------------ #
    # 3. Load parser model                                                #
    # ------------------------------------------------------------------ #
    print(f"[INFO] Loading AMR parser: {model_name} ...")
    from transition_amr_parser.parse import AMRParser  # noqa: PLC0415
    parser = AMRParser.from_pretrained(model_name)
    print("[INFO] Model loaded.")

    # ------------------------------------------------------------------ #
    # 4. Interrupt handler                                                #
    # ------------------------------------------------------------------ #
    interrupted = False

    def _handle_signal(sig, frame):  # noqa: ANN001
        nonlocal interrupted
        interrupted = True
        print("\n[INFO] Signal received - saving progress and exiting gracefully ...")

    signal.signal(signal.SIGINT, _handle_signal)
    try:
        signal.signal(signal.SIGTERM, _handle_signal)
    except (OSError, ValueError):
        pass

    # ------------------------------------------------------------------ #
    # 5. Open output file and process                                     #
    # ------------------------------------------------------------------ #
    output_mode = "a" if start_idx > 0 else "w"
    processed = start_idx

    print(
        f"[INFO] Output -> {output_file!r}  (mode={output_mode!r})\n"
        f"[INFO] Processing {total - start_idx} remaining sentences "
        f"(batch_size={batch_size}) ...\n"
    )

    with open(output_file, output_mode, encoding="utf-8") as out_f:
        i = start_idx
        while i < total and not interrupted:
            batch_end = min(i + batch_size, total)
            batch_sentences = sentences[i:batch_end]

            # ----------------------------------------------------------
            # Attempt batch parse; fall back to one-by-one on error
            # ----------------------------------------------------------
            try:
                batch_tokens = [parser.tokenize(s)[0] for s in batch_sentences]
                annotations, _ = parser.parse_sentences(batch_tokens)
            except Exception as batch_exc:  # noqa: BLE001
                print(
                    f"\n[WARNING] Batch [{i}:{batch_end}] failed "
                    f"({batch_exc}). Falling back to single-sentence mode.",
                    file=sys.stderr,
                )
                # Re-try each sentence individually
                annotations = []
                for s in batch_sentences:
                    try:
                        toks, _ = parser.tokenize(s)
                        ann, _ = parser.parse_sentence(toks)
                        annotations.append(ann)
                    except Exception as single_exc:  # noqa: BLE001
                        print(
                            f"\n[WARNING] Failed to parse: {s[:60]!r} - {single_exc}",
                            file=sys.stderr,
                        )
                        annotations.append(
                            f"(PARSE-ERROR :snt {json.dumps(s)})"
                        )

            # ----------------------------------------------------------
            # Write results
            # ----------------------------------------------------------
            for j, (sentence, annotation) in enumerate(
                zip(batch_sentences, annotations)
            ):
                entry = format_amr_entry(
                    i + j, sentence, str(annotation), model_name
                )
                out_f.write(entry)

            # Flush + fsync so data survives a GPU disconnect
            out_f.flush()
            try:
                os.fsync(out_f.fileno())
            except OSError:
                pass

            processed = batch_end

            # ----------------------------------------------------------
            # Save progress
            # ----------------------------------------------------------
            if process_file is not None:
                save_progress(
                    process_file,
                    {
                        "processed_count": processed,
                        "total_count": total,
                        "input_file": input_file,
                        "output_file": output_file,
                        "model": model_name,
                        "completed": processed >= total,
                    },
                )

            # Progress bar
            pct = processed / total * 100
            bar_len = 30
            filled = int(bar_len * processed / total)
            bar = "#" * filled + "-" * (bar_len - filled)
            print(
                f"\r[{bar}] {processed}/{total} ({pct:.1f}%) ",
                end="",
                flush=True,
            )

            i = batch_end

    print()  # newline after progress bar

    if interrupted:
        print(f"[INFO] Interrupted at sentence {processed}/{total}. Progress saved.")
        sys.exit(0)

    # Final completion record
    print(f"[INFO] Completed! {processed}/{total} sentences parsed.")
    print(f"[INFO] Output: {output_file}")
    if process_file is not None:
        save_progress(
            process_file,
            {
                "processed_count": total,
                "total_count": total,
                "input_file": input_file,
                "output_file": output_file,
                "model": model_name,
                "completed": True,
            },
        )
