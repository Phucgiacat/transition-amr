import json
import os
import re
import signal
import sys
import tempfile
from datetime import datetime
from typing import Dict, List, Optional

DEFAULT_BATCH_SIZE = 8

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None


def load_sentences(input_file: str) -> List[str]:
    """Load sentences from a JSONL file."""
    sentences: List[str] = []
    # Use utf-8-sig to transparently drop BOM if present in the first line.
    with open(input_file, "r", encoding="utf-8-sig") as file_obj:
        for line_no, line in enumerate(file_obj, 1):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"[WARNING] malformed line {line_no}: {exc}", file=sys.stderr)
                continue

            if "sent" in data:
                sentences.append(data["sent"])
            elif "sentence" in data:
                sentences.append(data["sentence"])
            else:
                sentences.append(next(iter(data.values())))
    return sentences


def load_progress(process_file: Optional[str]) -> Dict[str, object]:
    """Load progress state from JSON file."""
    if process_file is None or not os.path.exists(process_file):
        return {"processed_count": 0, "completed": False}

    try:
        with open(process_file, "r", encoding="utf-8") as file_obj:
            return json.load(file_obj)
    except Exception:
        return {"processed_count": 0, "completed": False}


def save_progress(process_file: Optional[str], data: Dict[str, object]) -> None:
    """Save progress to JSON atomically."""
    if process_file is None:
        return

    payload = dict(data)
    payload["last_updated"] = datetime.now().isoformat()
    dir_name = os.path.dirname(os.path.abspath(process_file)) or "."

    with tempfile.NamedTemporaryFile(
        "w",
        dir=dir_name,
        suffix=".tmp",
        delete=False,
        encoding="utf-8",
    ) as file_obj:
        json.dump(payload, file_obj, indent=2, ensure_ascii=False)
        tmp_path = file_obj.name

    os.replace(tmp_path, process_file)


def count_entries_in_output(output_file: str) -> int:
    """Count how many AMR entries exist in output file."""
    if not os.path.exists(output_file):
        return 0

    count = 0
    with open(output_file, "r", encoding="utf-8") as file_obj:
        for line in file_obj:
            if line.startswith("# ::id "):
                count += 1
    return count


def format_amr_entry(
    idx: int,
    sentence: str,
    annotation_str: str,
    model_name: str,
    strip_alignments: bool,
) -> str:
    """Format one AMR output block."""
    graph_lines = [
        line for line in (annotation_str or "").split("\n")
        if not line.startswith("# ")
    ]
    graph_str = "\n".join(graph_lines).strip()
    if strip_alignments:
        # Remove JAMR-style alignments, e.g. "boy~3" or "want-01~e.4"
        graph_str = re.sub(r"~(?:e\.)?\d+", "", graph_str)
    return (
        f"# ::id {idx}\n"
        f"# ::annotator {model_name}\n"
        f"# ::snt {sentence}\n"
        f"{graph_str}\n\n"
    )


def _build_progress_bar(total: int, start_idx: int):
    """Create tqdm progress bar; fallback when tqdm is unavailable."""
    if tqdm is None:
        print("[INFO] tqdm not found. Install with: pip install tqdm")
        return None

    return tqdm(
        total=total,
        initial=start_idx,
        desc="AMR Parsing",
        unit="sent",
        dynamic_ncols=True,
    )


def run_batch_inference(
    input_file: str,
    output_file: str,
    process_file: Optional[str] = None,
    model_name: str = "AMR3-structbart-L",
    no_resume: bool = False,
    batch_size: int = DEFAULT_BATCH_SIZE,
    strip_alignments: bool = True,
) -> None:
    """Run AMR parsing in batches with resume support."""
    sentences = load_sentences(input_file)
    total = len(sentences)
    if total == 0:
        print("[INFO] no sentences")
        return

    start_idx = 0
    if not no_resume and process_file is not None:
        progress = load_progress(process_file)
        if progress.get("completed", False):
            print("[INFO] already completed")
            return

        start_idx = min(
            int(progress.get("processed_count", 0)),
            count_entries_in_output(output_file),
        )

    from transition_amr_parser.parse import AMRParser

    parser = AMRParser.from_pretrained(model_name)

    interrupted = False

    def _sig(*_):
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, _sig)

    first_mode = "a" if start_idx > 0 else "w"
    pbar = _build_progress_bar(total=total, start_idx=start_idx)
    i = start_idx
    while i < total and not interrupted:
        end = min(i + batch_size, total)
        batch = sentences[i:end]

        try:
            tokenized = [parser.tokenize(sent)[0] for sent in batch]
            annotations, _ = parser.parse_sentences(tokenized)
        except Exception:
            annotations = []
            for sent in batch:
                try:
                    toks, _ = parser.tokenize(sent)
                    ann, _ = parser.parse_sentence(toks)
                    annotations.append(ann)
                except Exception:
                    annotations.append(f"(PARSE-ERROR :snt {json.dumps(sent)})")

        # Re-open output each batch so Google Drive sees updates sooner.
        write_mode = first_mode if i == start_idx else "a"
        with open(output_file, write_mode, encoding="utf-8") as out_file:
            for j, (sent, ann) in enumerate(zip(batch, annotations)):
                sentence_idx = i + j
                out_file.write(
                    format_amr_entry(
                        sentence_idx,
                        sent,
                        str(ann),
                        model_name,
                        strip_alignments,
                    )
                )
                if pbar is not None:
                    pbar.update(1)
                    pbar.set_postfix_str(f"last_id={sentence_idx}")
                else:
                    print(f"processed {sentence_idx + 1}/{total}")

            out_file.flush()
            try:
                os.fsync(out_file.fileno())
            except OSError:
                pass

        i = end
        if process_file is not None:
            save_progress(
                process_file,
                {
                    "processed_count": i,
                    "total_count": total,
                    "input_file": input_file,
                    "output_file": output_file,
                    "model": model_name,
                    "completed": i >= total,
                },
            )

    if pbar is not None:
        pbar.close()

    if process_file is not None:
        save_progress(
            process_file,
            {
                "processed_count": i,
                "total_count": total,
                "input_file": input_file,
                "output_file": output_file,
                "model": model_name,
                "completed": (not interrupted and i >= total),
            },
        )

    if interrupted:
        print(f"[INFO] interrupted at sentence {i}/{total}")
    else:
        print(f"[INFO] done {i}/{total}")