import argparse

from inference import run_batch_inference


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run transition-amr batch inference with resume support.",
    )
    parser.add_argument("--input", required=True, help="Input JSONL file")
    parser.add_argument("--output", required=True, help="Output AMR txt file")
    parser.add_argument(
        "--process",
        default=None,
        help="Progress JSON file (omit to disable resume tracking)",
    )
    parser.add_argument(
        "--model",
        default="AMR3-structbart-L",
        help="Model name (default: AMR3-structbart-L)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore progress file and start from beginning",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size (default: 8)",
    )
    parser.add_argument(
        "--keep-alignments",
        action="store_true",
        help="Keep alignment marks like ~3 in AMR graph (default: remove)",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    run_batch_inference(
        input_file=args.input,
        output_file=args.output,
        process_file=args.process,
        model_name=args.model,
        no_resume=args.no_resume,
        batch_size=args.batch_size,
        strip_alignments=not args.keep_alignments,
    )


if __name__ == "__main__":
    main()
