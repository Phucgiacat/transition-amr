import argparse

from inference import run_batch_inference



p = argparse.ArgumentParser()

p.add_argument('--input', required=True)

p.add_argument('--output', required=True)

p.add_argument('--process', default=None)

p.add_argument('--model', default='AMR3-structbart-L')

p.add_argument('--no-resume', action='store_true')

p.add_argument('--batch-size', type=int, default=8)

args = p.parse_args()



run_batch_inference(

    input_file=args.input,

    output_file=args.output,

    process_file=args.process,

    model_name=args.model,

    no_resume=args.no_resume,

    batch_size=args.batch_size,

)
