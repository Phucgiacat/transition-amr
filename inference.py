import json

import os

import signal

import sys

import tempfile

from datetime import datetime



DEFAULT_BATCH_SIZE = 8



def load_sentences(input_file):

    sentences = []

    with open(input_file, 'r', encoding='utf-8') as f:

        for line_no, line in enumerate(f, 1):

            line = line.strip()

            if not line:

                continue

            try:

                data = json.loads(line)

                if 'sent' in data:

                    sentences.append(data['sent'])

                elif 'sentence' in data:

                    sentences.append(data['sentence'])

                else:

                    sentences.append(next(iter(data.values())))

            except json.JSONDecodeError as exc:

                print(f'[WARNING] malformed line {line_no}: {exc}', file=sys.stderr)

    return sentences



def load_progress(process_file):

    if process_file is None or not os.path.exists(process_file):

        return {'processed_count': 0, 'completed': False}

    try:

        with open(process_file, 'r', encoding='utf-8') as f:

            return json.load(f)

    except Exception:

        return {'processed_count': 0, 'completed': False}



def save_progress(process_file, data):

    if process_file is None:

        return

    data = dict(data)

    data['last_updated'] = datetime.now().isoformat()

    dir_name = os.path.dirname(os.path.abspath(process_file)) or '.'

    with tempfile.NamedTemporaryFile('w', dir=dir_name, suffix='.tmp', delete=False, encoding='utf-8') as tf:

        json.dump(data, tf, indent=2, ensure_ascii=False)

        tmp_path = tf.name

    os.replace(tmp_path, process_file)



def count_entries_in_output(output_file):

    if not os.path.exists(output_file):

        return 0

    c = 0

    with open(output_file, 'r', encoding='utf-8') as f:

        for line in f:

            if line.startswith('# ::id '):

                c += 1

    return c



def format_amr_entry(idx, sentence, annotation_str, model_name):

    graph_lines = [line for line in (annotation_str or '').split('\n') if not line.startswith('# ')]

    graph_str = '\n'.join(graph_lines).strip()

    return f'# ::id {idx}\n# ::annotator {model_name}\n# ::snt {sentence}\n{graph_str}\n\n'



def run_batch_inference(input_file, output_file, process_file=None, model_name='AMR3-structbart-L', no_resume=False, batch_size=DEFAULT_BATCH_SIZE):

    sentences = load_sentences(input_file)

    total = len(sentences)

    if total == 0:

        print('[INFO] no sentences')

        return



    start_idx = 0

    if not no_resume and process_file is not None:

        p = load_progress(process_file)

        if p.get('completed', False):

            print('[INFO] already completed')

            return

        start_idx = min(p.get('processed_count', 0), count_entries_in_output(output_file))



    from transition_amr_parser.parse import AMRParser

    parser = AMRParser.from_pretrained(model_name)



    interrupted = False

    def _sig(*_):

        nonlocal interrupted

        interrupted = True

    signal.signal(signal.SIGINT, _sig)



    mode = 'a' if start_idx > 0 else 'w'

    with open(output_file, mode, encoding='utf-8') as out_f:

        i = start_idx

        while i < total and not interrupted:

            end = min(i + batch_size, total)

            batch = sentences[i:end]

            try:

                tokenized = [parser.tokenize(s)[0] for s in batch]

                annotations, _ = parser.parse_sentences(tokenized)

            except Exception:

                annotations = []

                for s in batch:

                    try:

                        toks, _ = parser.tokenize(s)

                        ann, _ = parser.parse_sentence(toks)

                        annotations.append(ann)

                    except Exception:

                        annotations.append(f'(PARSE-ERROR :snt {json.dumps(s)})')



            for j, (s, ann) in enumerate(zip(batch, annotations)):

                out_f.write(format_amr_entry(i + j, s, str(ann), model_name))



            out_f.flush()

            try:

                os.fsync(out_f.fileno())

            except OSError:

                pass



            i = end

            if process_file is not None:

                save_progress(process_file, {

                    'processed_count': i,

                    'total_count': total,

                    'input_file': input_file,

                    'output_file': output_file,

                    'model': model_name,

                    'completed': i >= total,

                })

            print(f'processed {i}/{total}')



    if process_file is not None:

        save_progress(process_file, {

            'processed_count': total,

            'total_count': total,

            'input_file': input_file,

            'output_file': output_file,

            'model': model_name,

            'completed': True,

        })