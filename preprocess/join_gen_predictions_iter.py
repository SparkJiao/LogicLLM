import json
import torch
import argparse
import os
from nltk import sent_tokenize
from nltk.translate.bleu_score import sentence_bleu

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--predictions', type=str, required=True)
    parser.add_argument('--iter_id', type=int, default=0)
    parser.add_argument('--predict_on_question', default=False, action='store_true')
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--filter_bleu', type=float, default=0.8)

    args = parser.parse_args()

    index_list, prediction_list = torch.load(args.predictions)
    assert len(index_list) == len(prediction_list)
    index2pred = {idx[0]: pred for idx, pred in zip(index_list, prediction_list)}

    data = json.load(open(args.input_file))
    idx = 0

    if not args.predict_on_question:
        repeat_predictions = 0
        for item in data:
            all_sentences = sent_tokenize(item['passage'])
            for i in range(args.iter_id):
                all_sentences.append(item[f'prediction_{i}'])

            item[f'prediction_{args.iter_id}'] = index2pred[idx]
            for sent in all_sentences:
                if sentence_bleu([sent.split()], index2pred[idx].split()) >= args.filter_bleu:
                    repeat_predictions += 1
                    item[f'prediction_{args.iter_id}'] = ''
                    break

            idx += 1
    else:
        repeat_predictions = 0
        for item in data:
            all_sentences = sent_tokenize(item['passage'])
            p_iter_id = 0
            while f'prediction_{p_iter_id}' in item:
                all_sentences.append(item[f'prediction_{p_iter_id}'])
                p_iter_id += 1

            for q in item['questions']:
                all_sentences.append(q['question'])

                q[f'prediction_{args.iter_id}'] = index2pred[idx]
                for sent in all_sentences:
                    if sentence_bleu([sent.split()], index2pred[idx].split()) >= args.filter_bleu:
                        repeat_predictions += 1
                        q[f'prediction_{args.iter_id}'] = ''
                        break
                idx += 1

    print(f'{repeat_predictions} / {idx} predictions repeated')

    if args.output is None:
        output_file = os.path.dirname(args.predictions) + f'/combine_{args.iter_id}.json'
        json.dump(data, open(output_file, 'w'), indent=2)
        print(f'Saved to {output_file}')
    else:
        json.dump(data, open(args.output, 'w'), indent=2)
        print(f'Saved to {args.output}')
