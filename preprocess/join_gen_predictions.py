import json
import torch
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--predictions', type=str, required=True)
    parser.add_argument('--pred_on_passage_only', default=False, action='store_true')
    parser.add_argument('--output', type=str, default=None)

    args = parser.parse_args()

    index_list, prediction_list = torch.load(args.predictions)
    assert len(index_list) == len(prediction_list)
    index2pred = {idx[0]: pred for idx, pred in zip(index_list, prediction_list)}

    data = json.load(open(args.input_file))
    idx = 0
    for item in data:
        if args.pred_on_passage_only:
            item['prediction'] = index2pred[idx]
            idx += 1
        else:
            for q in item['questions']:
                q['prediction'] = index2pred[idx]
                idx += 1
    if args.output is None:
        # Get the parent directory of the prediction file
        output_file = os.path.dirname(args.predictions) + '/combine.json'
        json.dump(data, open(output_file, 'w'), indent=2)
    else:
        json.dump(data, open(args.output, 'w'), indent=2)
