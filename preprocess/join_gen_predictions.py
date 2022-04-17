import json
import torch
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--predictions', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)

    args = parser.parse_args()

    index_list, prediction_list = torch.load(args.predictions)
    assert len(index_list) == len(prediction_list)
    index2pred = {idx[0]: pred for idx, pred in zip(index_list, prediction_list)}

    data = json.load(open(args.input_file))
    idx = 0
    for item in data:
        for q in item['questions']:
            q['prediction'] = index2pred[idx]
            idx += 1

    json.dump(data, open(args.output, 'w'), indent=2)
