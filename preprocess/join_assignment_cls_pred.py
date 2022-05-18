import json
import torch
import argparse
import os
from typing import List, Tuple, Dict
from glob import glob


def dfs_predict_assignment(assignment_list, index2pred: Dict[str, Dict[str, float]]):
    for assignment in assignment_list:
        if assignment['flag'] is False:
            continue
        if assignment['flag'] is None:
            assignment['flag'] = True if index2pred[assignment['id']]['pred'] == 1 else False
            assignment['prob'] = index2pred[assignment['id']]['prob']
            assignment['logit'] = index2pred[assignment['id']]['logit']
        elif assignment['flag'] is True:
            dfs_predict_assignment(assignment['assignment'], index2pred)
        else:
            raise ValueError('Unknown flag: {}'.format(assignment['flag']))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--predictions', type=str, required=True)
    parser.add_argument('--output_file', type=str, default=None)
    parser.add_argument('--prob_threshold', type=float, default=0.5)

    args = parser.parse_args()

    if os.path.exists(args.predictions):
        predictions = torch.load(args.predictions)
        logits, preds, index = predictions['logits'], predictions['preds'], predictions['index']
    else:
        logits, preds, index = [], [], []
        for file in glob(args.predictions):
            predictions = torch.load(file)
            logits.append(predictions['logits'])
            preds.append(predictions['preds'])
            if isinstance(predictions['index'], list):
                index.extend(predictions['index'])
            else:
                index.append(predictions['index'])
        logits = torch.cat(logits, dim=0)
        preds = torch.cat(preds, dim=0)
        if isinstance(index[0], torch.Tensor):
            index = torch.cat(index, dim=0)

    index2pred = {
        idx: {
            "logit": logit.tolist(),
            "prob": torch.softmax(logit.float(), dim=-1)[pred.item()].item(),
            "pred": pred.item()
        } for idx, pred, logit in zip(index, preds, logits)
    }

    data = json.load(open(args.input_file))

    for item in data:
        if 'assignment' in item:
            dfs_predict_assignment(item['assignment'], index2pred)

    if args.output_file is None:
        output_file = os.path.dirname(args.predictions) + '/assignment_pred.json'
        json.dump(data, open(output_file, 'w'), indent=2)
    else:
        json.dump(data, open(args.output_file, 'w'), indent=2)


if __name__ == '__main__':
    main()
