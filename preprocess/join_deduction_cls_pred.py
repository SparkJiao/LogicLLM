import json
import torch
import argparse
import os
import logging

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, default='AR-LSAT/data/AR_TrainingData.json')
    parser.add_argument('--predictions', type=str, required=True)
    parser.add_argument('--output_file', type=str, default=None)
    parser.add_argument('--prob_threshold', type=float, default=0.5)

    args = parser.parse_args()

    predictions = torch.load(args.predictions)
    logits, preds, index = predictions["logits"], predictions["preds"], predictions["index"]

    index2pred = {
        int(idx[0]): {
            "prob": torch.softmax(logit.float(), dim=-1)[pred.item()],
            "pred": pred.item()
        } for idx, pred, logit in zip(index, preds, logits)
    }

    data = json.load(open(args.input_file))
    idx = 0
    kept_num = 0
    for item in data:
        for q in item["questions"]:
            if index2pred[idx]["pred"] == 1 and index2pred[idx]["prob"] >= args.prob_threshold:
                kept_num += 1
            else:
                q.pop("prediction")
            idx += 1

    print(f"{kept_num}/{idx} = {kept_num * 1.0 / idx} deductions kept.")

    if args.output_file is None:
        output_file = os.path.dirname(args.predictions) + "/filter_predictions.json"
        json.dump(data, open(output_file, "w"), indent=2)
    else:
        json.dump(data, open(args.output_file, "w"), indent=2)
