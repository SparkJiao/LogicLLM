import json
import torch
import argparse

import os
import sys
from nltk import sent_tokenize

pwd = os.getcwd()
f_pwd = os.path.abspath(os.path.dirname(pwd) + os.path.sep + "..")

sys.path.append(f_pwd)

from data.examples import LSATDeductionExamples, LSATDeductionExamplesV2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=str, required=True)
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--prob_threshold", type=float, default=0.5)
    args = parser.parse_args()

    predictions = torch.load(args.predictions)
    logits, preds, index = predictions["logits"], predictions["preds"], predictions["index"]

    data = LSATDeductionExamples().data
    # data = LSATDeductionExamplesV2().data

    index2pred = {
        idx[0].item(): {
            "logit": logit.tolist(),
            "prob": torch.softmax(logit.float(), dim=-1)[pred.item()].item(),
            "pred": pred.item(),
        } for idx, pred, logit in zip(index, preds, logits)
    }

    cnt = 0
    for item in data:
        passage = item["passage"]
        sentences = sent_tokenize(passage)
        if item["positive_deductions"] and item["negative_deductions"]:
            item["positive_predictions"] = []
            for _ in item["positive_deductions"]:
                deduction_pred = []
                for sent in sentences:
                    index2pred[cnt]["sent"] = sent
                    deduction_pred.append(index2pred[cnt])
                    cnt += 1
                item["positive_predictions"].append(deduction_pred)

            item["negative_predictions"] = []
            for _ in item["negative_deductions"]:
                deduction_pred = []
                for sent in sentences:
                    index2pred[cnt]["sent"] = sent
                    deduction_pred.append(index2pred[cnt])
                    cnt += 1
                item["negative_predictions"].append(deduction_pred)

        for q in item["questions"]:
            question = q["question"]
            if q["positive_deductions"] and q["negative_deductions"]:
                q["positive_predictions"] = []
                for _ in q["positive_deductions"]:
                    deduction_pred = []
                    for sent in sentences:
                        index2pred[cnt]["sent"] = sent + ' ' + question
                        deduction_pred.append(index2pred[cnt])
                        cnt += 1
                    q["positive_predictions"].append(deduction_pred)

                q["negative_predictions"] = []
                for _ in q["negative_deductions"]:
                    deduction_pred = []
                    for sent in sentences:
                        index2pred[cnt]["sent"] = sent + ' ' + question
                        deduction_pred.append(index2pred[cnt])
                        cnt += 1
                    q["negative_predictions"].append(deduction_pred)

    assert cnt == len(index2pred), (cnt, len(index2pred))

    if args.output_file is None:
        output_file = os.path.dirname(args.predictions) + '/deductions_pred.json'
        json.dump(data, open(output_file, 'w'), indent=2)
    else:
        json.dump(data, open(args.output_file, 'w'), indent=2)


if __name__ == '__main__':
    main()
