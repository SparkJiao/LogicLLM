import json
import torch
import argparse
import os
import glob


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--predictions', type=str, required=True)
    parser.add_argument('--output_file', type=str, default=None)

    args = parser.parse_args()

    if os.path.exists(args.predictions):
        index_list, prediction_list = torch.load(args.predictions)
    else:
        index_list = []
        prediction_list = []
        for file in glob.glob(args.predictions):
            index, prediction = torch.load(file)
            index_list.extend(index)
            prediction_list.extend(prediction)

    assert len(index_list) == len(prediction_list)
    index2pred = {idx[0]: pred for idx, pred in zip(index_list, prediction_list)}

    data = json.load(open(args.input_file))
    idx = 0
    for item in data:
        context_sent_num = len(item["context"])
        item["pred_context_rule"] = []
        for i in range(context_sent_num):
            item["pred_context_rule"].append(index2pred[idx])
            idx += 1

        for qa_id, qa in enumerate(item["qa"]):
            item["qa"][qa_id]["pred_q_rule"] = index2pred[idx]
            idx += 1

            item["qa"][qa_id]["pred_o_rule"] = []
            for _ in range(len(qa["option"])):
                item["qa"][qa_id]["pred_o_rule"].append(index2pred[idx])
                idx += 1

    assert idx == len(index2pred)
    if args.output_file is not None:
        json.dump(data, open(args.output_file, 'w'), indent=2)
    else:
        output_file = os.path.dirname(args.predictions) + '/combine.json'
        json.dump(data, open(output_file, 'w'), indent=2)


if __name__ == '__main__':
    main()
