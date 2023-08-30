import argparse
import json

import torch
from tqdm import tqdm

from data.collators.api.wiki_seq2seq import WikiDatasetUnifyInterface, WikiRelationConsistent


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--data_file", type=str)
#     parser.add_argument("--prediction_file", type=str)
#     parser.add_argument("--seed", type=int, default=42)
#     parser.add_argument("--n_gpu", type=int, default=1)
#     args = parser.parse_args()
#
#     examples, _ = torch.load(args.data_file)
#
#     predictions = json.load(open(args.prediction_file, "r"))
#
#     from general_util.training_utils import set_seed
#
#     set_seed(args)
#
#     dataset = WikiDatasetUnifyInterface(args.data_file, sample_num=len(predictions), balance=False)
#
#     normal_data = []
#     normal_true = 0
#     counterfactual_data = []
#     counterfactual_true = 0
#     for pred in tqdm(predictions, total=len(predictions)):
#         if pred["pred"] == "Yes":
#             if "h" in dataset[pred["id"][0]]["example"]:  # counterfactual data
#                 counterfactual_data.append(pred)
#                 counterfactual_true += 1
#             else:
#                 normal_data.append(pred)
#                 normal_true += 1
#         elif pred["pred"] == "No":
#             if "h" in dataset[pred["id"][0]]["example"]:
#                 counterfactual_data.append(pred)
#             else:
#                 normal_data.append(pred)
#
#     print("Normal data: {} / {} = {}".format(normal_true, len(normal_data), normal_true / len(normal_data)))
#     print("Counterfactual data: {} / {} = {}".format(counterfactual_true, len(counterfactual_data), counterfactual_true / len(counterfactual_data)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str)
    parser.add_argument("--prediction_file", type=str)
    args = parser.parse_args()

    examples, _ = torch.load(args.data_file)

    predictions = json.load(open(args.prediction_file, "r"))

    normal_data = []
    normal_true = 0
    counterfactual_data = []
    counterfactual_true = 0
    for pred in tqdm(predictions, total=len(predictions)):
        if pred["pred"] == "Yes":
            if "h" in examples[pred["id"][0]]:  # counterfactual data
                counterfactual_data.append(pred)
                counterfactual_true += 1
            else:
                normal_data.append(pred)
                normal_true += 1
        elif pred["pred"] == "No":
            if "h" in examples[pred["id"][0]]:
                counterfactual_data.append(pred)
            else:
                normal_data.append(pred)

    print("Normal data: {} / {} = {}".format(normal_true, len(normal_data), normal_true / len(normal_data)))
    print("Counterfactual data: {} / {} = {}".format(counterfactual_true, len(counterfactual_data), counterfactual_true / len(counterfactual_data)))


if __name__ == '__main__':
    main()
