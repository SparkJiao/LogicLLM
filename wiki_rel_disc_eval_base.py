import argparse
import json

import torch
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str)
    parser.add_argument("--prediction_file", type=str)
    args = parser.parse_args()

    examples, _ = torch.load(args.data_file)

    if args.prediction_file.endswith(".json"):
        predictions = json.load(open(args.prediction_file, "r"))
    elif args.prediction_file.endswith(".jsonl"):
        predictions = []
        with open(args.prediction_file, "r") as f:
            for line in f:
                predictions.append(json.loads(line))
    else:
        raise ValueError("Unknown file format")

    pp = 0
    pp_corr = 0
    pn = 0
    pn_corr = 0
    np = 0
    np_corr = 0
    nn = 0
    nn_corr = 0
    for pred in tqdm(predictions, total=len(predictions)):
        index_a, index_b = pred["id"].split("_")
        index_a = int(index_a)
        index_b = int(index_b)
        type_a = "h" in examples[index_a]
        type_b = "h" in examples[index_b]
        if pred["pred"] == "Yes":
            if not type_a and not type_b:
                pp += 1
                pp_corr += 1
            elif type_a and not type_b:
                np += 1
                np_corr += 1
            elif not type_a and type_b:
                pn += 1
                pn_corr += 1
            elif type_a and type_b:
                nn += 1
                nn_corr += 1
            else:
                print("*****")
        elif pred["pred"] == "No":
            if not type_a and not type_b:
                pp += 1
            elif type_a and not type_b:
                np += 1
            elif not type_a and type_b:
                pn += 1
            elif type_a and type_b:
                nn += 1
            else:
                print("*****")
        else:
            print(pred["pred"])

    if pp > 0:
        print("pp: {} / {} = {}".format(pp_corr, pp, pp_corr / pp))
    else:
        print("PP = 0")
    if pn > 0:
        print("pn: {} / {} = {}".format(pn_corr, pn, pn_corr / pn))
    else:
        print("PN = 0")
    if np > 0:
        print("np: {} / {} = {}".format(np_corr, np, np_corr / np))
    else:
        print("NP = 0")
    if nn > 0:
        print("nn: {} / {} = {}".format(nn_corr, nn, nn_corr / nn))
    else:
        print("NN = 0")


if __name__ == '__main__':
    main()
