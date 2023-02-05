import argparse
import collections
import os.path
import pickle
from functools import partial
from glob import glob
from multiprocessing import Pool

import torch
from tqdm import tqdm


def unify_predictions(examples, losses, indices):
    item_id2index = {}
    for exp_id, exp in enumerate(examples):
        item_id2index[exp["id"]] = exp_id

    assert len(losses) == len(indices)

    for loss, index in tqdm(zip(losses, indices), total=len(losses)):
        item_id, s_id, neg_id = index.split("-")
        examples[item_id2index[item_id]]["templates"][int(s_id)]["neg"][int(neg_id)]["loss"] = loss
    return examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--feature_file", type=str)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--num_workers", type=int, default=48)
    args = parser.parse_args()

    if os.path.exists(args.input_file):
        files = [args.input_file]
    else:
        files = glob(args.input_file)

    losses = []
    indices = []
    for file in files:
        print(f"Loading predictions from {file}...")
        predictions = torch.load(file)

        losses.extend(predictions["losses"])
        indices.extend(predictions["indices"])

    examples = torch.load(args.feature_file)
    examples = unify_predictions(examples, losses, indices)

    torch.save(examples, args.output_file)


if __name__ == '__main__':
    main()
