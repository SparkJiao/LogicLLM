import argparse
import collections
import os.path
import pickle
from functools import partial
from glob import glob
from multiprocessing import Pool

import torch
from tqdm import tqdm


def process_single_item(examples, kept_num: int = 3):
    example_id, examples = examples
    examples = sorted(examples, key=lambda x: x[3])  # sorted by `op_id`
    pos = examples[0]
    assert pos[3] == '0', pos[3]
    neg = examples[1:]
    neg = sorted(neg, key=lambda x: x[0] - pos[0])  # sorted by loss delta.
    neg_ids = [int(x[3]) for x in neg[:kept_num]]
    return example_id, neg_ids


def process_items(exp_id2loss, kept_num: int = 3, num_workers: int = 48):
    with Pool(num_workers) as p:
        _annotate = partial(process_single_item, kept_num=kept_num)
        _results = list(tqdm(
            p.imap(_annotate, list(exp_id2loss.items()), chunksize=32),
            total=len(exp_id2loss),
            desc="Processing items",
        ))
    results = {}
    for res in _results:
        results[res[0]] = res[1]
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--feature_file", type=str)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--feat_output_file", type=str)
    parser.add_argument("--kept_num", type=int, default=3)
    parser.add_argument("--num_workers", type=int, default=48)
    args = parser.parse_args()

    if os.path.exists(args.input_file):
        files = [args.input_file]
    else:
        files = glob(args.input_file)

    losses = []
    indices = []
    for file in files:
        predictions = torch.load(file)

        losses.extend(predictions["losses"])
        indices.extend(predictions["indices"])

    exp_id2loss = collections.defaultdict(list)
    for loss, index in tqdm(zip(losses, indices), total=len(losses)):
        example_id, orig_id, label, op_id, aug_flag = index.split("-")
        exp_id2loss[example_id].append((loss, orig_id, label, op_id, aug_flag))

    results = process_items(exp_id2loss, args.kept_num, args.num_workers)
    if args.output_file:
        torch.save(results, args.output_file)

    all_examples, raw_texts = torch.load(args.feature_file)
    for exp_id, neg_ids in tqdm(results.items(), total=len(results), desc="parsing results."):
        options = [all_examples[int(exp_id)]["tokens"][0]]
        for neg_id in neg_ids:
            assert neg_id != 0
            options.append(all_examples[int(exp_id)]["tokens"][neg_id])
        all_examples[int(exp_id)]["tokens"] = options

    if args.feat_output_file:
        torch.save((all_examples, raw_texts), args.feat_output_file)


if __name__ == '__main__':
    main()
