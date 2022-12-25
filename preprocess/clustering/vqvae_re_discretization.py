import argparse
import os
from collections import Counter

import torch


def parse_prediction(file_path):
    predictions = torch.load(file_path, map_location="cpu")
    codes = predictions["codes"]
    indices = predictions["indices"]

    index2code = {}
    max_code_id = -1
    for code, index in zip(codes, indices):
        index2code[index] = code.item() if isinstance(code, torch.Tensor) else code
        max_code_id = max(max_code_id, index2code[index])

    code_cnt = Counter(list(index2code.values()))
    return index2code, max_code_id, code_cnt


def reformulate_prediction(index2code, orig_code2new_code, output_file):
    indices = []
    codes = []
    for index, code in index2code.items():
        indices.append(index)
        codes.append(orig_code2new_code[code])
    torch.save({
        "codes": codes,
        "indices": indices,
    }, output_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_predictions", type=str)
    parser.add_argument("--dev_predictions", type=str)
    parser.add_argument("--output_dir", type=str)
    args = parser.parse_args()

    train_index2code, train_max_code_id, train_code_cnt = parse_prediction(args.train_predictions)
    dev_index2code, dev_max_code_id, dev_code_cnt = parse_prediction(args.dev_predictions)

    max_code_id = max(train_max_code_id, dev_max_code_id)
    code_cnt = Counter()
    code_cnt.update(train_code_cnt)
    code_cnt.update(dev_code_cnt)

    orig_code2new_code = {}
    cnt = 0
    for code_i in range(max_code_id + 1):
        if code_i in code_cnt and code_cnt[code_i] > 0:
            assert code_i not in orig_code2new_code
            orig_code2new_code[code_i] = cnt
            cnt += 1

    print(cnt)
    print(len(orig_code2new_code))

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    reformulate_prediction(train_index2code, orig_code2new_code, os.path.join(args.output_dir, "train_predictions.pt"))
    reformulate_prediction(dev_index2code, orig_code2new_code, os.path.join(args.output_dir, "dev_predictions.pt"))
    torch.save(orig_code2new_code, os.path.join(args.output_dir, "orig_code2new_code.pt"))


if __name__ == '__main__':
    main()
