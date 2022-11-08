import pickle
import argparse
import torch


def extract_path_w_rel_and_parse(example):
    path = example["path"]
    rels = []
    for i, item in enumerate(path):
        if i == 0:
            continue
        edge = f"{path[i - 1][0]}\t{item[0]}"
        rels.append(edge)
    rels.append(f"{path[0][0]}\t{path[-1][0]}")

    return rels, "#".join(list(rels))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--output_file", type=str)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    data = pickle.load(open(args.input_file, "rb"))
    examples = data["examples"]

    rels = set()
    paths = set()
    avg_len = 0
    for example in examples:
        exp_rels, path = extract_path_w_rel_and_parse(example)
        avg_len += len(exp_rels)
        rels.update(exp_rels)
        paths.add(path)

    print(avg_len / len(examples))
    print(len(rels))
    print(len(paths))
    torch.save(list(rels), args.output_file)


if __name__ == '__main__':
    main()
