import argparse
import collections
import os
import pickle
from glob import glob
from typing import List

import torch
from tqdm import tqdm

file_name_mapping = {
    # f"train_distant_{i}.path_v9.1.pkl": f"train_distant_{i}.json" for i in range(10)
    f"train_distant_{i}.path_v9.2_mm5.pkl": f"train_distant_{i}.json" for i in range(10)
}


def organize_entity_hidden(file_path):
    # print(f"Loading entity hidden.")
    # predictions = torch.load(file_path, map_location="cpu")
    # print(f"Entity hidden loaded.")
    # entity_hidden = predictions["hidden"]
    # indices = predictions["index"]
    # print(indices[:20])
    print(file_path)
    if os.path.exists(file_path):
        files = [file_path]
    else:
        files = glob(file_path)
    print(files)
    entity_hidden = []
    indices = []
    for file in files:
        print(f"Loading entity hidden from {file}")
        predictions = torch.load(file, map_location="cpu")
        entity_hidden.extend(predictions["hidden"])
        indices.extend(predictions["index"])
    print(f"Loaded.")

    file_ent_hidden = collections.defaultdict(dict)
    for item_ent_hidden, index in tqdm(zip(entity_hidden, indices), total=len(indices)):
        file_name, item_id = index
        file_ent_hidden[file_name][item_id] = item_ent_hidden
    return file_ent_hidden


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


def extract_edge_hidden(example, item_hidden):
    edges, _ = extract_path_w_rel_and_parse(example)
    edge_hidden = {}
    ignored_edges = 0
    for edge in edges:
        h, t = edge.split("\t")

        if h not in item_hidden:
            ignored_edges += 1
            continue
        if t not in item_hidden:
            ignored_edges += 1
            continue

        h_hidden = item_hidden[h]
        t_hidden = item_hidden[t]
        edge_hidden[edge] = torch.cat([h_hidden, t_hidden], dim=-1)

    # print(f"{ignored_edges} edges missing.")
    return edge_hidden


def read_edge_hidden(input_files: List[str], file_ent_hidden):
    exp_id2edge_hidden = {}
    for _file in input_files:
        _file_name = file_name_mapping[_file.split("/")[-1]]

        data = pickle.load(open(_file, "rb"))["examples"]
        for item in tqdm(data, desc=f"extracting edge hidden from {_file_name}", total=len(data)):
            _id = item["id"]
            item_id = int(_id.split("_")[0])
            item_hidden = file_ent_hidden[_file_name][item_id]
            item_edge_hidden = extract_edge_hidden(item, item_hidden)
            exp_id2edge_hidden[_id] = item_edge_hidden

    return exp_id2edge_hidden


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--example_file", type=str)
    parser.add_argument("--entity_hidden_file", type=str)
    parser.add_argument("--output_file", type=str)
    args = parser.parse_args()

    if not os.path.exists(args.example_file):
        input_files = glob(args.example_file)
    else:
        input_files = [args.example_file]

    file_ent_hidden = organize_entity_hidden(args.entity_hidden_file)

    exp_id2edge_hidden = read_edge_hidden(input_files, file_ent_hidden)

    torch.save(exp_id2edge_hidden, args.output_file)


if __name__ == '__main__':
    print(f"Start")
    main()
