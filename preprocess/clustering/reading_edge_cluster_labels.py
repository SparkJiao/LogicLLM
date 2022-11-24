import argparse
import glob
import os.path
import pickle
from collections import defaultdict
from typing import Dict

import torch
from tqdm import tqdm


def extract_path_and_parse(example):
    path = example["path"]
    example_id = example["id"]
    ent_path = "\t".join([item[0] for item in path])
    return ent_path, example_id


def extract_path_w_rel_and_parse(example, edge_labels: Dict[str, int], limit: int = 1):
    path = example["path"]
    example_id = example["id"]
    rels = []
    ent_path = []
    empty_rel_cnt = 0
    for i, item in enumerate(path):
        ent_path.append(item[0])
        if i == 0:
            continue
        edge = f"{path[i - 1][0]}\t{item[0]}"
        if edge in edge_labels:
            rels.append(edge_labels[edge])
        else:
            empty_rel_cnt += 1

    if empty_rel_cnt > limit:
        return "\t".join(ent_path), None, example_id
    else:
        return "\t".join(ent_path), rels, example_id

    # return "\t".join(ent_path), "\t".join(rels) if None not in rels else None, example_id


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--edge_relation_file", type=str)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--path_output_file", type=str, default=None)
    parser.add_argument("--rel_vocab", type=str, default=None)
    parser.add_argument("--reid_examples", type=str, default=None)
    parser.add_argument("--edge_weights_save", type=str, default=None)
    parser.add_argument("--limit", type=int, default=0)

    args = parser.parse_args()
    return args


def load_edge_relation_file(edge_relation_file):
    edge_hidden, edge_relation_labels, _ = torch.load(edge_relation_file, map_location="cpu")
    edge_labels = {}
    for edge, edge_label in zip(edge_hidden.keys(), edge_relation_labels):
        edge_labels[edge] = edge_label.item()

    return edge_labels


def combine_multiple_edge_labels(*edge_labels_groups):
    edge_label_set = set()
    edge_label_concat = {}
    edge_labels = {}
    edges = list(edge_labels_groups[0].keys())
    for edge in edges:
        edge_label = []
        for edge_label_group_i in edge_labels_groups:
            edge_label.append(str(edge_label_group_i[edge]))
        edge_label = "$".join(edge_label)
        edge_label_set.add(edge_label)
        edge_label_concat[edge] = edge_label

    edge_label_vocab = {edge_label: i for i, edge_label in enumerate(edge_label_set)}
    for edge in edge_label_concat:
        edge_labels[edge] = edge_label_vocab[edge_label_concat[edge]]

    return edge_labels, edge_label_vocab


def rearrange_example_id(examples):
    id_set = {}
    for exp in examples:
        orig_id = str(exp["id"])
        if orig_id not in id_set:
            id_set[orig_id] = 0

        exp["id"] = orig_id + "_" + str(id_set[orig_id])
        id_set[orig_id] = id_set[orig_id] + 1
    return examples


def main():
    args = parse_args()

    data = pickle.load(open(args.input_file, "rb"))
    examples = data["examples"]
    print(len(set([exp["id"] for exp in examples])))
    print(len(examples))

    if os.path.exists(args.edge_relation_file):
        edge_labels = load_edge_relation_file(args.edge_relation_file)
    else:
        files = glob.glob(args.edge_relation_file)
        print(files)
        edge_labels_groups = [load_edge_relation_file(_file) for _file in files]
        edge_labels, edge_label_vocab = combine_multiple_edge_labels(*edge_labels_groups)
        # edge_label_vocab = list(edge_label_vocab.keys())
        # print(f"Combined edge labels: {len(edge_label_vocab)}")

    _sorted_edge_labels = sorted(edge_labels.items(), key=lambda x: x[1])
    assert _sorted_edge_labels[0][1] == 0, _sorted_edge_labels[0][1]
    edge_label_vocab = [x[0] for x in _sorted_edge_labels]

    edge_hidden, _, _ = torch.load(args.edge_relation_file, map_location="cpu")
    edge_pretrained_weights = torch.stack([
        edge_hidden[edge] for edge in tqdm(edge_label_vocab, total=len(edge_label_vocab))], dim=0)

    rel_path_set = defaultdict(list)
    results = []
    for example in tqdm(examples):
        ent_path, rel_path, example_id = extract_path_w_rel_and_parse(example, edge_labels, limit=args.limit)

        if rel_path is not None:
            rel_path_set["\t".join(map(str, rel_path))].append(example_id)

        results.append((ent_path, rel_path, example_id))

    print(len(rel_path_set))

    # id2pos_ls = defaultdict(list)
    id2rel_path_decode_ids = {}
    # ent_matching_cnt = 0
    no_direct_rel_cnt = 0
    # cnt = 0
    for res in tqdm(results):
        ent_path, rel_path, example_id = res
        # tmp_set = set()
        # tmp_set.update(set([x for x in ent_path_set[ent_path] if x != example_id]))
        # ent_matching_cnt += len(tmp_set)

        if rel_path is not None:
            # tmp_set.update(set([x for x in rel_path_set[rel_path] if x != example_id]))

            # rel_path_decode_id_a = [rel_vocab[_rel] for _rel in rel_path.split("\t")]

            ent_path = ent_path.split("\t")
            direct_edge = f"{ent_path[0]}\t{ent_path[-1]}"
            if direct_edge in edge_labels:
                # rel_path_decode_id_b = edge_labels[direct_edge]

                id2rel_path_decode_ids[example_id] = {
                    "input_a": rel_path,
                    # "input_b": rel_path_decode_id_b,
                    "input_b": edge_labels[direct_edge],
                }
            else:
                # rel_path_decode_id_b = -1
                no_direct_rel_cnt += 1

            # if rel_path_decode_id_b == -1:
            #     no_direct_rel_cnt += 1

        # id2pos_ls[example_id] = list(tmp_set)
        # cnt += len(id2pos_ls[example_id])

    # print(f"{cnt} / {len(results)} = {cnt / len(results)}")
    # print(f"{cnt - ent_matching_cnt} / {len(results)} = {(cnt - ent_matching_cnt) / len(results)}")
    print(f"{no_direct_rel_cnt} / {len(id2rel_path_decode_ids)} = {no_direct_rel_cnt / len(id2rel_path_decode_ids)}")

    # pickle.dump(id2pos_ls, open(args.output_file, "wb"))  # use pickle since the key are of type `int`.
    pickle.dump(rel_path_set, open(args.output_file, "wb"))  # V2.0
    if args.path_output_file is not None:
        pickle.dump(id2rel_path_decode_ids, open(args.path_output_file, "wb"))
    if args.rel_vocab is not None:
        pickle.dump(edge_label_vocab, open(args.rel_vocab, "wb"))  # id2rel_id
    if args.edge_weights_save is not None:
        torch.save(edge_pretrained_weights, args.edge_weights_save)
    # if args.reid_examples is not None:
    #     data["examples"] = examples
    #     pickle.dump(data, open(args.reid_examples, "wb"))


if __name__ == '__main__':
    main()

    # 472342
    # 100%|██████████| 472342/472342 [00:01<00:00, 332011.69it/s]
    # 786
    # 100%|██████████| 472342/472342 [00:00<00:00, 652365.50it/s]
    # 712078 / 472342 = 1.507547497364198
    # 657266 / 472342 = 1.3915044607508966
    # 5714 / 3102 = 1.8420373952288847
