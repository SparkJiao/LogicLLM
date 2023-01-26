import argparse
import pickle
from collections import defaultdict
from typing import Dict

import torch
from tqdm import tqdm


def extract_path_w_rel_and_parse(example, edge_labels: Dict[str, Dict[str, int]], limit: int = 1):
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
        if edge in edge_labels[example_id]:
            rels.append(edge_labels[example_id][edge])
        else:
            empty_rel_cnt += 1

    if empty_rel_cnt > limit:
        return "\t".join(ent_path), None, example_id
    else:
        return "\t".join(ent_path), rels, example_id


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--edge_cluster_file", type=str)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--path_output_file", type=str, default=None)
    parser.add_argument("--rel_vocab", type=str, default=None)
    parser.add_argument("--reid_examples", type=str, default=None)
    parser.add_argument("--limit", type=int, default=0)

    args = parser.parse_args()
    return args


def load_edge_relation_file(edge_cluster_file):
    (exp_ids, edges), edge_relation_labels, _ = torch.load(edge_cluster_file, map_location="cpu")
    edge_labels = defaultdict(dict)
    assert len(exp_ids) == len(edges) == len(edge_relation_labels), (len(exp_ids), len(edges), len(edge_relation_labels))
    for exp_id, edge, edge_label in zip(exp_ids, edges, edge_relation_labels):
        edge_labels[exp_id][edge] = edge_label.item()

    return edge_labels


def main():
    args = parse_args()

    data = pickle.load(open(args.input_file, "rb"))
    examples = data["examples"]
    print(len(set([exp["id"] for exp in examples])))
    print(len(examples))

    edge_labels = load_edge_relation_file(args.edge_cluster_file)

    rel_path_set = defaultdict(list)
    results = []
    for example in tqdm(examples):
        ent_path, rel_path, example_id = extract_path_w_rel_and_parse(example, edge_labels, limit=args.limit)

        if rel_path is not None:
            rel_path_set["\t".join(map(str, rel_path))].append(example_id)

        results.append((ent_path, rel_path, example_id))

    print(len(rel_path_set))

    id2rel_path_decode_ids = {}
    no_direct_rel_cnt = 0
    for res in tqdm(results):
        ent_path, rel_path, example_id = res

        if rel_path is not None:

            ent_path = ent_path.split("\t")
            direct_edge = f"{ent_path[0]}\t{ent_path[-1]}"
            if direct_edge in edge_labels[example_id]:

                id2rel_path_decode_ids[example_id] = {
                    "input_a": rel_path,
                    "input_b": edge_labels[example_id][direct_edge],
                }
            else:
                no_direct_rel_cnt += 1

    print(f"{no_direct_rel_cnt} / {len(id2rel_path_decode_ids)} = {no_direct_rel_cnt / len(id2rel_path_decode_ids)}")

    # pickle.dump(id2pos_ls, open(args.output_file, "wb"))  # use pickle since the key are of type `int`.
    pickle.dump(rel_path_set, open(args.output_file, "wb"))  # V2.0
    if args.path_output_file is not None:
        pickle.dump(id2rel_path_decode_ids, open(args.path_output_file, "wb"))
    if args.rel_vocab is not None:
        vocab = {}
        for exp_id, edge_label_dict in edge_labels.items():
            for edge, edge_label in edge_label_dict.items():
                vocab[f"{exp_id}${edge}"] = edge_label
        pickle.dump(vocab, open(args.rel_vocab, "wb"))  # id2rel_id


if __name__ == '__main__':
    main()

    # 472342
    # 100%|██████████| 472342/472342 [00:01<00:00, 332011.69it/s]
    # 786
    # 100%|██████████| 472342/472342 [00:00<00:00, 652365.50it/s]
    # 712078 / 472342 = 1.507547497364198
    # 657266 / 472342 = 1.3915044607508966
    # 5714 / 3102 = 1.8420373952288847
