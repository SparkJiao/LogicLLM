import argparse
import json
import pickle
from collections import defaultdict
from typing import Tuple, Dict, Union

from tqdm import tqdm

RES: Union[Tuple[str, int], Tuple[str, str, int]]


def extract_path_and_parse(example):
    path = example["path"]
    example_id = example["id"]
    ent_path = "\t".join([item[0] for item in path])
    return ent_path, example_id


def extract_path_w_rel_and_parse(example, edge2rel: Dict[str, str]):
    path = example["path"]
    example_id = example["id"]
    rels = []
    ent_path = []
    for i, item in enumerate(path):
        ent_path.append(item[0])
        if i == 0:
            continue
        edge = f"{path[i - 1][0]}\t{item[0]}"
        if edge in edge2rel:
            rels.append(edge2rel[edge])
        else:
            rels.append(None)
    return "\t".join(ent_path), "\t".join(rels) if None not in rels else None, example_id


def init(res):
    global RES
    RES = res


def find_similar_pattern(example):
    ent_path, example_id = example
    cnt = 0
    for _ent_path_b, _exp_id_b in RES:
        if _exp_id_b == example_id:
            continue
        if _ent_path_b == ent_path:
            cnt += 1
    return cnt


def find_similar_pattern_w_rel(example):
    ent_path, rel_path, example_id = example
    cnt = 0
    for _ent_path_b, _rel_path_b, _exp_id_b in RES:
        if _exp_id_b == example_id:
            continue
        if _ent_path_b == ent_path or _rel_path_b == rel_path:
            cnt += 1
    return cnt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--kg", type=str)
    parser.add_argument("--path_output_file", type=str, default=None)
    parser.add_argument("--rel_vocab", type=str, default=None)

    args = parser.parse_args()
    return args


def load_kg(kg_file):
    edge2rel = {}
    rel2edge = defaultdict(list)
    rel_vocab = {}
    rels = []
    with open(kg_file, "r") as f:
        line = f.readline()
        while line:
            tmp = line.strip().split("\t")
            s, rel, t = tmp
            edge2rel[f"{s}\t{t}"] = rel
            rel2edge[rel].append(f"{s}\t{t}")

            rel_vocab[rel] = len(rels)
            rels.append(rel)

            line = f.readline()

    return edge2rel, rel2edge, rels, rel_vocab


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
    examples = rearrange_example_id(data["examples"])
    print(len(set([exp["id"] for exp in examples])))
    print(len(examples))

    edge2rel, _, rels, rel_vocab = load_kg(args.kg)

    ent_path_set = defaultdict(list)
    rel_path_set = defaultdict(list)
    results = []
    for example in tqdm(examples):
        ent_path, rel_path, example_id = extract_path_w_rel_and_parse(example, edge2rel)
        ent_path_set[ent_path].append(example_id)

        if rel_path is not None:
            rel_path_set[rel_path].append(example_id)

        results.append((ent_path, rel_path, example_id))

    print(len(rel_path_set))

    id2pos_ls = defaultdict(list)
    id2rel_path_decode_ids = {}
    ent_matching_cnt = 0
    no_direct_rel_cnt = 0
    cnt = 0
    for res in tqdm(results):
        ent_path, rel_path, example_id = res
        tmp_set = set()
        tmp_set.update(set([x for x in ent_path_set[ent_path] if x != example_id]))
        ent_matching_cnt += len(tmp_set)

        if rel_path is not None:
            tmp_set.update(set([x for x in rel_path_set[rel_path] if x != example_id]))

            rel_path_decode_id_a = [rel_vocab[_rel] for _rel in rel_path.split("\t")]

            ent_path = ent_path.split("\t")
            direct_edge = f"{ent_path[0]}\t{ent_path[-1]}"
            rel_path_decode_id_b = rel_vocab[direct_edge] if direct_edge in rel_vocab else -1
            id2rel_path_decode_ids[example_id] = {
                "input_a": rel_path_decode_id_a,
                "input_b": rel_path_decode_id_b,
            }
            if rel_path_decode_id_b == -1:
                no_direct_rel_cnt += 1

        id2pos_ls[example_id] = list(tmp_set)
        cnt += len(id2pos_ls[example_id])

    print(f"{cnt} / {len(results)} = {cnt / len(results)}")
    print(f"{cnt - ent_matching_cnt} / {len(results)} = {(cnt - ent_matching_cnt) / len(results)}")
    print(f"{no_direct_rel_cnt} / {len(id2rel_path_decode_ids)} = {no_direct_rel_cnt / len(id2rel_path_decode_ids)}")

    pickle.dump(id2pos_ls, open(args.output_file, "wb"))  # use pickle since the key are of type `int`.
    if args.path_output_file is not None:
        pickle.dump(id2rel_path_decode_ids, open(args.path_output_file, "wb"))
    if args.rel_vocab is not None:
        pickle.dump(rels, open(args.rel_vocab, "wb"))  # id2rel_id


if __name__ == '__main__':
    main()

    # 472342
    # 100%|██████████| 472342/472342 [00:01<00:00, 332011.69it/s]
    # 786
    # 100%|██████████| 472342/472342 [00:00<00:00, 652365.50it/s]
    # 712078 / 472342 = 1.507547497364198
    # 657266 / 472342 = 1.3915044607508966
    # 5714 / 3102 = 1.8420373952288847
