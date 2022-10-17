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

    args = parser.parse_args()
    return args


def load_kg(kg_file):
    edge2rel = {}
    rel2edge = defaultdict(list)
    with open(kg_file, "r") as f:
        line = f.readline()
        while line:
            tmp = line.strip().split("\t")
            s, rel, t = tmp
            edge2rel[f"{s}\t{t}"] = rel
            rel2edge[rel].append(f"{s}\t{t}")

            line = f.readline()

    return edge2rel, rel2edge


def main():
    args = parse_args()

    data = pickle.load(open(args.input_file, "rb"))
    examples = data["examples"]
    print(len(examples))

    # ========================

    # results = list(map(extract_path_and_parse, examples))
    #
    # with Pool(args.num_workers, initializer=init, initargs=(results,)) as p:
    #     cnt = list(tqdm(
    #         p.imap(find_similar_pattern, results, chunksize=32),
    #         desc="counting similar pattern",
    #         total=len(results)
    #     ))
    #
    # cnt = sum(cnt)
    # print(cnt)
    # print(f"{cnt} / {len(results)} = {cnt / len(results)}")

    # ========================

    edge2rel, _ = load_kg(args.kg)

    ent_path_set = defaultdict(list)
    rel_path_set = defaultdict(list)
    results = []
    for example in tqdm(examples):
        ent_path, rel_path, example_id = extract_path_w_rel_and_parse(example, edge2rel)
        ent_path_set[ent_path].append(example_id)

        if rel_path is not None:
            rel_path_set[rel_path].append(example_id)

        results.append((ent_path, rel_path, example_id))

    id2pos_ls = defaultdict(list)
    cnt = 0
    for res in tqdm(results):
        ent_path, rel_path, example_id = res
        tmp_set = set()
        tmp_set.update(set([x for x in ent_path_set[ent_path] if x != example_id]))

        if rel_path is not None:
            tmp_set.update(set([x for x in rel_path_set[rel_path] if x != example_id]))

        id2pos_ls[example_id] = list(tmp_set)
        cnt += len(id2pos_ls[example_id])

    print(f"{cnt} / {len(results)} = {cnt / len(results)}")

    pickle.dump(id2pos_ls, open(args.output_file, "wb"))  # use pickle since the key are of type `int`.


if __name__ == '__main__':
    main()
