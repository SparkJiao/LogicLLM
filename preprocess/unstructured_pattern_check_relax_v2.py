import argparse
import copy
import json
import pickle
from collections import defaultdict
from multiprocessing import Pool
from typing import Tuple, Dict, Union, List, Set
from functools import partial

from tqdm import tqdm


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
            rels.append(-1)
            empty_rel_cnt += 1

    if empty_rel_cnt > limit:
        return "\t".join(ent_path), None, example_id
    else:
        return "\t".join(ent_path), rels, example_id


def extend_rels(rel: str):
    new_rel2edge = defaultdict(set)
    for nxt_rel in __rels__:
        if nxt_rel == rel:
            continue
        new_rel = "$".join([rel, nxt_rel])
        if new_rel in __rels__:
            continue
        for edge_a_right in __rel2right__[rel]:
            if edge_a_right not in __rel2left__[nxt_rel]:
                continue
            for edge_a in __rel2right__[rel][edge_a_right]:
                for edge_b in __rel2left__[nxt_rel][edge_a_right]:
                    edge_a_left, _ = edge_a.split("\t")
                    _, edge_b_right = edge_b.split("\t")
                    new_edge = "\t".join([edge_a_left, edge_b_right])
                    if new_edge in __edges__:
                        continue
                    if new_edge in new_rel2edge[new_rel]:
                        continue
                    new_rel2edge[new_rel].add(new_edge)

    return new_rel2edge


def rels_extend_proxy(edge2rel: Dict[str, str],
                      rel2edge: Dict[str, List[str]],
                      num_workers: int):
    edges = set(list(edge2rel.keys()))
    rels = set(list(rel2edge.keys()))

    rel2left = {}
    rel2right = {}
    for rel, rel_edges in rel2edge.items():
        rel2left[rel] = defaultdict(list)
        rel2right[rel] = defaultdict(list)
        for rel_edge in rel_edges:
            rel_edge_a, rel_edge_b = rel_edge.split("\t")
            if rel_edge_a not in rel2left[rel]:
                rel2left[rel][rel_edge_a].append(rel_edge)
                rel2right[rel][rel_edge_b].append(rel_edge)

    def _init_(_edges, _rels, _rel2left, _rel2right):
        global __edges__
        global __rels__
        global __rel2left__
        global __rel2right__
        __edges__ = _edges
        __rels__ = _rels
        __rel2left__ = _rel2left
        __rel2right__ = _rel2right

    with Pool(num_workers, initializer=_init_, initargs=(edges, rels, rel2left, rel2right)) as p:
        results = list(tqdm(
            p.imap(extend_rels, rels, chunksize=32),
            total=len(rels),
            desc="relaxing..."
        ))

    new_edge2rels = {}
    new_rel2edges = {}
    for res in tqdm(results, total=len(results)):
        for res_k, res_v in res.items():
            assert res_k not in rels
            for new_edge in res_v:
                assert new_edge not in edges
                new_edge2rels[new_edge] = res_k

            if res_k not in new_rel2edges:
                new_rel2edges[res_k] = res_v
            else:
                new_rel2edges[res_k].update(res_v)

    return new_edge2rels, new_rel2edges


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--kg", type=str)
    parser.add_argument("--path_output_file", type=str, default=None)
    parser.add_argument("--rel_vocab", type=str, default=None)
    parser.add_argument("--hop", type=int, default=2)
    parser.add_argument("--limit", type=int, default=1)

    args = parser.parse_args()
    return args


def load_kg(kg_file):
    g = defaultdict(list)
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

            if rel not in rel_vocab:
                rel_vocab[rel] = len(rels)
                rels.append(rel)

            g[s].append((rel, t))

            line = f.readline()

    return edge2rel, rel2edge, rels, rel_vocab, g


def main():
    args = parse_args()

    data = pickle.load(open(args.input_file, "rb"))
    examples = data["examples"]
    print(len(set([exp["id"] for exp in examples])))
    print(len(examples))

    edge2rel, rel2edge, rels, rel_vocab, g = load_kg(args.kg)

    # Add <unk>
    # rel_vocab["<unk>"] = len(rels)
    # rels.append("<unk>")

    # for _ in range(args.hop - 1):
    for _ in range(args.hop):
        print(f"Before relaxation: {len(rels)}")

        relaxed_edges, relaxed_rels = rels_extend_proxy(edge2rel, rel2edge, args.num_workers)
        print(len(relaxed_edges))
        edge2rel.update(relaxed_edges)
        # for _edge, _rel in tqdm(relaxed_edges.items(), total=len(relaxed_edges), desc="Updating rel vocab and graph G"):
        #     if _rel not in rel_vocab:
        #         rel_vocab[_rel] = len(rels)
        #         rels.append(_rel)

        for rel, rel_edges in relaxed_rels.items():
            assert rel not in rel_vocab
            rel_vocab[rel] = len(rels)
            rels.append(rel)
            assert rel not in rel2edge
            rel2edge[rel] = list(rel_edges)

        print(f"After relaxation: {len(rels)}")

    edge_labels = {}
    for edge, rel in edge2rel.items():
        edge_labels[edge] = rel_vocab[rel]

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
                # rel_path_decode_id_b = rel_vocab[edge2rel[direct_edge]]
                id2rel_path_decode_ids[example_id] = {
                    "input_a": rel_path,
                    "input_b": edge_labels[direct_edge],
                }
            else:
                # rel_path_decode_id_b = -1
                no_direct_rel_cnt += 1

            # if rel_path_decode_id_b == -1:

        # id2pos_ls[example_id] = list(tmp_set)
        # cnt += len(id2pos_ls[example_id])

    # print(f"{cnt} / {len(results)} = {cnt / len(results)}")
    # print(f"{cnt - ent_matching_cnt} / {len(results)} = {(cnt - ent_matching_cnt) / len(results)}")
    print(f"{no_direct_rel_cnt} / {len(id2rel_path_decode_ids)} = {no_direct_rel_cnt / len(id2rel_path_decode_ids)}")

    # pickle.dump(id2pos_ls, open(args.output_file, "wb"))  # use pickle since the key are of type `int`.
    pickle.dump(rel_path_set, open(args.output_file, "wb"))
    if args.path_output_file is not None:
        pickle.dump(id2rel_path_decode_ids, open(args.path_output_file, "wb"))
    if args.rel_vocab is not None:
        pickle.dump(edge_labels, open(args.rel_vocab, "wb"))  # id2rel_id
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
