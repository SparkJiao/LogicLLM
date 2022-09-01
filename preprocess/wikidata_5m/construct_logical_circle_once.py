import argparse
import collections
import copy
import json
import os
import os.path
from collections import defaultdict
from functools import partial
from multiprocessing import Pool
from typing import List, Dict, Tuple, Set
import random

from tqdm import tqdm

"""
What's different from `construct_logical_circle.py`

The previous script tries to find each possible path, 
which resulting in large storage and difficulty in following process.

e.g., 182739 nodes can induce 147498326 path.
While trying to loading the path into memory, it exceeds more than 100GB memory.

So in this script, for each edge (relation triplet), we only find exactly one path connecting it (with specific length).

Besides, in this script, the triplet is stored in string directly to save memory.
"""

_edges: Dict[str, List[Tuple[str, str]]]
_edge2rel: Dict[str, List[str]]
_triplet2id: Dict[str, int]


def init(graph: Dict[str, List[Tuple[str, str]]], edge2rel: Dict[str, List[str]], triplet2id: Dict[str, int]):
    global _edges
    global _edge2rel
    global _triplet2id
    _edges = graph
    _edge2rel = edge2rel
    _triplet2id = triplet2id


def load_kg_to_edge(kg_file: str, id2ent_file: str, id2rel_file: str):
    id2ent = json.load(open(id2ent_file, 'r'))
    id2rel = json.load(open(id2rel_file, 'r'))

    graph = defaultdict(list)
    edge2rel = defaultdict(list)

    triplets = []
    triplet2id = {}
    with open(kg_file, 'r') as f:
        line = f.readline()
        while line:
            tmp = line.strip().split('\t')
            assert len(tmp) == 3

            line = f.readline()

            s, rel, t = tmp
            if s not in id2ent or t not in id2ent or rel not in id2rel:
                continue

            graph[s].append((rel, t))
            edge2rel[f"{s}\t{t}"].append(rel)  # Should take a note at the direction/order.

            key = '\t'.join(tmp)
            if key not in triplet2id:
                triplet2id[key] = len(triplets)
                triplets.append(key)  # use '\t' for seperator

    return graph, triplet2id, edge2rel


def get_path_unique_id(path: List[Tuple[str, str, str]]):
    # Treat each path as a set of all edges.
    # The order between edges is ignored.
    # The order between nodes is sensitive.
    path_ids = list(map(lambda x: _triplet2id['\t'.join(x)], path))
    path_ids = sorted(path_ids)
    return '#'.join(map(str, path_ids))


def triplet2str(path: List[Tuple[str, str, str]]) -> List[str]:
    return ['\t'.join(triplet) for triplet in path]


def path_check(path: List[Tuple[str, str, str]], min_length: int, max_length: int):
    if not (min_length <= len(path) <= max_length):
        return False

    s = path[0][0]
    t = path[-1][2]
    key = f"{s}\t{t}"
    if key in _edge2rel and len(_edge2rel[key]):
        return True

    return False


def dfs(s: str, t: str, target_depth: int, cur_depth: int,
        path_node_vis: Set[str], path: List[str]):
    """
    Args:

    """

    if cur_depth == target_depth:
        if s == t:
            return True, path
        return False, -1

    for rel, next_n in _edges[s]:
        if next_n in path_node_vis:
            continue

        next_triplet = triplet2str([(s, rel, next_n)])

        path_node_vis.add(next_n)
        res = dfs(next_n, t, target_depth, cur_depth + 1, path_node_vis, path + next_triplet)
        if res[0]:
            return res
        path_node_vis.remove(next_n)

    return False, -1


def dfs_mem(s: str, t: str, target_depth: int, cur_depth: int,
            path_node_vis: Set[str], path: List[str],
            failures: Dict[str, Set[int]]):
    """
    Args:

    """

    if cur_depth == target_depth:
        if s == t:
            return True, path
        return False, -1

    tmp_failure = set()
    for rel, next_n in _edges[s]:
        if next_n in path_node_vis:
            continue

        diff = target_depth - cur_depth
        assert diff > 0

        if next_n in failures and diff in failures[next_n]:
            continue

        next_triplet = triplet2str([(s, rel, next_n)])

        path_node_vis.add(next_n)
        res = dfs(next_n, t, target_depth, cur_depth + 1, path_node_vis, path + next_triplet)
        if res[0]:
            return res
        else:
            ...

        path_node_vis.remove(next_n)

    return False, -1


def dfs_proxy(edge: str, min_depth: int, max_depth: int, sample: bool = False):
    s, t = edge.split("\t")

    all_path = []
    path_node_vis = {s}
    if sample:
        dep = random.choice(list(range(min_depth, max_depth + 1)))
        res = dfs(s, t, dep, 0, path_node_vis, [])
        if res[0]:
            all_path.append(res[1])
    else:
        for dep in range(min_depth, max_depth + 1):
            res = dfs(s, t, dep, 0, path_node_vis, [])
            if res[0]:
                assert res[1] != -1
                all_path.append(res[1])

    return all_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kg", type=str)
    parser.add_argument("--id2ent", type=str)
    parser.add_argument("--id2rel", type=str)
    parser.add_argument("--min_depth", type=int, default=2)
    parser.add_argument("--max_depth", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--sample", default=False, action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split_chunk", type=str, default=None)
    args = parser.parse_args()

    random.seed(args.seed)

    graph, triplet2id, edge2rel = load_kg_to_edge(args.kg, args.id2ent, args.id2rel)
    print(len(graph), len(triplet2id), len(edge2rel))

    edges = list(edge2rel.keys())

    if args.split_chunk is not None:
        total_num = len(edges)
        split, split_id = map(int, args.split_chunk.split(","))
        split_num = total_num // split
        edges = edges[split_id * split_num: (split_id + 1) * split_num]
    else:
        split = 0
        split_id = -1

    all_paths = []
    with Pool(args.num_workers, initializer=init, initargs=(graph, edge2rel, triplet2id)) as p:
        _annotate = partial(dfs_proxy, min_depth=2, max_depth=args.max_depth, sample=args.sample)
        _results = list(tqdm(
            p.imap(_annotate, edges),
            total=len(edges),
            dynamic_ncols=True,
            desc="Searching path"
        ))

        for res in _results:
            all_paths.extend(res)
    print(f"Generate {len(all_paths)} paths.")

    file_name = f"logic_circle_d{args.min_depth}_{args.max_depth}_{args.sample}_s{args.seed}_v2.json"
    if split:
        file_name = file_name[:-5] + f"_{split}_{split_id}.json"

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=False)

    json.dump(all_paths, open(os.path.join(args.output_dir, file_name), 'w'))
    json.dump(edge2rel, open(os.path.join(args.output_dir, "edge2rel.json"), 'w'))

    print(f"Saved file to {os.path.join(args.output_dir, file_name)}")


if __name__ == '__main__':
    main()
