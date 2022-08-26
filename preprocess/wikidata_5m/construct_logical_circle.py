import argparse
import json
import os
import os.path
from collections import defaultdict
from functools import partial
from multiprocessing import Pool
from typing import List, Dict, Tuple, Set

from tqdm import tqdm

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


def load_kg_to_edge(kg_file: str, id2ent_file: str):
    id2ent = json.load(open(id2ent_file, 'r'))

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
            if s not in id2ent or t not in id2ent:
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


def path_check(path: List[Tuple[str, str, str]], min_length: int, max_length: int):
    if not (min_length <= len(path) <= max_length):
        return False

    s = path[0][0]
    t = path[-1][2]
    key = f"{s}\t{t}"
    if key in _edge2rel and len(_edge2rel[key]):
        return True

    return False


def dfs(s: str, t: str, depth: int, vis: Set[str]):
    pass


def bfs(s: str, min_depth: int, max_depth: int):
    queue = [(s, 0, [])]
    node_vis = set(s)

    all_paths = []
    while queue:
        n, cur_depth, cur_path = queue.pop(0)

        if len(cur_path) > max_depth:
            break

        for edge, next_n in _edges[n]:
            if next_n not in node_vis:
                node_vis.add(next_n)

                next_triplet = (n, edge, next_n)
                new_path = cur_path + [next_triplet]

                if len(new_path) < max_depth:
                    queue.append((next_n, cur_depth + 1, new_path))  # Not a path but can be further extended.

                if path_check(new_path, min_depth, max_depth):
                    # new_path_unique_id = get_path_unique_id(new_path)
                    # if new_path_unique_id not in path_vis: # path can be further extended if in different direction.
                    #     all_paths.append(new_path)
                    #     path_vis.add(new_path_unique_id)
                    all_paths.append(new_path)

    return all_paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kg", type=str)
    # parser.add_argument("--triplet2sent", type=str)
    parser.add_argument("--id2ent", type=str)
    parser.add_argument("--max_depth", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    graph, triplet2id, edge2rel = load_kg_to_edge(args.kg, args.id2ent)

    nodes = list(graph.keys())
    # path_visit = set()

    all_paths = []
    with Pool(args.num_workers, initializer=init, initargs=(graph, edge2rel, triplet2id)) as p:
        _annotate = partial(bfs, min_depth=2, max_depth=args.max_depth)
        _results = list(tqdm(
            p.imap(_annotate, nodes, chunksize=32),
            total=len(nodes),
            dynamic_ncols=True,
            desc="Searching path"
        ))

        for res in _results:
            all_paths.extend(res)

    file_name = f"logic_circle_d{args.max_depth}_v1.json"

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=False)

    json.dump(all_paths, open(os.path.join(args.output_dir, file_name), 'w'))
    json.dump(edge2rel, open(os.path.join(args.output_dir, "edge2rel.json"), 'w'))


if __name__ == '__main__':
    main()
