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


def dfs(s: str, t: str, max_depth: int, cur_depth: int,
        path_node_vis: Set[str],
        res: Dict[str, Dict[str, List[List[Tuple[str, str, str]]]]]):
    """
    :param:
        `res`: Maintain the path with length `k` from `s` to `t` as res[s][k]
    """

    if s == t:
        # res[s] = {}
        # res[s]["0"] = [[]]  # Padding: [] + [prev_triplet] = [prev_triplet] // 0 + 1 = 1
        return

    if cur_depth >= max_depth:
        return

    for rel, next_n in _edges[s]:
        next_triplet = (s, rel, next_n)

        if next_n not in res:
            if next_n not in res:
                new_path_node_vis = copy.deepcopy(path_node_vis)
                new_path_node_vis.add(next_n)
                dfs(next_n, t, max_depth, cur_depth + 1, new_path_node_vis, res)

        if next_n not in res:
            return

        for path_len, next_n_path_ls in res[next_n].items():
            # new_len = depth + 1 + int(path_len)
            new_len = 1 + int(path_len)
            if new_len <= max_depth:
                if str(new_len) not in res[s]:
                    res[s][str(new_len)] = []
                for next_n_path in next_n_path_ls:
                    res[s][str(new_len)].append([next_triplet] + next_n_path)


def dfs_proxy(edge: str, min_depth: int, max_depth: int):
    # res = collections.defaultdict(dict)
    # for _, ending in _edges[s]:
    #     assert ending != s
    #     res[ending]["0"] = [[]]  # Padding: [] + [prev_triplet] = [prev_triplet] // 0 + 1 = 1
    #     node_vis = {s}
    #     dfs(s, ending, max_depth, 0, node_vis, res)

    s, t = edge.split("\t")
    # print(s, t)
    res = collections.defaultdict(dict)
    res[t]["0"] = [[]]
    dfs(s, t, max_depth, 0, {s}, res)

    all_path = []
    for path_len, path_ls in res[s].items():
        if int(path_len) < min_depth or int(path_len) > max_depth:
            continue
        for path in path_ls:
            assert path_check(path, min_depth, max_depth)
            all_path.append(path)
    # print(len(all_path))
    return all_path


"""
Annotation in Chinese for memory in BFS (maybe translated into English afterwards):

bfs中应用memory的原理是：
当访问一个节点时，如果该节点已经进入过队列，则不再添加该节点，改为通过memory维护经过该节点的path，
并记录到达所有到达该节点时因该节点已经在队列中而没有继续搜索所产生的序列前缀
在bfs完成后，枚举所有序列前缀，从memory中找到该前缀最后一个节点所经过的所有路径的后缀，将前缀和后缀逐个拼接即可得到新的路径

bfs的特点保证了，当一个节点已经进入队列时，如果该节点之前没有找到一条符合要求的path，那么再次进入也依然不会找到一条新的path
因为该节点最先进入队列时，一定是从源节点到该节点的最短路径，如果没有符合要求的path，要么是超过了长度限制，要么是不存在对应的路径，
后者可以理解，不需要重复搜索；对于前者，既然bfs保证了到达某个节点的最短距离，
那么后续再次访问该节点时不可能距离更短，因此即使找到路径也一定会超过长度限制。

对于每个节点，后缀memory中可能保存了多条关于该节点的记录，因为从该节点出发可能存在多条合法的路径，但这些路径不存在公共节点；

FIXME: 这里似乎有问题，在拼接的时候，没有考虑到重复节点的问题（即有可能出现自环）
FIXED: 加了一个set的判断
"""


def bfs(s: str, min_depth: int, max_depth: int):
    queue = [(s, 0, [])]

    all_paths = []
    while queue:
        n, cur_depth, cur_path = queue.pop(0)

        if len(cur_path) >= max_depth:
            break

        path_nodes = set()
        if len(cur_path):
            path_nodes.add(cur_path[0][0])
            for triplet in cur_path:
                path_nodes.add(triplet[2])

        for edge, next_n in _edges[n]:
            # if next_n not in node_vis:
            if next_n not in path_nodes:  # No self-circle.
                # node_vis.add(next_n)  # This is not searching for a shorted path, so is this step correct? or we should use dfs instead?
                # We shouldn't maintain the set of all visited nodes, which causes that all 1-hop nodes are visited at first
                # and no path will end at the 1-hop nodes,
                # meaning that no direct relation can be as the bridge (`path_check` method always return `False`).

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

    # print(len(all_paths))
    return all_paths


def bfs_memory(s: str, min_depth: int, max_depth: int):
    queue = [(s, [])]

    all_paths = []
    prefix_memory = []
    suffix_memory = defaultdict(list)
    queue_vis = set()

    path_id_set = set()

    while queue:
        n, cur_path = queue.pop(0)

        if len(cur_path) >= max_depth:
            break

        path_nodes = set()
        if len(cur_path):
            path_nodes.add(cur_path[0][0])
            for triplet in cur_path:
                path_nodes.add(triplet[2])

        for edge, next_n in _edges[n]:
            # if next_n not in node_vis:
            if next_n not in path_nodes:  # No self-circle.
                # node_vis.add(next_n)  # This is not searching for a shorted path, so is this step correct? or we should use dfs instead?
                # We shouldn't maintain the set of all visited nodes, which causes that all 1-hop nodes are visited at first
                # and no path will end at the 1-hop nodes,
                # meaning that no direct relation can be as the bridge (`path_check` method always return `False`).

                next_triplet = (n, edge, next_n)
                new_path = cur_path + [next_triplet]

                if next_n in queue_vis:
                    prefix_memory.append((new_path, next_n))
                    continue

                if len(new_path) > 1:  # A new rule to process the one-hop nodes during using memory.
                    queue_vis.add(next_n)

                if len(new_path) < max_depth:
                    queue.append((next_n, new_path))  # Not a path but can be further extended.

                if path_check(new_path, min_depth, max_depth):
                    # new_path_unique_id = get_path_unique_id(new_path)
                    # if new_path_unique_id not in path_vis: # path can be further extended if in different direction.
                    #     all_paths.append(new_path)
                    #     path_vis.add(new_path_unique_id)

                    all_paths.append(new_path)
                    new_path_id = get_path_unique_id(new_path)
                    if new_path_id in path_id_set:
                        print(f"Warning: repeat path during bfs: {cur_path}")
                    path_id_set.add(new_path_id)

                    for triplet_id, triplet in enumerate(new_path):
                        if triplet_id == 0:
                            continue
                        suffix_memory[triplet[0]].append(new_path[triplet_id:])

    for prefix, next_n in prefix_memory:
        if next_n not in suffix_memory:
            continue

        prefix_node = set([triplet[2] for triplet in prefix])
        for suffix in suffix_memory[next_n]:
            if len(prefix) + len(suffix) > max_depth:
                continue

            suffix_node = set([triplet[2] for triplet in suffix])
            if len(prefix_node & suffix_node) > 0:
                continue

            concat_path = prefix + suffix
            concat_path_id = get_path_unique_id(concat_path)

            if concat_path_id in path_id_set:
                print(f"Warning: repeat path during concat: {concat_path}")
            else:
                path_id_set.add(concat_path_id)
                all_paths.append(concat_path)

    # print(len(all_paths))
    return all_paths


def memorized_bfs(edge: str, min_depth: int, max_depth: int):
    s, t = edge.split("\t")

    queue = [(s, 0, [])]
    queue_vis = {s}
    res = defaultdict(list)
    all_paths = []
    path_unique_id_set = set()
    memory: List[Tuple[List[Tuple[str, str, str]], str]] = []

    while queue:
        n, cur_depth, cur_path = queue.pop(0)

        if n == t:
            if len(cur_path) > 1:
                for triplet_id, triplet in enumerate(cur_path):
                    res[triplet[0]].append(cur_path[triplet_id:])

                if len(cur_path) > min_depth:
                    assert len(cur_path) <= max_depth
                    all_paths.append(cur_path)
                    path_unique_id = get_path_unique_id(cur_path)
                    if path_unique_id in path_unique_id_set:
                        print(f"Warning: repeat path during bfs: {cur_path}")
                    path_unique_id_set.add(path_unique_id)
            continue

        if len(cur_path) >= max_depth:
            break

        for rel, next_n in _edges[n]:
            if next_n in queue_vis:
                memory.append((cur_path, next_n))
            else:
                queue_vis.add(next_n)

                next_triplet = (n, rel, next_n)
                new_path = cur_path + [next_triplet]

                queue.append((next_n, cur_depth + 1, new_path))

    memory = sorted(memory, key=lambda x: len(x[0]))

    for path_mem, next_n in memory:
        if next_n not in res:
            continue
        for res_path in res[next_n]:
            concat_path = path_mem + res_path
            path_unique_id = get_path_unique_id(concat_path)
            if path_unique_id in path_unique_id_set:
                print(f"Warning: repeat path during concat: {concat_path}")
            else:
                path_unique_id_set.add(path_unique_id)
                all_paths.append(concat_path)

    return all_paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kg", type=str)
    # parser.add_argument("--triplet2sent", type=str)
    parser.add_argument("--id2ent", type=str)
    parser.add_argument("--max_depth", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--split_chunk", type=str, default=None)
    args = parser.parse_args()

    graph, triplet2id, edge2rel = load_kg_to_edge(args.kg, args.id2ent)
    print(len(graph), len(triplet2id), len(edge2rel))

    nodes = list(graph.keys())
    # path_visit = set()

    if args.split_chunk is not None:
        total_num = len(nodes)
        split, split_id = map(int, args.split_chunk.split(","))
        split_num = total_num // split
        nodes = nodes[split_id * split_num: (split_id + 1) * split_num]
    else:
        split = 0
        split_id = -1

    all_paths = []
    with Pool(args.num_workers, initializer=init, initargs=(graph, edge2rel, triplet2id)) as p:
        # _annotate = partial(bfs, min_depth=2, max_depth=args.max_depth)
        _annotate = partial(bfs_memory, min_depth=2, max_depth=args.max_depth)  # TODO: Fix bug.
        # _annotate = partial(dfs_proxy, min_depth=2, max_depth=args.max_depth)
        # _annotate = partial(memorized_bfs, min_depth=2, max_depth=args.max_depth)
        _results = list(tqdm(
            p.imap(_annotate, nodes, chunksize=32),
            # p.imap(_annotate, list(edge2rel.keys()), chunksize=32),
            # total=len(edge2rel),
            total=len(nodes),
            dynamic_ncols=True,
            desc="Searching path"
        ))

        for res in _results:
            all_paths.extend(res)
    print(f"Generate {len(all_paths)} paths.")

    file_name = f"logic_circle_d{args.max_depth}_v1.json"
    if split:
        file_name = file_name[:-5] + f"_{split}_{split_id}.json"

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=False)

    json.dump(all_paths, open(os.path.join(args.output_dir, file_name), 'w'))
    json.dump(edge2rel, open(os.path.join(args.output_dir, "edge2rel.json"), 'w'))


if __name__ == '__main__':
    main()
