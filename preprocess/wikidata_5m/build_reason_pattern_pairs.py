import argparse
import collections
import functools
import glob
import json
import os
import random
from multiprocessing import Pool
from typing import List, Union, Dict, Set, Tuple

import numpy as np
import torch
from nltk import word_tokenize
from tqdm import tqdm

"""
First construct positive and negative pairs following specific rules:
    1.  Given a reasoning pattern: a -> b -> c -> d; a -> d. (3-hop) 
        Define the positive pairs as: 
            (1) a -> b -> c; a -> c; if a -> c -> d exists (?). (double 2-hop)
            (2) a -> b -> c -> d; a -> d; (3-hop) 
                but the involved entities/concepts/events are different (can be obtained by manual augmentation ?)
            (3) a -> b -> c -> e -> d;
            (4) a -> d; The direct answer. 
        Any hard negative pairs?
        
        A quick experiments: If MERIt could be used for training samples retrieval quickly?
        
        For reader training, we may add a simple post-processing technique to choose the retrieved patterns.
        That is all the relations should be matched (we may need to seek the help from sentence bert to calculate semantic distance).
        Entities are overlooked since the pattern is ignored with the entity.
        
        Positive examples retrieval process:
            (1) Build the key-value map: relation-path -> list of original paths (including entities)
            (2) Find a sub-relation-path in the original relation path -> find the corresponding paths through the map built in (1)
            (3) Check if there exists a direct path connecting the ending entities of the found paths in (2)
            (4) To be continued.
        

Then generate the corresponding text pairs.
"""


def _path2ent_key(path: List[str]):
    s, rel, t = path[0].split("\t")
    ent_ls = [s, t]
    for triplet in path[1:]:
        s, rel, t = triplet.split("\t")
        assert s == ent_ls[-1], path
        ent_ls.append(t)

    path_ent_key = "\t".join(ent_ls)
    return path_ent_key


def _path2key(path: List[str]):
    s, rel, t = path[0].split("\t")
    item_ls = [s, rel, t]
    for triplet in path[1:]:
        s, rel, t = triplet.split("\t")
        assert s == item_ls[-1]
        item_ls.extend([rel, t])
    return "\t".join(item_ls)


def _path2rel_key(path: List[str]):
    rel_ls = []
    for triplet in path:
        _, rel, _ = triplet.split("\t")
        rel_ls.append(rel)
    return "\t".join(rel_ls)


def generate_path_vocab(all_paths: List[List[str]]):
    """
    If we need to generate a vocab to notes the indices of the positive pairs of each sample,
    so that we may mask them out during in-batch negative sampling.
    """
    path_vocab = {}
    cnt = 0
    for path in all_paths:
        path_key = _path2key(path)
        path_rel_key = _path2rel_key(path)
        path_ent_key = _path2ent_key(path)
        path_vocab[path] = {
            "id": cnt,
            "key": path_key,
            "rel_key": path_rel_key,
            "ent_key": path_ent_key,
        }
        cnt += 1

    return path_vocab


def generate_path_mapping(all_paths: List[List[str]], edge2rel: Dict[str, List[str]]):
    warn_cnt_a = 0
    warn_cnt_b = 0
    mapped_rel_num = 0
    rel_mapping = collections.defaultdict(set)
    for path in all_paths:
        path_rel_key = _path2rel_key(path)
        s = path[0].split("\t")[0]
        t = path[-1].split("\t")[-1]
        edge = f"{s}\t{t}"
        mapped_rel = edge2rel[edge]
        if len(mapped_rel) > 1:
            print("warning: Multiple relations between entity pair detected.")
            warn_cnt_a += 1

        rel_mapping[path_rel_key].update(set(mapped_rel))

    for path_rel_key, mapped_rels in rel_mapping.items():
        if len(mapped_rels) > 1:
            if warn_cnt_b == 0:
                print("Warning: Multiple relations detected between single relation path.")
            warn_cnt_b += 1
        mapped_rel_num += len(mapped_rels)

    print(warn_cnt_a)
    print(warn_cnt_b)
    print(mapped_rel_num / len(rel_mapping))

    return rel_mapping


def generate_path_rel_vocab(all_paths: List[List[str]]):
    path_rel_vocab = collections.defaultdict(list)

    for path in all_paths:
        path_rel_key = _path2rel_key(path)
        path_rel_vocab[path_rel_key].append(path)
    return path_rel_vocab


_path_vocab: Dict[str, Dict[str, Union[int, str]]]
_path_rel_vocab: Dict[str, List[List[str]]]  # rel-path to original paths (including the entities)
_rel_mapping: Dict[str, Set[str]]  # rel-path to edge


def rel_pair_init(path_rel_vocab: Dict[str, List[List[str]]], rel_mapping: Dict[str, Set[str]]):
    global _path_rel_vocab
    global _rel_mapping
    _path_rel_vocab = path_rel_vocab
    _rel_mapping = rel_mapping


def find_substring_rel_key(rel_path: str, tgt: str, mapping: str):
    rel_items = rel_path.split("\t")
    tgt_items = tgt.split("\t")
    tgt_len = len(tgt_items)
    for i in range(len(rel_items)):
        if "\t".join(rel_items[i: i + tgt_len]) == tgt:
            mapped_items = rel_items[:i] + [mapping] + rel_items[i + tgt_len:]
            return True, i, "\t".join(mapped_items)
    return False, -1, ""


def find_positive_pairs(path_rel_key: str):
    # path_rel_key = _path2rel_key(path)

    rel_items = path_rel_key.split("\t")
    path_len = len(rel_items)

    results = []

    # TODO: Currently I don't consider the case (3) (maybe too difficult).
    for tgt_len in range(2, path_len):
        for i in range(0, path_len, tgt_len):
            rel_substring_items = rel_items[i: i + tgt_len]
            rel_substring = "\t".join(rel_substring_items)
            if rel_substring not in _rel_mapping or rel_substring not in _path_rel_vocab:
                continue

            # We find a continuous sub-relation-path here and the newly mapped one after reducing can thus
            # be viewed as another positive instance.
            # And we can avoid finding the discontinuous paths since it maybe difficult to implement.
            mapped_path_items = rel_items[:i] + [list(_rel_mapping[rel_substring])[0]] + rel_items[i + tgt_len:]
            # for _mapped_rel in _rel_mapping[rel_substring]:  # TODO: Seems a little counterintuitive here.
            # mapped_path_items = rel_items[:i] + [_mapped_rel] + rel_items[i + tgt_len:]
            mapped_rel_path = "\t".join(mapped_path_items)

            # Ensure that we can use the mapped relation path to find a corresponding text segment.
            # This could be ignored if it is not necessary to also involve the paired text segment.
            if mapped_rel_path not in _path_rel_vocab:
                continue

            results.append((rel_substring, mapped_rel_path))  # Case (1)

    # Original path but different entities; Case (2)
    results.append((path_rel_key, -1))
    # Mapped relation for answer retrieval; Case (4)
    # results.append((list(_rel_mapping[path_rel_key])[0], -1))
    results.extend([(_mapped_rel, -1) for _mapped_rel in _rel_mapping[path_rel_key]])

    return path_rel_key, results


_rel_pos_pairs: Dict[str, List[Tuple[str, Union[str, int]]]]
_triplet2sent: Dict[str, List[str]]
_id2ent: Dict[str, List[str]]
_id2rel: Dict[str, List[str]]


def pattern2text_init(path_rel_vocab: Dict[str, List[List[str]]], rel_mapping: Dict[str, Set[str]],
                      rel_pos_pairs: Dict[str, List[Tuple[str, Union[str, int]]]],
                      triplet2sent: Dict[str, List[str]],
                      id2ent: Dict[str, List[str]],
                      id2rel: Dict[str, List[str]]):
    global _path_rel_vocab
    global _rel_mapping
    global _rel_pos_pairs
    global _triplet2sent
    global _id2ent
    global _id2rel
    _path_rel_vocab = path_rel_vocab
    _rel_mapping = rel_mapping
    _rel_pos_pairs = rel_pos_pairs
    _triplet2sent = triplet2sent
    _id2ent = id2ent
    _id2rel = id2rel


def logical_pattern_to_text_v1(path_rel_key: str, skip_symbols: bool = False, max_query_per_rel: int = 10, max_path_per_rel: int = 10):
    query_paths = _path_rel_vocab[path_rel_key][:max_query_per_rel]
    pos_path_rel_keys = _rel_pos_pairs[path_rel_key]

    pos_paths = []
    single_rels = []
    for sub_key in pos_path_rel_keys:
        if len(sub_key[0].split("\t")) == 1:  # Process mapped relation case and process it separately.
            single_rels.append(sub_key[0])
            continue

        if len(single_rels) + len(pos_paths) >= max_path_per_rel:
            break

        for _path in _path_rel_vocab[sub_key[0]]:
            pos_paths.append((_path, sub_key[0]))

            if len(single_rels) + len(pos_paths) >= max_path_per_rel:
                break

        if len(single_rels) + len(pos_paths) >= max_path_per_rel:
            break

        if sub_key[1] != -1:
            for _path in _path_rel_vocab[sub_key[1]]:
                pos_paths.append((_path, sub_key[1]))

                if len(single_rels) + len(pos_paths) >= max_path_per_rel:
                    break

    positive_pairs = []
    # for query_path in tqdm(query_paths, total=len(query_paths), desc="enumerating over `query_paths`."):
    for query_path in query_paths:
        query = path2text(query_path, skip_symbols=skip_symbols)
        for pos_path, pos_rel_key in pos_paths:
            pos_text = path2text(pos_path, skip_symbols=skip_symbols)
            # Avoid the same entities in case (2) but include it in case (4).
            # In case (4), both the same entities and not are ok since we focus on the relation mapping.
            if _path2ent_key(query_path) == _path2ent_key(pos_path):
                continue
            positive_pairs.append({
                "query_rel_key": path_rel_key,
                "query_text": query,  # Check the empty string later.
                "pos_rel_key": pos_rel_key,
                "pos_text": pos_text,  # Check the empty string later.
            })

        for single_rel in single_rels:
            positive_pairs.append({
                "query_rel_key": path_rel_key,
                "query_text": query,  # Check the empty string later.
                "pos_rel_key": single_rel,
                "pos_text": "#REL"  # Process it later.
            })

    # print(len(positive_pairs))
    return positive_pairs


def text2triplet(triplet: str):
    return triplet.split("\t")


# Copied from preprocess.wikidata_5m.logical_circle_to_text.triplet2texts.
def triplet2texts(triplet: str, triplet2sent, id2ent, id2rel, skip_symbols: bool = False) -> Union[List[Dict[str, str]], None]:
    s, rel, t = triplet.split("\t")
    key = triplet
    texts = []
    if key in triplet2sent:
        for item in triplet2sent[key]:
            s_alias = item["s"]
            t_alias = item["t"]
            text = item["text"]
            assert s_alias in text, (s_alias, t_alias, text)
            assert t_alias in text, (s_alias, t_alias, text)  # FIXED: Found an assertion error here.
            # if s_alias not in text:
            #     continue
            # if t_alias not in text:
            #     continue
            if len(set(word_tokenize(s_alias)) & set(word_tokenize(
                    t_alias))):  # FIXED in `align_triplet_text.py`: This case should be removed during text-triplet aligning.
                continue
            texts.append({
                "s": s,
                "t": t,
                "rel": rel,
                "s_alias": s_alias,
                "t_alias": t_alias,
                "rel_alias": "",
                "text": text,
            })

    if skip_symbols:
        return texts

    if s not in id2ent or t not in id2ent:
        return texts

    s_alias = random.choice(id2ent[s])
    t_alias = random.choice(id2ent[t])
    # FIXED: KeyError here. Should be checked in `align_triplet_text.py`.
    # if rel not in id2rel:
    #     if texts:
    #         return texts
    #     return None
    rel_alias = random.choice(id2rel[rel])

    text = ' '.join([s_alias, rel_alias, t_alias]) + '.'
    texts.append({
        "s": s,
        "t": t,
        "rel": rel,
        "s_alias": s_alias,
        "t_alias": t_alias,
        "rel_alias": rel_alias,
        "text": text
    })

    return texts


def path2text(path: List[str], skip_symbols: bool = False):
    assert isinstance(path, list) and isinstance(path[0], str), path
    res = [triplet2texts(triplet, _triplet2sent, _id2ent, _id2rel, skip_symbols=skip_symbols) for triplet in path]
    if any(len(dic_ls) == 0 for dic_ls in res):
        return ""
    res = [random.choice(dic_ls) for dic_ls in res]
    res = [dic["text"] for dic in res]
    res = " ".join(res)
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--path", type=str)
    parser.add_argument("--edge2rel", type=str)
    parser.add_argument("--triplet2sent", type=str)
    parser.add_argument("--id2ent", type=str)
    parser.add_argument("--id2rel", type=str)
    parser.add_argument("--max_query_per_rel", type=int, default=10)
    parser.add_argument("--max_path_per_rel", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=32)
    parser.add_argument("--pattern_pair_save_file", type=str)
    parser.add_argument("--text_pair_save_file", type=str)
    parser.add_argument("--skip_symbols", default=False, action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if os.path.exists(args.path):
        path_files = [args.path]
    else:
        path_files = sorted(list(glob.glob(args.path)))

    edge2rel = json.load(open(args.edge2rel))

    all_paths = []
    for path_file in path_files:
        print(f"Reading path file from {path_file}...")
        all_path = json.load(open(path_file))
        all_paths.extend(all_path)

    rel_mapping = generate_path_mapping(all_paths, edge2rel)
    path_rel_vocab = generate_path_rel_vocab(all_paths)

    rel_key_candidates = [rel_key for rel_key in rel_mapping.keys() if len(rel_key.split("\t")) > 2]
    with Pool(args.num_workers, initializer=rel_pair_init, initargs=(path_rel_vocab, rel_mapping)) as p:
        positive_pairs = list(tqdm(
            p.imap(find_positive_pairs, rel_key_candidates, chunksize=32),
            total=len(rel_key_candidates),
            desc="Generating positive pairs",
            dynamic_ncols=True
        ))

    rel_pos_pairs = {}
    for rel_key, pos_pairs in positive_pairs:
        tmp = list(set(pos_pairs))
        if len(tmp):
            rel_pos_pairs[rel_key] = tmp

    json.dump(rel_pos_pairs, open(args.pattern_pair_save_file, "w"))
    # print(sum(map(len, positive_pairs)))

    # 需要按照长度对path过滤一下，我们最终只考虑长度大于等于3的path作为query的情况

    id2rel = json.load(open(args.id2rel, "r"))
    id2ent = json.load(open(args.id2ent, "r"))
    triplet2sent = json.load(open(args.triplet2sent, "r"))

    # rel_key_candidates = rel_key_candidates[:1000]
    with Pool(args.num_workers, initializer=pattern2text_init,
              initargs=(path_rel_vocab, rel_mapping, rel_pos_pairs, triplet2sent, id2ent, id2rel)) as p:
        # pattern2text_init(path_rel_vocab, rel_mapping, rel_pos_pairs, triplet2sent, id2ent, id2rel)
        _annotate = functools.partial(logical_pattern_to_text_v1, skip_symbols=args.skip_symbols,
                                      max_query_per_rel=args.max_query_per_rel,
                                      max_path_per_rel=args.max_path_per_rel)
        pairs = list(tqdm(
            p.imap(_annotate, rel_key_candidates, chunksize=4),
            # map(_annotate, rel_key_candidates),
            total=len(rel_key_candidates),
            desc="Generating text pairs",
            dynamic_ncols=True
        ))

    rel2edge = collections.defaultdict(list)
    for edge, rel in edge2rel.items():
        assert isinstance(rel, list)
        rel = rel[0]
        s, t = edge.split("\t")
        rel2edge[rel].append("\t".join([s, rel, t]))

    pattern2text_init(path_rel_vocab, rel_mapping, rel_pos_pairs, triplet2sent, id2ent, id2rel)

    results = []
    for res_ls in tqdm(pairs, total=len(pairs), desc="cleaning results"):
        for res in res_ls:
            if res["query_text"] == "":
                continue
            if res["pos_text"] == "#REL":
                # res["pos_text"] = triplet2texts(random.choice(rel2edge[res["pos_rel_key"]]), id2ent, id2rel, args.skip_symbols)
                # FIXME: s_alias = random.choice(id2ent[s])
                #   KeyError: 'Q5822626'
                res["pos_text"] = path2text([random.choice(rel2edge[res["pos_rel_key"]])], skip_symbols=args.skip_symbols)
            if res["pos_text"] == "":
                continue
            results.append(res)
    print(len(results))

    json.dump(results, open(args.text_pair_save_file, "w"))


if __name__ == '__main__':
    main()
