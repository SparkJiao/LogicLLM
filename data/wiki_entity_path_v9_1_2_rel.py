import collections
import copy
import os
import pickle
import random
from functools import partial
from multiprocessing import Pool
from typing import Dict, List

import torch
from torch.distributions.geometric import Geometric
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from data.collators.wiki import WikiPathDatasetV6wPatternPair, WikiPathDatasetRelGenerateV1, WikiPathDatasetRelGenerateV3
from data.collators.wiki_structure_pair import WikiPathDatasetV6wPair
from data.data_utils import get_all_permutation, get_sep_tokens
from general_util.logger import get_child_logger

"""
Version 6.0:
    During augmentation the dataset, if some negative samples should be sampled from other data items, use the corresponding head/tail
    entities for replacement, so that at least the these negative samples are model fluent to remove the bias.
Version 7.0:
    1.  Prior to use the examples from the same item.
Version 7.1:
    1.  Fix some minors.
Version 8.0:
    1.  Add replacement of context sentence.
Version 8.1:
    1.  Randomly sample negative samples instead of sampling sequentially.
Version 8.2:
    1.  Add hyper-parameters to control how much negative samples of each category to be involved in.
Version 9.0:
    1.  Add random sentences into context as noise.
Version split:
    1.  Don't combine the sentences together.
Version VQVAE:
    1.  Add the index of positive candidate and path sentences.
Version v9.1.2:
    1.  Add head-tail character positions for tagging / span extraction
Version rel:
    1.  Add relation grouped examples.
"""

logger = get_child_logger("Wiki.Entity.Path.V9.1.2")

_entity_pool: Dict
_negative_pool: Dict
_rel2sent: Dict
_sent2rel: Dict
_all_neg_candidates: Dict
_all_path_sentences: Dict
_geometric_dist: torch.distributions.Distribution

_permutation_sample_num: int = 6
MAX_NEG_SAMPLE_NUM: int = 8


def _switch_replace_neg(candidate, h_ent_id, h_ent_str, t_ent_id, t_ent_str, rep_pairs: Dict[int, str] = None):
    """
    Enumerate all the possible triplets and replace.
    """
    entities = candidate["ent"]

    non_target_ent_ids = [ent_id for ent_id in entities if ent_id not in [h_ent_id, t_ent_id]]
    h_t_ent_ids = [h_ent_id, t_ent_id]
    assert h_ent_id in entities and t_ent_id in entities

    str_map = {
        h_ent_id: h_ent_str,
        t_ent_id: t_ent_str
    }

    ent_name_set = {
        ent_id: set([_mention["name"] for _mention in entities[ent_id]]) for ent_id in entities
    }
    if rep_pairs is None:
        rep_pairs = {}

    # Currently, we only sample exactly one non-target
    neg_res = []
    for _non_tgt in non_target_ent_ids:
        _non_tgt_str = get_ent_str(candidate, _non_tgt)
        str_map[_non_tgt] = _non_tgt_str

        _source = h_t_ent_ids + [_non_tgt]
        _target = h_t_ent_ids + [_non_tgt]

        # The ``get_all_permutation`` function ensure that the obtained permutations
        # are **all** not the same with the initial permutation.
        _all_perm = get_all_permutation(_target)

        if _permutation_sample_num < len(_all_perm):
            _perm_sample_ls = random.sample(_all_perm, _permutation_sample_num)
        else:
            _perm_sample_ls = _all_perm

        for _perm in _perm_sample_ls:
            assert len(_perm) == len(_source)
            assert _perm != _source
            _rep_pairs_copy = copy.deepcopy(rep_pairs)
            _same_n = 0
            for _src, _tgt in zip(_source, _perm):
                _rep_pairs_copy[_src] = rep_pairs[_tgt] if _tgt in rep_pairs else str_map[_tgt]
                if _rep_pairs_copy[_src].lower() == str_map[_src].lower() or _rep_pairs_copy[_src].lower() in ent_name_set[_src]:
                    _same_n += 1
            if _same_n == len(_source):
                continue
            neg_res.append(_replace_entities_w_str(candidate, _rep_pairs_copy))

    return neg_res


def replace_ent_neg_double(candidate, h_ent_id, h_ent_str, t_ent_id, t_ent_str, rep_pairs: Dict[int, str] = None,
                           out_of_domain: bool = False):
    # `out_of_domain` should always be `False` since all entities use the unified id system from Wikidata.
    assert out_of_domain is False

    if rep_pairs is not None and h_ent_id in rep_pairs:
        h_ent_str = rep_pairs[h_ent_id]
    if rep_pairs is not None and t_ent_id in rep_pairs:
        t_ent_str = rep_pairs[t_ent_id]

    entities = candidate["ent"]

    if rep_pairs is None or out_of_domain:
        rep_pairs = {}

    # id2ent = {ent_id: ent.values() for ent_id, ent in entities.items()}
    id2ent = entities

    if out_of_domain:
        filtered_entities = list(entities.keys())
        h_t_entities = []
    else:
        filtered_entities = [ent_id for ent_id in entities if ent_id not in [h_ent_id, t_ent_id]]
        h_t_entities = []
        for _tmp in [h_ent_id, t_ent_id]:
            if _tmp in entities:
                h_t_entities.append(_tmp)
        assert len(h_t_entities) < 2, (candidate["ent"], (h_ent_str, t_ent_str))

    if len(filtered_entities) == 0:
        return []

    _source_ls = []

    if len(h_t_entities) == 1:
        for tgt in filtered_entities:
            _source_ls.append([h_t_entities[0], tgt])
    else:
        tgt_num = len(filtered_entities)
        for tgt_id_1 in range(tgt_num):
            for tgt_id_2 in range(tgt_id_1 + 1, tgt_num):
                tgt1 = filtered_entities[tgt_id_1]
                tgt2 = filtered_entities[tgt_id_2]
                assert tgt1 != tgt2
                _source_ls.append([tgt1, tgt2])

    neg_res = []
    for _perm in _source_ls:
        tgt_str = (h_ent_str, t_ent_str)
        tgt_ls = [
            (_perm[0], _perm[1]),
            (_perm[1], _perm[0])
        ]

        for tgt in tgt_ls:
            name_set_1 = set([_tmp_ent_mention["name"] for _tmp_ent_mention in id2ent[tgt[0]]])
            name_set_2 = set([_tmp_ent_mention["name"] for _tmp_ent_mention in id2ent[tgt[1]]])
            if h_ent_str.lower() in name_set_1 and t_ent_str.lower() in name_set_2:
                continue

            # If out of domain, ``rep_pairs`` is already empty.
            _cur_rep_pairs_copy = copy.deepcopy(rep_pairs)

            _cur_rep_pairs_copy[tgt[0]] = tgt_str[0]
            _cur_rep_pairs_copy[tgt[1]] = tgt_str[1]

            neg_res.append(_replace_entities_w_str(candidate, _cur_rep_pairs_copy))

    return neg_res


def _replace_entities_w_str(candidate, rep_pairs: Dict[int, str], enforce_h_t: bool = False):
    candidate = copy.deepcopy(candidate)  # Avoid in-place modification of `r["tgt"]`
    ent_to_rep = []

    ent_vis = set()

    for ent_id in candidate["ent"]:
        if ent_id in rep_pairs:
            for r in candidate["ent"][ent_id]:
                r["tgt"] = rep_pairs[ent_id]
                ent_to_rep.append(r)
            assert ent_id not in ent_vis
            ent_vis.add(ent_id)

    if enforce_h_t:
        h_id = candidate["h"]
        t_id = candidate["t"]
        if h_id not in rep_pairs:
            for r in candidate["ent"][h_id]:
                assert r["id"] not in ent_vis
                r["tgt"] = pos2str(r["pos"][0], r["pos"][1], candidate["sent"])
                ent_to_rep.append(r)
        if t_id not in rep_pairs:
            for r in candidate["ent"][t_id]:
                assert r["id"] not in ent_vis
                r["tgt"] = pos2str(r["pos"][0], r["pos"][1], candidate["sent"])
                ent_to_rep.append(r)

    re = sorted(ent_to_rep, key=lambda x: x["pos"][0])
    # Non-overlapping check.
    for _tmp_id, _tmp in enumerate(re):
        if _tmp_id == 0:
            continue
        assert _tmp["pos"][0] >= re[_tmp_id - 1]["pos"][1]

    new_spans = []
    _last_e = 0
    for r in re:
        s, e = r["pos"]
        if s > _last_e:
            new_spans.append(" ".join(candidate["sent"][_last_e: s]))
        r["span_index"] = len(new_spans)
        new_spans.append(r["tgt"])
        _last_e = e

    if _last_e < len(candidate["sent"]):
        new_spans.append(" ".join(candidate["sent"][_last_e:]))

    return {
        "spans": new_spans,
        "entity_replacement": re
    }


def annotate_entity(sent_candi: Dict):
    sent_candi = copy.deepcopy(sent_candi)  # Avoid in-place modification of `r["tgt"]`
    h_id = sent_candi["h"]
    t_id = sent_candi["t"]
    re = []
    for r in sent_candi["ent"][h_id] + sent_candi["ent"][t_id]:
        r["tgt"] = pos2str(r["pos"][0], r["pos"][1], sent_candi["sent"])
        re.append(r)

    re = sorted(re, key=lambda x: x["pos"][0])
    # Non-overlapping check.
    for _tmp_id, _tmp in enumerate(re):
        if _tmp_id == 0:
            continue
        assert _tmp["pos"][0] >= re[_tmp_id - 1]["pos"][1]

    new_spans = []
    _last_e = 0
    for r in re:
        s, e = r["pos"]
        if s > _last_e:
            new_spans.append(" ".join(sent_candi["sent"][_last_e: s]))
        r["span_index"] = len(new_spans)
        new_spans.append(r["tgt"])
        _last_e = e

    if _last_e < len(sent_candi["sent"]):
        new_spans.append(" ".join(sent_candi["sent"][_last_e:]))

    return {
        "spans": new_spans,
        "entity_replacement": re,
        "h": h_id,
        "t": t_id
    }


def replace_neg(pos_candidate, neg_candidate, rep_pairs: Dict[int, str] = None, out_of_domain: bool = False):
    h_ent_str = get_ent_str(pos_candidate, pos_candidate["h"])

    t_ent_str = get_ent_str(pos_candidate, pos_candidate["t"])

    _rep_res = replace_ent_neg_double(neg_candidate, pos_candidate["h"], h_ent_str, pos_candidate["t"], t_ent_str,
                                      rep_pairs=rep_pairs, out_of_domain=out_of_domain)

    return _rep_res


def switch_replace_neg(pos_candidate, neg_candidate, rep_pairs: Dict[int, str] = None):
    h_ent_str = get_ent_str(pos_candidate, pos_candidate["h"])

    t_ent_str = get_ent_str(pos_candidate, pos_candidate["t"])

    _rep_res = _switch_replace_neg(neg_candidate, pos_candidate["h"], h_ent_str, pos_candidate["t"], t_ent_str,
                                   rep_pairs=rep_pairs)

    return _rep_res


def get_ent_str(candi, ent_id):
    tokens = candi["sent"]
    # ent_mentions = list(candi["ent"][ent_id].values())
    ent_mentions = candi["ent"][ent_id]

    mention = random.choice(ent_mentions)
    return pos2str(mention["pos"][0], mention["pos"][1], tokens)


def pos2str(ent_s, ent_e, tokens):
    return " ".join(tokens[ent_s: ent_e])


def sample_entity(pool, path_ent_ids, k):
    all_ent_id_ls = list(pool.keys())

    res = []
    pool_vis = copy.deepcopy(path_ent_ids)
    for _ in range(k):
        ent_id = random.choice(all_ent_id_ls)
        while ent_id in pool_vis:
            ent_id = random.choice(all_ent_id_ls)

        pool_vis.add(ent_id)

        entity_str = random.choice(list(pool[ent_id]))
        res.append(entity_str)
    return res


def rep_context_sent(selected_sentences, tgt_sent_id, rep_tgt_sent):
    context_sentences = {s_id: annotate_entity(s) for s_id, s in selected_sentences.items()}
    rep_tgt_sent["h"] = context_sentences[tgt_sent_id]["h"]
    rep_tgt_sent["t"] = context_sentences[tgt_sent_id]["t"]
    context_sentences[tgt_sent_id] = rep_tgt_sent

    return list(context_sentences.values())


def context_replace_neg(context_sent, neg_candidate, rep_pairs: Dict[int, str] = None, out_of_domain: bool = False):
    assert out_of_domain is False
    _common_num = 0
    if context_sent["h"] in neg_candidate["ent"]:
        _common_num += 1
    if context_sent["t"] in neg_candidate["ent"]:
        _common_num += 1

    if _common_num == 2 and not out_of_domain:
        _rep_res = switch_replace_neg(context_sent, neg_candidate, rep_pairs=rep_pairs)
        flag = 0
    else:
        _rep_res = replace_neg(context_sent, neg_candidate, rep_pairs=rep_pairs, out_of_domain=out_of_domain)
        flag = 1

    return _rep_res, flag


def sample_context_sentence(num: int, cur_item):
    sampled_item_key = random.choice(list(_all_path_sentences.keys()))
    while sampled_item_key == cur_item["id"] or len(_all_path_sentences[sampled_item_key]) < num:
        sampled_item_key = random.choice(list(_all_path_sentences.keys()))

    return random.sample(_all_path_sentences[sampled_item_key], num)


def insert_sentences(orig_sentences: List[str], new_sentences: List[str], shuffle: bool = True) -> List[str]:
    index = [0] * len(orig_sentences) + [1] * len(new_sentences)
    random.shuffle(index)
    new_s_idx_map = list(range(len(new_sentences)))
    if shuffle:
        random.shuffle(new_s_idx_map)

    res = []
    orig_idx = 0
    new_idx = 0
    for x in index:
        if x == 0:
            res.append(orig_sentences[orig_idx])
            orig_idx += 1
        else:
            res.append(new_sentences[new_s_idx_map[new_idx]])
            new_idx += 1
    assert orig_idx == len(orig_sentences), (orig_idx, len(orig_sentences))
    assert new_idx == len(new_sentences), (new_idx, len(new_sentences))
    return res


def add_noise_sentence(num: int, item: Dict, orig_sentences: List[str], rep_pairs: Dict[int, str] = None,
                       shuffle: bool = False) -> List[str]:
    assert num == 0, num
    if num == 0:
        # return " ".join(orig_sentences)
        return orig_sentences

    noise_src_sent = sample_context_sentence(num, item)
    noise_tgt_sent = random.sample(list(item["selected_sentences"].values()), num)
    noise_sent = []
    for noise_tgt, noise_s in zip(noise_tgt_sent, noise_src_sent):
        # _res = replace_neg(noise_tgt, noise_s, rep_pairs=rep_pairs, out_of_domain=False)
        _res, _ = context_replace_neg(noise_tgt, noise_s, rep_pairs=rep_pairs, out_of_domain=False)
        if len(_res) > 0:
            noise_sent.append(_res[0])

    noise_ctx = insert_sentences(orig_sentences, noise_sent, shuffle=shuffle)
    # return " ".join(noise_ctx)
    return noise_ctx


def extract_replaced_path_entity(sent, rep_pairs: Dict[int, str] = None):
    ent_str_ls = []
    h_id = sent["h"]
    t_id = sent["t"]
    if rep_pairs is not None and h_id in rep_pairs:
        ent_str_ls.append(rep_pairs[h_id])
    else:
        ent_str_ls.append(get_ent_str(sent, h_id))

    if rep_pairs is not None and t_id in rep_pairs:
        ent_str_ls.append(rep_pairs[t_id])
    else:
        ent_str_ls.append(get_ent_str(sent, t_id))

    assert len(ent_str_ls) == 2, (sent, rep_pairs)

    return ent_str_ls


def _initializer(entity_pool: Dict, negative_pool: Dict, all_neg_candidates: Dict, all_path_sentences: Dict,
                 rel2sent: Dict, sent2rel: Dict,
                 geometric_dist: torch.distributions.Distribution,
                 max_neg_samples_num: int):
    global _entity_pool
    global _negative_pool
    global _all_neg_candidates
    global _all_path_sentences
    global _geometric_dist
    global _rel2sent
    global _sent2rel
    global MAX_NEG_SAMPLE_NUM

    _entity_pool = entity_pool
    _negative_pool = negative_pool
    _all_neg_candidates = all_neg_candidates
    _all_path_sentences = all_path_sentences
    _geometric_dist = geometric_dist
    MAX_NEG_SAMPLE_NUM = max_neg_samples_num

    _rel2sent = rel2sent
    _sent2rel = sent2rel


def get_sent_unique_id(item, s_id):
    item_id = item["id"].split("_")[:-1]
    return "_".join(item_id + [str(s_id)])


def get_sent2rel(rel2sent: Dict):
    sent2rel = {}
    for rel, rel_sents in rel2sent.items():
        for sent in rel_sents:
            s_id = sent["s_id"]
            item_id = sent["id"].split("_")[:-1]
            s_id = "_".join(item_id + [str(s_id)])
            sent2rel[s_id] = rel
    return sent2rel


def sample_neg_sent_w_rel(rel, h_id, t_id):
    all_rels = list(_rel2sent.keys())
    ht_set = {h_id, t_id}
    neg_sents = []

    tmp_cnt = 0
    while len(neg_sents) < MAX_NEG_SAMPLE_NUM:
        tmp_cnt += 1

        neg_rel = random.choice(all_rels)
        while neg_rel == rel:
            neg_rel = random.choice(all_rels)

        sents = _rel2sent[neg_rel]
        neg_s = random.choice(sents)
        if {neg_s["h"], neg_s["t"]} & ht_set == 2:
            continue

        neg_sents.append(neg_s)

        if tmp_cnt > 100:
            break

    return neg_sents


def sample_pos_sent_w_rel(rel, h_id, t_id):
    ht_set = {h_id, t_id}
    sents = _rel2sent[rel]

    if len(sents) <= 1:
        return None

    pos_sent = None

    tmp_cnt = 0
    while pos_sent is None:
        tmp_cnt += 1

        tmp_sent = random.choice(sents)
        if {tmp_sent["h"], tmp_sent["t"]} & ht_set == 2:
            continue

        pos_sent = tmp_sent

        if tmp_cnt > 100:
            break

    return pos_sent


def replace_w_rel(neg_sent, pos_sent, rep_pairs: Dict[str, str] = None):
    if rep_pairs is None:
        rep_pairs = {}

    # FIXME: 这里会出现 <h, t> 和 <t, h> 代表的是不同的被标注好的relation吗，如果存在这种情况 在替换的时候需要注意顺序
    h_ent_str = get_ent_str(pos_sent, pos_sent["h"])
    t_ent_str = get_ent_str(pos_sent, pos_sent["t"])

    # neg_sent: neg_h -> pos_h
    #           neg_t -> pos_t
    #           if pos_h in rep_pairs: neg_h -> rep_pairs[pos_h]
    _cur_rep_pairs_copy = copy.deepcopy(rep_pairs)
    if pos_sent["h"] in rep_pairs:
        _cur_rep_pairs_copy[neg_sent["h"]] = rep_pairs[pos_sent["h"]]
    else:
        _cur_rep_pairs_copy[neg_sent["h"]] = h_ent_str

    if pos_sent["t"] in rep_pairs:
        _cur_rep_pairs_copy[neg_sent["t"]] = rep_pairs[pos_sent["t"]]
    else:
        _cur_rep_pairs_copy[neg_sent["t"]] = t_ent_str

    return _replace_entities_w_str(neg_sent, _cur_rep_pairs_copy)


def get_s_id2edge_rank(path):
    s_id2edge_rank = {}
    rel_cnt = 0
    for i, item in enumerate(path):
        if i == 0:
            assert item[1] == -1
            continue
        s_id2edge_rank[item[1]] = rel_cnt
        rel_cnt += 1
    return s_id2edge_rank


def extract_path_sent_follow_order(example):
    path = example["path"]
    rel_s_ids = []
    for i, item in enumerate(path):
        if i == 0:
            continue
        rel_s_ids.append(item[1])
    return rel_s_ids


def _process_single_item(item, max_neg_num: int, aug_num: int, min_rep_num: int, shuffle_context: bool,
                         deduct_ratio: float = 1.0, context_ratio: float = 1.0, noise_sent_ratio: float = 0.5,
                         remove_deduct: bool = False, remove_context: bool = False):
    examples = []
    context_examples = []

    noise_sent_num = int(len(item["selected_sentences"]) * noise_sent_ratio)

    selected_sentences = item["selected_sentences"]
    if len(selected_sentences) == 0:
        return []

    orig_sentences = [annotate_entity(s) for s_id, s in selected_sentences.items()]
    sent_w_rel_candidates = []
    for s_id, s in selected_sentences.items():
        s_unique_id = get_sent_unique_id(item, s_id)
        if s_unique_id in _sent2rel and len(_rel2sent[_sent2rel[s_unique_id]]) > 1:
            sent_w_rel_candidates.append((s_id, s))

    path_sent2rep = [(s_id, s) for s_id, s in selected_sentences.items() if s["h"] != -1 and s["t"] != -1]

    # neg_candidates = [x for x in item["rest_sentences"].values() if len(x["ent"]) > 1]

    sent_order2s_id = [s_id for s_id in selected_sentences.keys()]

    rel_s_ids_order = extract_path_sent_follow_order(item)

    h_sent_ids = [m["sent_id"] for m in item["entity"][item["pos"][0]["h"]]]
    t_sent_ids = [m["sent_id"] for m in item["entity"][item["pos"][0]["t"]]]
    common_pos_ids = set(h_sent_ids) & set(t_sent_ids)
    assert len(common_pos_ids) == len(item["pos"])
    for pos_idx, pos_candi in enumerate(item["pos"]):
        for s_id in common_pos_ids:
            if pos_candi["sent"] == item["all_sentences"][s_id]:
                pos_candi["s_id"] = s_id
                common_pos_ids.remove(s_id)
                break
    for pos_candi in item["pos"]:
        assert "s_id" in pos_candi

    for pos_idx, pos_candi in enumerate(item["pos"]):

        unique_id = get_sent_unique_id(item, pos_candi["s_id"])
        if unique_id not in _sent2rel:
            continue

        pos_rel = _sent2rel[unique_id]
        neg_candidates = sample_neg_sent_w_rel(pos_rel, pos_candi["h"], pos_candi["t"])

        neg_res = []
        for neg_sent in neg_candidates:
            neg_h_id = neg_sent["h"]
            neg_t_id = neg_sent["t"]
            neg_res.append(replace_w_rel(neg_sent, pos_candi, rep_pairs=None))
            neg_res[-1]["h"] = neg_h_id
            neg_res[-1]["t"] = neg_t_id
            if len(neg_res) == max_neg_num:
                break

        path_ent_str_ls = []
        for s_id in rel_s_ids_order:
            path_ent_str_ls.append(extract_replaced_path_entity(selected_sentences[s_id]))
        path_ent_str_ls.append(extract_replaced_path_entity(pos_candi))

        aug_pos = sample_pos_sent_w_rel(pos_rel, pos_candi["h"], pos_candi["t"])
        if aug_pos is not None:
            aug_h_id = aug_pos["h"]
            aug_t_id = aug_pos["t"]
            aug_pos = replace_w_rel(aug_pos, pos_candi, rep_pairs=None)
            aug_pos["h"] = aug_h_id
            aug_pos["t"] = aug_t_id

        _r = random.random()
        if len(neg_res) == max_neg_num and (not remove_deduct and _r < deduct_ratio):
            examples.append({
                "context": add_noise_sentence(num=noise_sent_num, item=item, orig_sentences=orig_sentences, rep_pairs=None, shuffle=True),
                "negative": neg_res,
                "positive": annotate_entity(pos_candi) if aug_pos is None else aug_pos,
                "orig_id": item["id"],
                "path_s_ids": sent_order2s_id,
                "path_s_ids_order": rel_s_ids_order,
                "pos_id": pos_idx,
                "ent_mentions": path_ent_str_ls,
            })

        # ============= context replaced-based examples ==================== #
        if len(path_sent2rep) == 0:
            continue

        if len(sent_w_rel_candidates) == 0:
            continue

        tgt_ctx_sent_id, tgt_ctx_sent = random.choice(sent_w_rel_candidates)

        unique_id = get_sent_unique_id(item, tgt_ctx_sent_id)
        assert unique_id in _sent2rel

        pos_rel = _sent2rel[unique_id]
        neg_candidates = sample_neg_sent_w_rel(pos_rel, tgt_ctx_sent["h"], tgt_ctx_sent["t"])

        neg_ctx_sent = []
        for neg_sent in neg_candidates:
            neg_h_id = neg_sent["h"]
            neg_t_id = neg_sent["t"]
            neg_sent_res = replace_w_rel(neg_sent, tgt_ctx_sent, rep_pairs=None)
            neg_sent_res["h"] = neg_h_id
            neg_sent_res["t"] = neg_t_id
            neg_ctx_sent.append((tgt_ctx_sent_id, neg_sent_res))
            if len(neg_ctx_sent) == max_neg_num:
                break

        if len(neg_ctx_sent) < max_neg_num:
            continue

        assert len(neg_ctx_sent) == max_neg_num

        aug_ctx_sent = sample_pos_sent_w_rel(pos_rel, tgt_ctx_sent["h"], tgt_ctx_sent["t"])
        assert aug_ctx_sent is not None
        aug_h_id = aug_ctx_sent["h"]
        aug_t_id = aug_ctx_sent["t"]
        aug_ctx_sent = replace_w_rel(aug_ctx_sent, tgt_ctx_sent, rep_pairs=None)
        aug_ctx_sent["h"] = aug_h_id
        aug_ctx_sent["t"] = aug_t_id

        context_sentences = {s_id: annotate_entity(s) for s_id, s in selected_sentences.items()}
        context_sentences[tgt_ctx_sent_id] = aug_ctx_sent
        context_sentences = list(context_sentences.values())

        _r = random.random()
        if not remove_context and _r < context_ratio:
            negative_context = [
                rep_context_sent(selected_sentences, _neg_sent[0], _neg_sent[1]) for _neg_sent in neg_ctx_sent
            ]
            negative_noise_context = [
                add_noise_sentence(noise_sent_num, item, neg_ctx, rep_pairs=None, shuffle=True) for neg_ctx in negative_context
            ]

            context_examples.append({
                "context": add_noise_sentence(num=noise_sent_num, item=item, orig_sentences=context_sentences, rep_pairs=None,
                                              shuffle=True),
                "condition": annotate_entity(pos_candi),
                "negative_context": negative_noise_context,
                "orig_id": item["id"],
                "path_s_ids": sent_order2s_id,
                "path_s_ids_order": rel_s_ids_order,
                "pos_id": pos_idx,
                "ent_mentions": path_ent_str_ls,
            })

    # Augment the context
    # 1. ~~Choose the head entity or the tail entity as the target entity.~~
    #    Randomly sample some entities in the path to be replaced.
    # 2. Randomly sample other entities from the entity pool.
    # 3. Replace the target entity in the context with the sampled entity.
    # 4. Replace the target entity in negative samples with the sampled entity.

    # Gather the entity ids in the meta-path for sampling.
    path_ent_ids = set([p_ent_id for p_ent_id, p_sent_id in item["path"]])
    h_t_ent_ids = [item["pos"][0]["h"], item["pos"][0]["t"]]
    for x in h_t_ent_ids:
        assert x in path_ent_ids
        path_ent_ids.remove(x)
    if "relation_connect_ent" in item:
        for x in item["relation_connect_ent"]:
            if x in h_t_ent_ids:
                continue
            path_ent_ids.remove(x)

    _h_str = get_ent_str(item["pos"][0], h_t_ent_ids[0])
    _t_str = get_ent_str(item["pos"][0], h_t_ent_ids[1])

    for _ in range(aug_num):  # Repeat for augmentation

        for pos_idx, pos_candi in enumerate(item["pos"]):

            unique_id = get_sent_unique_id(item, pos_candi["s_id"])
            if unique_id not in _sent2rel:
                continue

            # Sample the amount of entities to be replaced from the geometric distribution.
            if min_rep_num >= len(path_ent_ids) + len(h_t_ent_ids):
                _sampled_ent_num = len(path_ent_ids) + len(h_t_ent_ids)
            else:
                if _geometric_dist is not None:
                    _sampled_ent_num = int(_geometric_dist.sample().item()) + min_rep_num
                else:
                    _sampled_ent_num = min_rep_num
                cnt = 0
                while _sampled_ent_num >= (len(path_ent_ids) + len(h_t_ent_ids)):
                    cnt += 1
                    _sampled_ent_num = int(_geometric_dist.sample().item()) + min_rep_num
                    if cnt > 1000:
                        logger.warning("Wrong here.")
                        raise RuntimeError()
                assert min_rep_num <= _sampled_ent_num < (len(path_ent_ids) + len(h_t_ent_ids))

            # Make sure the head/tail entity in the entities to be replaced.
            if _sampled_ent_num <= 2:
                sampled_ent_ids = random.sample(h_t_ent_ids, _sampled_ent_num)
            else:
                sampled_ent_ids = h_t_ent_ids + random.sample(list(path_ent_ids), _sampled_ent_num - 2)

            # target_ent_str = sample_entity(_entity_pool, item["id"], _sampled_ent_num)
            target_ent_str = sample_entity(_entity_pool, path_ent_ids | set(h_t_ent_ids), _sampled_ent_num)
            sampled_rep_pairs = {_ent_id: _rep_str for _ent_id, _rep_str in zip(sampled_ent_ids, target_ent_str)}

            # ======================================================================== #

            # TODO: Following snip can be removed?
            # _sampled_neg_item_key = random.choice(list(_negative_pool.keys()))
            # while _sampled_neg_item_key == item["id"] or _sampled_neg_item_key not in _all_neg_candidates:
            #     _sampled_neg_item_key = random.choice(list(_negative_pool.keys()))
            #
            # _cur_aug_rep_pairs = copy.deepcopy(sampled_rep_pairs)
            # sampled_neg_candidates = _negative_pool[_sampled_neg_item_key]
            #
            # # Replace the replacement with the head/tail entity string from the sampled negative data item.
            # for _tmp in h_t_ent_ids:
            #     if _tmp == h_t_ent_ids[0]:  # head entity
            #         _neg_head_str = get_ent_str(sampled_neg_candidates[0], sampled_neg_candidates[0]["h"])
            #         _cur_aug_rep_pairs[_tmp] = _neg_head_str  # If the head entity isn't to be replaced, add it.
            #
            #     if _tmp == h_t_ent_ids[1]:  # tail entity
            #         _neg_tail_str = get_ent_str(sampled_neg_candidates[0], sampled_neg_candidates[0]["t"])
            #         _cur_aug_rep_pairs[_tmp] = _neg_tail_str  # If the head entity isn't to be replaced, add it.
            _cur_aug_rep_pairs = copy.deepcopy(sampled_rep_pairs)

            new_sentences = {}
            for s_id, sent in selected_sentences.items():
                new_sentences[s_id] = _replace_entities_w_str(sent, _cur_aug_rep_pairs, enforce_h_t=True)
                new_sentences[s_id]["h"] = sent["h"]
                new_sentences[s_id]["t"] = sent["t"]

            path_ent_str_ls = []
            for s_id in rel_s_ids_order:
                path_ent_str_ls.append(extract_replaced_path_entity(selected_sentences[s_id], _cur_aug_rep_pairs))
            path_ent_str_ls.append(extract_replaced_path_entity(pos_candi, _cur_aug_rep_pairs))

            new_pos_candi_sent = _replace_entities_w_str(pos_candi, _cur_aug_rep_pairs, enforce_h_t=True)
            new_pos_candi_sent["h"] = pos_candi["h"]
            new_pos_candi_sent["t"] = pos_candi["t"]

            # ===============================
            pos_rel = _sent2rel[unique_id]
            neg_candidates = sample_neg_sent_w_rel(pos_rel, pos_candi["h"], pos_candi["t"])

            neg_res = []
            for neg_sent in neg_candidates:
                neg_h_id = neg_sent["h"]
                neg_t_id = neg_sent["t"]
                neg_res.append(replace_w_rel(neg_sent, pos_candi, rep_pairs=_cur_aug_rep_pairs))
                neg_res[-1]["h"] = neg_h_id
                neg_res[-1]["t"] = neg_t_id
                if len(neg_res) == max_neg_num:
                    break

            if len(neg_res) < max_neg_num:
                continue

            aug_pos = sample_pos_sent_w_rel(pos_rel, pos_candi["h"], pos_candi["t"])
            if aug_pos is not None:
                aug_h_id = aug_pos["h"]
                aug_t_id = aug_pos["t"]
                aug_pos = replace_w_rel(aug_pos, pos_candi, rep_pairs=_cur_aug_rep_pairs)
                aug_pos["h"] = aug_h_id
                aug_pos["t"] = aug_t_id

            if shuffle_context:
                new_sentences = list(new_sentences.items())
                random.shuffle(new_sentences)
                shuffled_s_ids, new_sentences = list(zip(*new_sentences))
            else:
                shuffled_s_ids, new_sentences = list(zip(*list(new_sentences.items())))

            new_context = add_noise_sentence(noise_sent_num, item, new_sentences, rep_pairs=_cur_aug_rep_pairs, shuffle=True)

            if not remove_deduct:
                examples.append({
                    "context": new_context,
                    "negative": neg_res,
                    "positive": new_pos_candi_sent if aug_pos is None else aug_pos,
                    "orig_id": item["id"],
                    "h": _cur_aug_rep_pairs[h_t_ent_ids[0]] if h_t_ent_ids[0] in _cur_aug_rep_pairs else _h_str,
                    "t": _cur_aug_rep_pairs[h_t_ent_ids[1]] if h_t_ent_ids[1] in _cur_aug_rep_pairs else _t_str,
                    "rep_ent_num": _sampled_ent_num,
                    "path_s_ids": shuffled_s_ids,
                    "path_s_ids_order": rel_s_ids_order,
                    "pos_id": pos_idx,
                    "ent_mentions": path_ent_str_ls,
                })

            # ============= context replaced-based examples ==================== #
            if len(path_sent2rep) == 0:
                continue

            if len(sent_w_rel_candidates) == 0:
                continue

            tgt_ctx_sent_id, tgt_ctx_sent = random.choice(sent_w_rel_candidates)
            unique_id = get_sent_unique_id(item, tgt_ctx_sent_id)
            assert unique_id in _sent2rel

            pos_rel = _sent2rel[unique_id]
            neg_candidates = sample_neg_sent_w_rel(pos_rel, tgt_ctx_sent["h"], tgt_ctx_sent["t"])

            neg_ctx_sent = []
            for neg_sent in neg_candidates:
                neg_h_id = neg_sent["h"]
                neg_t_id = neg_sent["t"]
                neg_sent_res = replace_w_rel(neg_sent, tgt_ctx_sent, rep_pairs=_cur_aug_rep_pairs)
                neg_sent_res["h"] = neg_h_id
                neg_sent_res["t"] = neg_t_id
                neg_ctx_sent.append((tgt_ctx_sent_id, neg_sent_res))
                if len(neg_ctx_sent) == max_neg_num:
                    break

            if len(neg_ctx_sent) < max_neg_num:
                continue

            assert len(neg_ctx_sent) == max_neg_num

            aug_ctx_sent = sample_pos_sent_w_rel(pos_rel, tgt_ctx_sent["h"], tgt_ctx_sent["t"])
            assert aug_ctx_sent is not None
            aug_h_id = aug_ctx_sent["h"]
            aug_t_id = aug_ctx_sent["t"]
            aug_ctx_sent = replace_w_rel(aug_ctx_sent, tgt_ctx_sent, rep_pairs=_cur_aug_rep_pairs)
            aug_ctx_sent["h"] = aug_h_id
            aug_ctx_sent["t"] = aug_t_id

            # To avoid shuffling, re-generate the new sentences
            new_sentences = dict()
            for s_id, sent in selected_sentences.items():
                new_sentences[s_id] = _replace_entities_w_str(sent, _cur_aug_rep_pairs, enforce_h_t=True)
                new_sentences[s_id]["h"] = sent["h"]
                new_sentences[s_id]["t"] = sent["t"]

            negative_context = []
            for _neg_sent in neg_ctx_sent:
                _new_context = copy.deepcopy(new_sentences)
                _new_context[_neg_sent[0]] = _neg_sent[1]
                _new_context_sentences = list(_new_context.values())
                if shuffle_context:
                    random.shuffle(_new_context_sentences)
                # negative_context.append(" ".join(_new_context_sentences))
                negative_context.append(add_noise_sentence(noise_sent_num, item, _new_context_sentences, rep_pairs=_cur_aug_rep_pairs,
                                                           shuffle=True))

            new_sentences[tgt_ctx_sent_id] = aug_ctx_sent

            new_sentences = list(new_sentences.items())
            if shuffle_context:
                random.shuffle(new_sentences)
                shuffled_s_ids, new_sentences = list(zip(*new_sentences))
            else:
                shuffled_s_ids, new_sentences = list(zip(*new_sentences))
            # new_context = " ".join(new_sentences)
            new_context = add_noise_sentence(noise_sent_num, item, new_sentences, rep_pairs=_cur_aug_rep_pairs, shuffle=True)

            if not remove_context:
                context_examples.append({
                    "context": new_context,
                    "condition": new_pos_candi_sent,
                    "negative_context": negative_context,
                    "orig_id": item["id"],
                    "h": _cur_aug_rep_pairs[h_t_ent_ids[0]] if h_t_ent_ids[0] in _cur_aug_rep_pairs else _h_str,
                    "t": _cur_aug_rep_pairs[h_t_ent_ids[1]] if h_t_ent_ids[1] in _cur_aug_rep_pairs else _t_str,
                    "rep_ent_num": _sampled_ent_num,
                    "path_s_ids": shuffled_s_ids,
                    "path_s_ids_order": rel_s_ids_order,
                    "pos_id": pos_idx,
                    "ent_mentions": path_ent_str_ls,
                })

    return examples, context_examples


def read_examples(file_path: str, shuffle_context: bool = False,
                  max_neg_num: int = 3, aug_num: int = 10,
                  geo_p: float = 0.5, min_rep_num: int = 1,
                  deduct_ratio: float = 1.0, context_ratio: float = 1.0, noise_sent_ratio: float = 0.5,
                  remove_deduct: bool = False, remove_context: bool = False,
                  max_neg_samples_num: int = 8, rel2sent_file: str = None, iter_num: int = 1,
                  num_workers: int = 48):
    logger.info(f"Loading raw examples from {file_path}...")
    raw_data = pickle.load(open(file_path, "rb"))
    data = raw_data["examples"]
    raw_texts = raw_data["raw_texts"]

    assert iter_num >= 1
    assert rel2sent_file
    rel2sent = pickle.load(open(rel2sent_file, "rb"))
    sent2rel = get_sent2rel(rel2sent)

    if geo_p > 0:
        geometric_dist = Geometric(torch.tensor([geo_p]))
    else:
        geometric_dist = None

    all_neg_candidates = {}
    negative_pool = {}
    all_path_sentences = {}
    _neg_cnt_1 = 0
    _neg_cnt_2 = 0
    _neg_cnt_3 = 0
    _neg_candi_cnt_2 = 0
    _neg_candi_cnt_3 = 0
    _enough = 0
    for x in tqdm(data, desc="preparing negative samples and candidates", total=len(data)):
        if x["id"] in all_neg_candidates:
            continue
        if x["id"] in negative_pool:
            continue
        # if len(x["rest_sentences"]) == 0:
        #     continue
        tmp = [y for y in x["rest_sentences"].values() if len(y["ent"]) > 1]
        if len(tmp) == 1:
            _neg_cnt_1 += 1
        elif len(tmp) == 2:
            _neg_cnt_2 += 1
        elif len(tmp) >= 3:
            _neg_cnt_3 += 1

        if len(tmp) > 0:
            all_neg_candidates[x["id"]] = tmp
            negative_pool[x["id"]] = x["pos"]
            if len(x["pos"]) == 2:
                _neg_candi_cnt_2 += 1
            elif len(x["pos"]) == 3:
                _neg_candi_cnt_3 += 1

        all_path_sentences[x["id"]] = x["pos"] + list(x["selected_sentences"].values())

        if len(tmp) + len(x["pos"]) >= max_neg_num + 1:
            _enough += 1
        # all_neg_candidates.extend([(x["id"], y) for y in x["rest_sentences"].values() if len(y["ent"]) > 1])
    logger.info(f"All negative candidates with size ``1``: {_neg_cnt_1}, size ``2``: {_neg_cnt_2} and ``3``: {_neg_cnt_3}")
    logger.info(f"Negative pools with size ``2``: {_neg_candi_cnt_2}, and size ``3``: {_neg_candi_cnt_3}.")
    logger.info(f"Enough negative samples: {_enough} / {len(data)} = {_enough * 1.0 / len(data)}")

    entity_pool = collections.defaultdict(set)  # entity unique id (wikidata) -> appeared string: List[str]
    for x in tqdm(data, desc="Preparing entity pool", total=len(data)):
        if x["id"] in entity_pool:
            continue

        for ent_id, ent_pos_ls in x["entity"].items():
            entity_pool[ent_id].update(set([
                pos2str(
                    _e_pos["pos"][0], _e_pos["pos"][1], x["all_sentences"][_e_pos["sent_id"]]
                ) for _e_pos in ent_pos_ls
            ]))

    examples = []
    context_examples = []
    for _ in range(iter_num):
        with Pool(num_workers, initializer=_initializer,
                  initargs=(entity_pool, negative_pool, all_neg_candidates, all_path_sentences, rel2sent, sent2rel,
                            geometric_dist, max_neg_samples_num)) as p:
            _annotate = partial(_process_single_item,
                                max_neg_num=max_neg_num, aug_num=aug_num, min_rep_num=min_rep_num, shuffle_context=shuffle_context,
                                deduct_ratio=deduct_ratio, context_ratio=context_ratio, noise_sent_ratio=noise_sent_ratio,
                                remove_deduct=remove_deduct, remove_context=remove_context)
            _results = list(tqdm(
                p.imap(_annotate, data, chunksize=32),
                total=len(data),
                desc="Reading examples"
            ))

        for _res, _context_res in _results:
            if _res:
                examples.extend(_res)
            if _context_res:
                context_examples.extend(_context_res)

    logger.info(f"{len(examples)} examples are loaded from {file_path}.")
    logger.info(f"{len(context_examples)} context examples are loaded from {file_path}.")

    return examples, context_examples, raw_texts


_tokenizer: PreTrainedTokenizer


def length_filter_init(tokenizer: PreTrainedTokenizer):
    global _tokenizer
    _tokenizer = tokenizer


def add_special_tokens_to_sentence(sentences: List[str]):
    context = _tokenizer.sep_token.join(sentences)  # <s> s1 </s> s2 </s> s3 </s></s> s4 </s>
    return context


def length_filter(sample, max_seq_length: int):
    sample["context"] = add_special_tokens_to_sentence(sample["context"])

    if "negative_context" in sample:
        sample["negative_context"] = [add_special_tokens_to_sentence(neg_ctx) for neg_ctx in sample["negative_context"]]
        tokens_b = _tokenizer.tokenize(sample["condition"])
        for ctx in sample["negative_context"] + [sample["context"]]:
            tokens_a = _tokenizer.tokenize(ctx)
            tokens = _tokenizer.build_inputs_with_special_tokens(tokens_a, tokens_b)
            if len(tokens) > max_seq_length:
                return False, sample
    else:
        tokens_a = _tokenizer.tokenize(sample["context"])
        for option in sample["negative"] + [sample["positive"]]:
            tokens_b = _tokenizer.tokenize(option)
            tokens = _tokenizer.build_inputs_with_special_tokens(tokens_a, tokens_b)
            if len(tokens) > max_seq_length:
                return False, sample
    return True, sample


def obtain_sentence_spans(sentences_a: List[str], sentences_b: List[str]):
    tokens = [_tokenizer.cls_token]
    sentence_spans = []

    for s_id, sent in enumerate(sentences_a):
        if s_id > 0:
            sent = " " + sent
        s = len(tokens)
        tokens.extend(_tokenizer.tokenize(sent))
        e = len(tokens)
        sentence_spans.append((s, e))

    tokens.extend(get_sep_tokens(_tokenizer))

    for s_id, sent in enumerate(sentences_b):
        if s_id > 0:
            sent = " " + sent
        s = len(tokens)
        tokens.extend(_tokenizer.tokenize(sent))
        e = len(tokens)
        sentence_spans.append((s, e))

    return sentence_spans


def length_filter_w_sentence_spans_only_pos(sample, max_seq_length: int):
    if "negative_context" in sample:
        sentence_spans = obtain_sentence_spans(sample["context"], [sample["condition"]])
    else:
        sentence_spans = obtain_sentence_spans(sample["context"], [sample["positive"]])
    sample["sentence_spans"] = sentence_spans

    sample["context"] = " ".join(sample["context"])

    if "negative_context" in sample:
        sample["negative_context"] = [" ".join(neg_ctx) for neg_ctx in sample["negative_context"]]
        tokens_b = _tokenizer.tokenize(sample["condition"])
        for ctx in sample["negative_context"] + [sample["context"]]:
            tokens_a = _tokenizer.tokenize(ctx)
            tokens = _tokenizer.build_inputs_with_special_tokens(tokens_a, tokens_b)
            if len(tokens) > max_seq_length:
                return False, sample
    else:
        tokens_a = _tokenizer.tokenize(sample["context"])
        for option in sample["negative"] + [sample["positive"]]:
            tokens_b = _tokenizer.tokenize(option)
            tokens = _tokenizer.build_inputs_with_special_tokens(tokens_a, tokens_b)
            if len(tokens) > max_seq_length:
                return False, sample
    return True, sample


def obtain_entity_spans(spans: List[str], entity_replacement: List[Dict], h_ent_id: str, t_ent_id: str, sent_id: int,
                        assertion: bool = False):
    assert h_ent_id != t_ent_id

    h_span_ids = set()
    t_span_ids = set()
    for re in entity_replacement:
        if re["id"] == h_ent_id:
            h_span_ids.add(re["span_index"])
        if re["id"] == t_ent_id:
            t_span_ids.add(re["span_index"])
        if assertion:
            assert spans[re["span_index"]] == re["tgt"], (spans, re)

    if assertion:
        assert len(h_span_ids) >= 1, (entity_replacement, h_ent_id, t_ent_id)
        assert len(t_span_ids) >= 1, (entity_replacement, h_ent_id, t_ent_id)

    tokens = []
    h_spans = []
    t_spans = []
    for span_id, span in enumerate(spans):
        if sent_id > 0 or span_id > 0:
            span = " " + span
        span_tokens = _tokenizer.tokenize(span)
        _s = len(tokens)
        _e = len(tokens) + len(span_tokens)

        if span_id in h_span_ids:
            h_spans.append((_s, _e))

        if span_id in t_span_ids:
            t_spans.append((_s, _e))

        tokens.extend(span_tokens)

    if assertion:
        assert len(h_spans), (len(spans), entity_replacement, h_ent_id, t_ent_id, spans)
        assert len(t_spans), (len(spans), entity_replacement, h_ent_id, t_ent_id, spans)

    return tokens, h_spans, t_spans


def obtain_sentence_spans_w_ent_spans(sentence_tokens_a, sentence_h_spans_a, sentence_t_spans_a,
                                      sentence_tokens_b, sentence_h_spans_b, sentence_t_spans_b):
    tokens = [_tokenizer.cls_token]
    sentence_spans = []
    sentence_h_spans = []
    sentence_t_spans = []
    offset = len(tokens)

    for s_id, (s_tokens, s_h_spans, s_t_spans) in enumerate(zip(sentence_tokens_a, sentence_h_spans_a, sentence_t_spans_a)):
        start = len(tokens)
        tokens.extend(s_tokens)
        end = len(tokens)
        sentence_spans.append((start, end))

        s_h_spans = [(h_span[0] + offset, h_span[1] + offset) for h_span in s_h_spans]
        s_t_spans = [(t_span[0] + offset, t_span[1] + offset) for t_span in s_t_spans]
        sentence_h_spans.append(s_h_spans)
        sentence_t_spans.append(s_t_spans)

        offset = end

    tokens.extend(get_sep_tokens(_tokenizer))
    offset = len(tokens)

    for s_id, (s_tokens, s_h_spans, s_t_spans) in enumerate(zip(sentence_tokens_b, sentence_h_spans_b, sentence_t_spans_b)):
        start = len(tokens)
        tokens.extend(s_tokens)
        end = len(tokens)
        sentence_spans.append((start, end))

        s_h_spans = [(h_span[0] + offset, h_span[1] + offset) for h_span in s_h_spans]
        s_t_spans = [(t_span[0] + offset, t_span[1] + offset) for t_span in s_t_spans]
        sentence_h_spans.append(s_h_spans)
        sentence_t_spans.append(s_t_spans)

        offset = end

    tokens.append(_tokenizer.sep_token)

    return tokens, sentence_spans, sentence_h_spans, sentence_t_spans


def prepare_context_h_t_spans(context_samples, option_sample, assertion: bool = False):
    context_tokens, context_h_spans, context_t_spans = [], [], []
    for sent_id, sent in enumerate(context_samples):
        sent_tokens, sent_h_spans, sent_t_spans = obtain_entity_spans(sent["spans"], sent["entity_replacement"],
                                                                      sent["h"], sent["t"], sent_id=sent_id, assertion=assertion)
        context_tokens.append(sent_tokens)
        context_h_spans.append(sent_h_spans)
        context_t_spans.append(sent_t_spans)

    option_tokens, option_h_spans, option_t_spans = obtain_entity_spans(option_sample["spans"], option_sample["entity_replacement"],
                                                                        option_sample["h"], option_sample["t"], sent_id=0,
                                                                        assertion=assertion)

    tokens, sentence_spans, sentence_h_spans, sentence_t_spans = obtain_sentence_spans_w_ent_spans(context_tokens,
                                                                                                   context_h_spans,
                                                                                                   context_t_spans,
                                                                                                   [option_tokens],
                                                                                                   [option_h_spans],
                                                                                                   [option_t_spans],
                                                                                                   )
    assert len(sentence_h_spans) == len(sentence_t_spans) == len(context_samples) + 1, (len(sentence_h_spans), len(sentence_t_spans),
                                                                                        len(context_samples))
    return tokens, sentence_spans, sentence_h_spans, sentence_t_spans


def length_filter_w_sentence_h_t_spans(sample, max_seq_length: int):
    ctx_tokens = []
    ctx_sentence_spans = []
    ctx_h_spans = []
    ctx_t_spans = []
    if "negative_context" in sample:
        tokens, sentence_spans, sentence_h_spans, sentence_t_spans = prepare_context_h_t_spans(sample["context"], sample["condition"],
                                                                                               assertion=True)
        ctx_tokens.append(tokens)
        ctx_sentence_spans.append(sentence_spans)
        ctx_h_spans.append(sentence_h_spans)
        ctx_t_spans.append(sentence_t_spans)

        # FIXME: We temporarily remove the entity span annotation in negative candidates the above processing workflow doesn't
        #   correctly annotate the position of the replaced head-tail entities.
        for neg_ctx in sample["negative_context"]:
            tokens, sentence_spans, sentence_h_spans, sentence_t_spans = prepare_context_h_t_spans(neg_ctx, sample["condition"])
            ctx_tokens.append(tokens)
            # ctx_sentence_spans.append(sentence_spans)
            # ctx_h_spans.append(sentence_h_spans)
            # ctx_t_spans.append(sentence_t_spans)

    else:
        tokens, sentence_spans, sentence_h_spans, sentence_t_spans = prepare_context_h_t_spans(sample["context"], sample["positive"],
                                                                                               assertion=True)
        ctx_tokens.append(tokens)
        ctx_sentence_spans.append(sentence_spans)
        ctx_h_spans.append(sentence_h_spans)
        ctx_t_spans.append(sentence_t_spans)

        for option in sample["negative"]:
            tokens, sentence_spans, sentence_h_spans, sentence_t_spans = prepare_context_h_t_spans(sample["context"], option)
            ctx_tokens.append(tokens)
            # ctx_sentence_spans.append(sentence_spans)
            # ctx_h_spans.append(sentence_h_spans)
            # ctx_t_spans.append(sentence_t_spans)

    for per_tokens in ctx_tokens:
        if len(per_tokens) > max_seq_length:
            return False, None

    for op_id, op_tokens in enumerate(ctx_tokens):
        if op_id == 0:
            continue
        op_string = _tokenizer.convert_tokens_to_string(op_tokens)
        ctx_tokens[op_id] = op_string

    assert len(ctx_tokens) == 4

    sample["tokens"] = ctx_tokens
    sample["sentence_spans"] = ctx_sentence_spans
    sample["h_spans"] = ctx_h_spans
    sample["t_spans"] = ctx_t_spans
    sample.pop("context")
    if "negative_context" in sample:
        sample.pop("negative_context")
        sample.pop("condition")
    else:
        sample.pop("positive")
        sample.pop("negative")

    return True, sample


def get_para_id(example):
    exp_id = example["orig_id"]
    para_id, file_id, sample_id = exp_id.split("_")
    return f"{para_id}_{file_id}"


def _init_(_exp_info, _para_id2exp_id):
    global __exp_info__
    global __para_id2exp_id__

    __exp_info__ = _exp_info
    __para_id2exp_id__ = _para_id2exp_id


def search_neg_pairs(_exp_id):
    _para_id = __exp_info__[_exp_id]["para_id"]
    _path = __exp_info__[_exp_id]["path"]
    neg_ids = []
    hard_neg_ids = []
    for tgt_exp_id in __para_id2exp_id__[_para_id]:
        if tgt_exp_id == _exp_id:
            continue

        # negative samples should be generated from different original examples.
        if __exp_info__[_exp_id]["orig_id"] == __exp_info__[tgt_exp_id]["orig_id"]:
            continue

        if __exp_info__[tgt_exp_id]["path"] == _path:
            hard_neg_ids.append(tgt_exp_id)
        else:
            neg_ids.append(tgt_exp_id)

    return _exp_id, neg_ids, hard_neg_ids


def construct_structure_neg_pairs(examples, num_workers: int = 64):
    para_id2exp_id = collections.defaultdict(list)
    exp_id_info = {}
    _cf_num = 0
    for exp_id, exp in tqdm(enumerate(examples), total=len(examples)):
        para_id = get_para_id(exp)
        para_id2exp_id[para_id].append(exp_id)

        exp_id_info[exp_id] = {}
        exp_id_info[exp_id]["para_id"] = para_id
        exp_id_info[exp_id]["orig_id"] = exp["orig_id"]

        path_s_ids = "$".join(list(map(str, sorted(map(int, exp["path_s_ids"])))))
        exp_id_info[exp_id]["path"] = path_s_ids

        exp_id_info[exp_id]["pos_id"] = exp["pos_id"]

        if "h" in exp:
            exp_id_info[exp_id]["if_aug"] = 1
            _cf_num += 1
        else:
            exp_id_info[exp_id]["if_aug"] = 0

    logger.info(f"Augmented data: {_cf_num}")

    with Pool(num_workers, initializer=_init_, initargs=(exp_id_info, para_id2exp_id)) as p:
        _results = list(tqdm(
            p.imap(search_neg_pairs, list(range(len(examples))), chunksize=32),
            total=len(examples),
            desc="constructing structure pairs..."
        ))

    neg_num = 0
    hard_neg_num = 0

    for _res in _results:
        examples[_res[0]]["neg_ids"] = _res[1]
        examples[_res[0]]["hard_neg_ids"] = _res[2]
        neg_num += len(_res[1])
        hard_neg_num += len(_res[2])

    logger.info(f"Negative candidates num: {neg_num} / {len(examples)} = {neg_num / len(examples)}")
    logger.info(f"Hard negative candidates num: {hard_neg_num} / {len(examples)} = {hard_neg_num / len(examples)}")

    # construct positive pairs
    orig_id2exp = collections.defaultdict(list)
    for exp_id, exp in enumerate(examples):
        orig_id2exp[exp["orig_id"]].append(exp_id)

    group_pos_pairs = collections.defaultdict(list)
    g_cnt = 0
    p_cnt = 0
    for orig_id, group_exp_ids in tqdm(orig_id2exp.items(), total=len(orig_id2exp)):
        for i, exp_id1 in enumerate(group_exp_ids):
            if len(examples[exp_id1]["neg_ids"]) + len(examples[exp_id1]["hard_neg_ids"]) == 0:
                break
            for exp_id2 in group_exp_ids[(i + 1):]:
                # if both come from original data, skip it, because no augmentation is performed.
                if exp_id_info[exp_id1]["if_aug"] == 0 and exp_id_info[exp_id2]["if_aug"] == 0:
                    continue
                # one augmented and one original is ok.
                if exp_id_info[exp_id1]["if_aug"] == 1 and exp_id_info[exp_id2]["if_aug"] == 1:
                    # if `pos_id` from two are equal, the two are using the same entity replacement, leading to same augmentation. Skip it.
                    if exp_id_info[exp_id1]["pos_id"] == exp_id_info[exp_id2]["pos_id"]:
                        continue
                group_pos_pairs[orig_id].append((exp_id1, exp_id2))
                p_cnt += 1
        if len(group_pos_pairs[orig_id]):
            g_cnt += 1

    logger.info(f"Group pair amount: {g_cnt}")
    logger.info(f"Pair amount: {p_cnt}")

    return examples, group_pos_pairs


def get_group_pos_pairs_keeping_original(examples):
    orig_id2exp = collections.defaultdict(list)
    for exp_id, exp in enumerate(examples):
        orig_id2exp[exp["orig_id"]].append(exp_id)

    group_pos_pairs = collections.defaultdict(list)
    g_cnt = 0
    p_cnt = 0
    for orig_id, group_exp_ids in tqdm(orig_id2exp.items(), total=len(orig_id2exp)):
        for i, exp_id1 in enumerate(group_exp_ids):
            if len(examples[exp_id1]["neg_ids"]) + len(examples[exp_id1]["hard_neg_ids"]) == 0:
                break
            for exp_id2 in group_exp_ids[(i + 1):]:
                group_pos_pairs[orig_id].append((exp_id1, exp_id2))
                p_cnt += 1
                assert sorted(examples[exp_id1]["path_s_ids"]) == sorted(examples[exp_id2]["path_s_ids"]), (
                    examples[exp_id1]["path_s_ids"], examples[exp_id2]["path_s_ids"], orig_id,
                    [examples[exp_id]["path_s_ids"] for exp_id in group_exp_ids]
                )
        if len(group_pos_pairs[orig_id]):
            g_cnt += 1

    logger.info(f"Group pair amount: {g_cnt}")
    logger.info(f"Pair amount: {p_cnt}")

    return group_pos_pairs


def convert_examples_into_features(file_path: str, tokenizer: PreTrainedTokenizer, id2rel_path_decode_id_file: str, rel_vocab: str,
                                   shuffle_context: bool = False, max_neg_num: int = 3, aug_num: int = 10,
                                   max_seq_length: int = 512, geo_p: float = 0.5, min_rep_num: int = 1,
                                   deduct_ratio: float = 1.0, context_ratio: float = 1.0, noise_sent_ratio: float = 0.5,
                                   remove_deduct: bool = False, remove_context: bool = False,
                                   max_neg_samples_num: int = 8, num_workers=48,
                                   code_drop_threshold: int = 0, rel2sent_file: str = None,
                                   add_cf_pair_data: bool = True):
    tokenizer_name = tokenizer.__class__.__name__
    tokenizer_name = tokenizer_name.replace('TokenizerFast', '')
    tokenizer_name = tokenizer_name.replace('Tokenizer', '').lower()

    file_suffix_1 = f"{tokenizer_name}_{shuffle_context}_{max_neg_num}_{aug_num}_" \
                    f"{max_seq_length}_{geo_p}_{min_rep_num}_" \
                    f"{deduct_ratio}_{context_ratio}_{noise_sent_ratio}_{max_neg_samples_num}_" \
                    f"{'' if not remove_context else 'no-ctx-ex_'}{'' if not remove_deduct else 'no-duc-ex_'}path_v9.1.2"
    cached_file_path_1 = f"{file_path}_{file_suffix_1}"
    if os.path.exists(cached_file_path_1):
        logger.info(f"Loading cached file from {cached_file_path_1}")
        examples_1, _ = torch.load(cached_file_path_1)
    else:
        examples_1 = []

    file_suffix_2 = file_suffix_1 + "_rel"
    cached_file_path_2 = f"{file_path}_{file_suffix_2}"
    if os.path.exists(cached_file_path_2):
        logger.info(f"Loading cached file from {cached_file_path_2}")
        all_examples, raw_texts = torch.load(cached_file_path_2)
        all_examples = all_examples + examples_1

        dataset = WikiPathDatasetRelGenerateV1(all_examples, raw_texts, id2rel_path_decode_id_file, rel_vocab, code_drop_threshold)

        return dataset

    examples, context_examples, raw_texts = read_examples(file_path, shuffle_context=shuffle_context, max_neg_num=max_neg_num,
                                                          aug_num=aug_num, geo_p=geo_p, min_rep_num=min_rep_num,
                                                          deduct_ratio=deduct_ratio, context_ratio=context_ratio,
                                                          noise_sent_ratio=noise_sent_ratio,
                                                          remove_deduct=remove_deduct, remove_context=remove_context,
                                                          max_neg_samples_num=max_neg_samples_num,
                                                          rel2sent_file=rel2sent_file,
                                                          num_workers=num_workers)

    all_examples = examples + context_examples
    with Pool(num_workers, initializer=length_filter_init, initargs=(tokenizer,)) as p:
        _annotate = partial(length_filter_w_sentence_h_t_spans, max_seq_length=max_seq_length)
        results = list(tqdm(
            p.imap(_annotate, all_examples, chunksize=32),
            total=len(all_examples),
            desc="filtering examples by length",
        ))
    all_examples = []
    for flag, exp in results:
        if flag:
            all_examples.append(exp)

    # Save
    logger.info(f"Saving processed features into {cached_file_path_2}.")
    torch.save((all_examples, raw_texts), cached_file_path_2)

    all_examples = all_examples + examples_1

    return WikiPathDatasetRelGenerateV1(all_examples, raw_texts, id2rel_path_decode_id_file, rel_vocab, code_drop_threshold)


def convert_examples_into_features_iter(file_path: str, tokenizer: PreTrainedTokenizer, id2rel_path_decode_id_file: str, rel_vocab: str,
                                        shuffle_context: bool = False, max_neg_num: int = 3, aug_num: int = 10,
                                        max_seq_length: int = 512, geo_p: float = 0.5, min_rep_num: int = 1,
                                        deduct_ratio: float = 1.0, context_ratio: float = 1.0, noise_sent_ratio: float = 0.5,
                                        remove_deduct: bool = False, remove_context: bool = False,
                                        max_neg_samples_num: int = 8, num_workers=48,
                                        code_drop_threshold: int = 0, rel2sent_file: str = None, iter_num: int = 5,
                                        add_cf_pair_data: bool = True):
    tokenizer_name = tokenizer.__class__.__name__
    tokenizer_name = tokenizer_name.replace('TokenizerFast', '')
    tokenizer_name = tokenizer_name.replace('Tokenizer', '').lower()

    file_suffix = f"{tokenizer_name}_{shuffle_context}_{max_neg_num}_{aug_num}_{iter_num}_" \
                  f"{max_seq_length}_{geo_p}_{min_rep_num}_" \
                  f"{deduct_ratio}_{context_ratio}_{noise_sent_ratio}_{max_neg_samples_num}_" \
                  f"{'' if not remove_context else 'no-ctx-ex_'}{'' if not remove_deduct else 'no-duc-ex_'}path_v9.1.2_rel"

    cached_file_path = f"{file_path}_{file_suffix}"
    if os.path.exists(cached_file_path):
        logger.info(f"Loading cached file from {cached_file_path}")
        all_examples, raw_texts = torch.load(cached_file_path)

        dataset = WikiPathDatasetRelGenerateV1(all_examples, raw_texts, id2rel_path_decode_id_file, rel_vocab, code_drop_threshold)

        return dataset

    examples, context_examples, raw_texts = read_examples(file_path, shuffle_context=shuffle_context, max_neg_num=max_neg_num,
                                                          aug_num=aug_num, geo_p=geo_p, min_rep_num=min_rep_num,
                                                          deduct_ratio=deduct_ratio, context_ratio=context_ratio,
                                                          noise_sent_ratio=noise_sent_ratio,
                                                          remove_deduct=remove_deduct, remove_context=remove_context,
                                                          max_neg_samples_num=max_neg_samples_num,
                                                          rel2sent_file=rel2sent_file, iter_num=iter_num,
                                                          num_workers=num_workers)

    all_examples = examples + context_examples
    with Pool(num_workers, initializer=length_filter_init, initargs=(tokenizer,)) as p:
        _annotate = partial(length_filter_w_sentence_h_t_spans, max_seq_length=max_seq_length)
        results = list(tqdm(
            p.imap(_annotate, all_examples, chunksize=32),
            total=len(all_examples),
            desc="filtering examples by length",
        ))
    all_examples = []
    for flag, exp in results:
        if flag:
            all_examples.append(exp)

    # Save
    logger.info(f"Saving processed features into {cached_file_path}.")
    torch.save((all_examples, raw_texts), cached_file_path)

    return WikiPathDatasetRelGenerateV1(all_examples, raw_texts, id2rel_path_decode_id_file, rel_vocab, code_drop_threshold)


# def convert_examples_into_features_pair(file_path: str, tokenizer: PreTrainedTokenizer,
#                                         shuffle_context: bool = False, max_neg_num: int = 3, aug_num: int = 10,
#                                         max_seq_length: int = 512, geo_p: float = 0.5, min_rep_num: int = 1,
#                                         deduct_ratio: float = 1.0, context_ratio: float = 1.0, noise_sent_ratio: float = 0.5,
#                                         remove_deduct: bool = False, remove_context: bool = False,
#                                         max_neg_samples_num: int = 8, num_workers=48,
#                                         add_cf_pair_data: bool = True,
#                                         add_hard_negative: bool = True, add_negative: bool = True, pair_max_neg_num: int = 2):
#     tokenizer_name = tokenizer.__class__.__name__
#     tokenizer_name = tokenizer_name.replace('TokenizerFast', '')
#     tokenizer_name = tokenizer_name.replace('Tokenizer', '').lower()
#
#     file_suffix = f"{tokenizer_name}_{shuffle_context}_{max_neg_num}_{aug_num}_" \
#                   f"{max_seq_length}_{geo_p}_{min_rep_num}_" \
#                   f"{deduct_ratio}_{context_ratio}_{noise_sent_ratio}_{max_neg_samples_num}_" \
#                   f"{'' if not remove_context else 'no-ctx-ex_'}{'' if not remove_deduct else 'no-duc-ex_'}path_v9.1.2_pair"
#
#     cached_file_path = f"{file_path}_{file_suffix}"
#     if os.path.exists(cached_file_path):
#         logger.info(f"Loading cached file from {cached_file_path}")
#         all_examples, group_pos_pairs, raw_texts = torch.load(cached_file_path)
#
#         dataset = WikiPathDatasetV6wPair(all_examples, group_pos_pairs, raw_texts,
#                                          add_hard_negative=add_hard_negative, max_neg_num=pair_max_neg_num,
#                                          add_negative=add_negative)
#
#         return dataset
#
#     examples, context_examples, raw_texts = read_examples(file_path, shuffle_context=shuffle_context, max_neg_num=max_neg_num,
#                                                           aug_num=aug_num, geo_p=geo_p, min_rep_num=min_rep_num,
#                                                           deduct_ratio=deduct_ratio, context_ratio=context_ratio,
#                                                           noise_sent_ratio=noise_sent_ratio,
#                                                           remove_deduct=remove_deduct, remove_context=remove_context,
#                                                           max_neg_samples_num=max_neg_samples_num,
#                                                           num_workers=num_workers)
#
#     all_examples = examples + context_examples
#     with Pool(num_workers, initializer=length_filter_init, initargs=(tokenizer,)) as p:
#         _annotate = partial(length_filter_w_sentence_h_t_spans, max_seq_length=max_seq_length)
#         results = list(tqdm(
#             p.imap(_annotate, all_examples, chunksize=32),
#             total=len(all_examples),
#             desc="filtering examples by length",
#         ))
#     all_examples = []
#     for flag, exp in results:
#         if flag:
#             all_examples.append(exp)
#
#     all_examples, group_pos_pairs = construct_structure_neg_pairs(all_examples)
#
#     # Save
#     logger.info(f"Saving processed features into {cached_file_path}.")
#     torch.save((all_examples, group_pos_pairs, raw_texts), cached_file_path)
#
#     return WikiPathDatasetV6wPair(all_examples, group_pos_pairs, raw_texts,
#                                   add_hard_negative=add_hard_negative, max_neg_num=pair_max_neg_num, add_negative=add_negative)
#
#
# def convert_examples_into_features_pair_all(file_path: str, tokenizer: PreTrainedTokenizer,
#                                             shuffle_context: bool = False, max_neg_num: int = 3, aug_num: int = 10,
#                                             max_seq_length: int = 512, geo_p: float = 0.5, min_rep_num: int = 1,
#                                             deduct_ratio: float = 1.0, context_ratio: float = 1.0, noise_sent_ratio: float = 0.5,
#                                             remove_deduct: bool = False, remove_context: bool = False,
#                                             max_neg_samples_num: int = 8, num_workers=48,
#                                             add_cf_pair_data: bool = True,
#                                             add_hard_negative: bool = True, add_negative: bool = True, pair_max_neg_num: int = 2):
#     tokenizer_name = tokenizer.__class__.__name__
#     tokenizer_name = tokenizer_name.replace('TokenizerFast', '')
#     tokenizer_name = tokenizer_name.replace('Tokenizer', '').lower()
#
#     file_suffix = f"{tokenizer_name}_{shuffle_context}_{max_neg_num}_{aug_num}_" \
#                   f"{max_seq_length}_{geo_p}_{min_rep_num}_" \
#                   f"{deduct_ratio}_{context_ratio}_{noise_sent_ratio}_{max_neg_samples_num}_" \
#                   f"{'' if not remove_context else 'no-ctx-ex_'}{'' if not remove_deduct else 'no-duc-ex_'}path_v9.1.2_pair"
#
#     cached_file_path = f"{file_path}_{file_suffix}"
#     if os.path.exists(cached_file_path):
#         logger.info(f"Loading cached file from {cached_file_path}")
#         all_examples, _, raw_texts = torch.load(cached_file_path)
#         group_pos_pairs = get_group_pos_pairs_keeping_original(all_examples)
#
#         dataset = WikiPathDatasetV6wPair(all_examples, group_pos_pairs, raw_texts,
#                                          add_hard_negative=add_hard_negative, max_neg_num=pair_max_neg_num,
#                                          add_negative=add_negative)
#
#         return dataset
#
#     examples, context_examples, raw_texts = read_examples(file_path, shuffle_context=shuffle_context, max_neg_num=max_neg_num,
#                                                           aug_num=aug_num, geo_p=geo_p, min_rep_num=min_rep_num,
#                                                           deduct_ratio=deduct_ratio, context_ratio=context_ratio,
#                                                           noise_sent_ratio=noise_sent_ratio,
#                                                           remove_deduct=remove_deduct, remove_context=remove_context,
#                                                           max_neg_samples_num=max_neg_samples_num,
#                                                           num_workers=num_workers)
#
#     all_examples = examples + context_examples
#     with Pool(num_workers, initializer=length_filter_init, initargs=(tokenizer,)) as p:
#         _annotate = partial(length_filter_w_sentence_h_t_spans, max_seq_length=max_seq_length)
#         results = list(tqdm(
#             p.imap(_annotate, all_examples, chunksize=32),
#             total=len(all_examples),
#             desc="filtering examples by length",
#         ))
#     all_examples = []
#     for flag, exp in results:
#         if flag:
#             all_examples.append(exp)
#
#     all_examples, _ = construct_structure_neg_pairs(all_examples)
#     group_pos_pairs = get_group_pos_pairs_keeping_original(all_examples)
#
#     # Save
#     logger.info(f"Saving processed features into {cached_file_path}.")
#     torch.save((all_examples, group_pos_pairs, raw_texts), cached_file_path)
#
#     return WikiPathDatasetV6wPair(all_examples, group_pos_pairs, raw_texts,
#                                   add_hard_negative=add_hard_negative, max_neg_num=pair_max_neg_num, add_negative=add_negative)


# def convert_examples_into_features_v3_ent_local_pair(file_path: str, tokenizer: PreTrainedTokenizer, id2rel_path_decode_id_file: str,
#                                                      shuffle_context: bool = False, max_neg_num: int = 3, aug_num: int = 10,
#                                                      max_seq_length: int = 512, geo_p: float = 0.5, min_rep_num: int = 1,
#                                                      deduct_ratio: float = 1.0, context_ratio: float = 1.0, noise_sent_ratio: float = 0.5,
#                                                      remove_deduct: bool = False, remove_context: bool = False,
#                                                      max_neg_samples_num: int = 8, num_workers=48,
#                                                      rel_path_set_file: str = None,
#                                                      code_drop_threshold: int = 0,
#                                                      ):
#     tokenizer_name = tokenizer.__class__.__name__
#     tokenizer_name = tokenizer_name.replace('TokenizerFast', '')
#     tokenizer_name = tokenizer_name.replace('Tokenizer', '').lower()
#
#     file_suffix = f"{tokenizer_name}_{shuffle_context}_{max_neg_num}_{aug_num}_" \
#                   f"{max_seq_length}_{geo_p}_{min_rep_num}_" \
#                   f"{deduct_ratio}_{context_ratio}_{noise_sent_ratio}_{max_neg_samples_num}_" \
#                   f"{'' if not remove_context else 'no-ctx-ex_'}{'' if not remove_deduct else 'no-duc-ex_'}path_v9.1_split_vqvae_v3_ent"
#
#     cached_file_path = f"{file_path}_{file_suffix}"
#     if os.path.exists(cached_file_path):
#         logger.info(f"Loading cached file from {cached_file_path}")
#         all_examples, raw_texts = torch.load(cached_file_path)
#         dataset = WikiPathDatasetPatternPairKMeansLocal(all_examples, raw_texts, rel_path_set_file, id2rel_path_decode_id_file,
#                                                         code_drop_threshold)
#         return dataset
#
#     examples, context_examples, raw_texts = read_examples(file_path, shuffle_context=shuffle_context, max_neg_num=max_neg_num,
#                                                           aug_num=aug_num, geo_p=geo_p, min_rep_num=min_rep_num,
#                                                           deduct_ratio=deduct_ratio, context_ratio=context_ratio,
#                                                           noise_sent_ratio=noise_sent_ratio,
#                                                           remove_deduct=remove_deduct, remove_context=remove_context,
#                                                           max_neg_samples_num=max_neg_samples_num,
#                                                           num_workers=num_workers)
#     with Pool(num_workers, initializer=length_filter_init, initargs=(tokenizer,)) as p:
#         _annotate = partial(length_filter_w_sentence_spans_only_pos, max_seq_length=max_seq_length)
#         results = list(tqdm(
#             p.imap(_annotate, examples + context_examples, chunksize=32),
#             total=(len(examples) + len(context_examples)),
#             desc="filtering examples by length",
#         ))
#     all_examples = []
#     for flag, exp in results:
#         if flag:
#             all_examples.append(exp)
#
#     # Save
#     logger.info(f"Saving processed features into {cached_file_path}.")
#     torch.save((all_examples, raw_texts), cached_file_path)
#
#     return WikiPathDatasetPatternPairKMeansLocal(all_examples, raw_texts, rel_path_set_file, id2rel_path_decode_id_file,
#                                                  code_drop_threshold)


def _quick_loading(file_path: str, pattern_pair_file: str, **kwargs):
    logger.info(f"Quickly loading cached file from {file_path}")
    all_examples, raw_texts = torch.load(file_path)
    dataset = WikiPathDatasetV6wPatternPair(all_examples, raw_texts, pattern_pair_file)
    return dataset
