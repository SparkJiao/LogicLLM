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

from data.collators.wiki import WikiPathDatasetV6wPatternPair, WikiPathDatasetRelGenerateV1
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
"""

logger = get_child_logger("Wiki.Entity.Path.V9.1.2")

_entity_pool: Dict
_negative_pool: Dict
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
    # context_sentences = {s_id: " ".join(s["sent"]) for s_id, s in selected_sentences.items()}
    # context_sentences[tgt_sent_id] = rep_tgt_sent
    #
    # if return_sentences:
    #     return list(context_sentences.values())
    #
    # return " ".join(list(context_sentences.values()))
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
                 geometric_dist: torch.distributions.Distribution,
                 max_neg_samples_num: int):
    global _entity_pool
    global _negative_pool
    global _all_neg_candidates
    global _all_path_sentences
    global _geometric_dist
    global MAX_NEG_SAMPLE_NUM

    _entity_pool = entity_pool
    _negative_pool = negative_pool
    _all_neg_candidates = all_neg_candidates
    _all_path_sentences = all_path_sentences
    _geometric_dist = geometric_dist
    MAX_NEG_SAMPLE_NUM = max_neg_samples_num


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
    # context = " ".join([" ".join(s["sent"]) for s_id, s in selected_sentences.items()])
    # orig_sentences = [" ".join(s["sent"]) for s_id, s in selected_sentences.items()]
    orig_sentences = [annotate_entity(s) for s_id, s in selected_sentences.items()]

    path_sent2rep = [(s_id, s) for s_id, s in selected_sentences.items() if s["h"] != -1 and s["t"] != -1]

    neg_candidates = [x for x in item["rest_sentences"].values() if len(x["ent"]) > 1]

    sent_order2s_id = [s_id for s_id in selected_sentences.keys()]

    rel_s_ids_order = extract_path_sent_follow_order(item)

    for pos_idx, pos_candi in enumerate(item["pos"]):
        # Statistics
        _res_aug = 0
        _pos_aug = 0
        _sim_aug = 0

        neg_res = []

        # Other positive candidates
        mutual_samples = [candi for candi_idx, candi in enumerate(item["pos"]) if candi_idx != pos_idx]
        for neg in mutual_samples:
            # TODO: `switch`的一个问题在于，引入了本句中另外的一个实体的关系，可能出现一些side information来辅助模型判断。
            #   当然这个问题在使用随机采样的句子的时候也会出现。因为随机采样的句子也可能包含新的实体信息。
            #   有可能解决吗？只交换两个实体的位置？可能会导致换了又没换
            _rep_res = switch_replace_neg(pos_candi, neg, rep_pairs=None)
            if len(_rep_res) > 0:
                neg_res.extend(_rep_res)
        _pos_aug += len(neg_res)

        # Easy samples.
        for neg in neg_candidates:
            _rep_res = replace_neg(pos_candi, neg, rep_pairs=None)
            if _rep_res:
                neg_res.extend(_rep_res)
            # if len(neg_res) >= MAX_NEG_SAMPLE_NUM:
            #     break
        _res_aug += max(len(neg_res) - _pos_aug, 0)

        while len(neg_res) < MAX_NEG_SAMPLE_NUM:
            neg_data_item_id = random.choice(list(_all_neg_candidates.keys()))
            while neg_data_item_id == item["id"]:
                neg_data_item_id = random.choice(list(_all_neg_candidates.keys()))
            neg = random.choice(_all_neg_candidates[neg_data_item_id])

            # _rep_res = replace_neg(pos_candi, neg, rep_pairs=None, out_of_domain=False)
            _rep_res, _ = context_replace_neg(pos_candi, neg, rep_pairs=None, out_of_domain=False)
            if _rep_res:
                neg_res.extend(_rep_res)

        _sim_aug = max(len(neg_res) - _res_aug - _pos_aug, 0)

        assert len(neg_res) >= max_neg_num

        path_ent_str_ls = []
        for s_id in rel_s_ids_order:
            path_ent_str_ls.append(extract_replaced_path_entity(selected_sentences[s_id]))
        path_ent_str_ls.append(extract_replaced_path_entity(pos_candi))

        # Add head-tail entity id to negative result.
        neg_res = random.sample(neg_res, max_neg_num)
        for _neg in neg_res:
            _neg["h"] = pos_candi["h"]
            _neg["t"] = pos_candi["t"]

        _r = random.random()
        if not remove_deduct and _r < deduct_ratio:
            examples.append({
                "context": add_noise_sentence(num=noise_sent_num, item=item, orig_sentences=orig_sentences, rep_pairs=None, shuffle=True),
                # "negative": random.sample(neg_res, max_neg_num),
                "negative": neg_res,
                # "positive": " ".join(pos_candi["sent"]),
                "positive": annotate_entity(pos_candi),
                "orig_id": item["id"],
                "pos_aug_num": _pos_aug,
                "res_aug_num": _res_aug,
                "sim_aug_num": _sim_aug,
                "path_s_ids": sent_order2s_id,
                "path_s_ids_order": rel_s_ids_order,
                "pos_id": pos_idx,
                "ent_mentions": path_ent_str_ls,
            })

        # ============= context replaced-based examples ==================== #
        if len(path_sent2rep) == 0:
            continue

        neg_ctx_sent = []
        _ctx_pos_aug = 0
        _ctx_res_aug = 0
        _ctx_sim_aug = 0

        for neg in mutual_samples + neg_candidates:
            tgt_ctx_sent_id, tgt_ctx_sent = random.choice(path_sent2rep)
            _rep_res, _rep_flag = context_replace_neg(tgt_ctx_sent, neg, rep_pairs=None, out_of_domain=False)
            _rep_res = [(tgt_ctx_sent_id, _tmp_res) for _tmp_res in _rep_res]

            if _rep_res:
                _left_num = min(MAX_NEG_SAMPLE_NUM, len(_rep_res))
                neg_ctx_sent.extend(_rep_res[:_left_num])
                if _rep_flag == 0:
                    _ctx_pos_aug += _left_num
                else:
                    _ctx_res_aug += _left_num

        while len(neg_ctx_sent) < MAX_NEG_SAMPLE_NUM:
            neg_data_item_id = random.choice(list(_all_neg_candidates.keys()))
            while neg_data_item_id == item["id"]:
                neg_data_item_id = random.choice(list(_all_neg_candidates.keys()))
            neg = random.choice(_all_neg_candidates[neg_data_item_id])

            # _rep_res = replace_neg(tgt_ctx_sent, neg, rep_pairs=None, out_of_domain=True)
            tgt_ctx_sent_id, tgt_ctx_sent = random.choice(path_sent2rep)
            _rep_res, _ = context_replace_neg(tgt_ctx_sent, neg, rep_pairs=None, out_of_domain=False)
            _rep_res = [(tgt_ctx_sent_id, _tmp_res) for _tmp_res in _rep_res]
            if _rep_res:
                _left_num = MAX_NEG_SAMPLE_NUM - len(neg_ctx_sent)
                neg_ctx_sent.extend(_rep_res[:_left_num])
                _ctx_sim_aug += min(_left_num, len(_rep_res))

        assert len(neg_ctx_sent) >= max_neg_num
        neg_ctx_sent = random.sample(neg_ctx_sent, max_neg_num)

        _r = random.random()
        if not remove_context and _r < context_ratio:
            negative_context = [
                rep_context_sent(selected_sentences, _neg_sent[0], _neg_sent[1]) for _neg_sent in neg_ctx_sent
            ]
            negative_noise_context = [
                add_noise_sentence(noise_sent_num, item, neg_ctx, rep_pairs=None, shuffle=True) for neg_ctx in negative_context
            ]

            context_examples.append({
                "context": add_noise_sentence(num=noise_sent_num, item=item, orig_sentences=orig_sentences, rep_pairs=None, shuffle=True),
                # "condition": " ".join(pos_candi["sent"]),
                "condition": annotate_entity(pos_candi),
                "negative_context": negative_noise_context,
                "orig_id": item["id"],
                "pos_aug_num": _ctx_pos_aug,
                "res_aug_num": _ctx_res_aug,
                "sim_aug_num": _ctx_sim_aug,
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

            _sampled_neg_item_key = random.choice(list(_negative_pool.keys()))
            while _sampled_neg_item_key == item["id"] or _sampled_neg_item_key not in _all_neg_candidates:
                _sampled_neg_item_key = random.choice(list(_negative_pool.keys()))

            _cur_aug_rep_pairs = copy.deepcopy(sampled_rep_pairs)
            sampled_neg_candidates = _negative_pool[_sampled_neg_item_key]

            # Replace the replacement with the head/tail entity string from the sampled negative data item.
            for _tmp in h_t_ent_ids:
                if _tmp == h_t_ent_ids[0]:  # head entity
                    _neg_head_str = get_ent_str(sampled_neg_candidates[0], sampled_neg_candidates[0]["h"])
                    _cur_aug_rep_pairs[_tmp] = _neg_head_str  # If the head entity isn't to be replaced, add it.

                if _tmp == h_t_ent_ids[1]:  # tail entity
                    _neg_tail_str = get_ent_str(sampled_neg_candidates[0], sampled_neg_candidates[0]["t"])
                    _cur_aug_rep_pairs[_tmp] = _neg_tail_str  # If the head entity isn't to be replaced, add it.

            # TODO: Should other entities in ``sampled_rep_paris`` be replaced with the entity strings from the same negative sample item?

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

            # Statistics
            _res_aug = 0
            _pos_aug = 0
            _sim_aug = 0

            neg_res = []

            # Other positive candidates
            mutual_samples = [candi for candi_idx, candi in enumerate(item["pos"]) if candi_idx != pos_idx]
            for neg in mutual_samples:
                # _rep_res = switch_replace_neg(pos_candi, neg, rep_pairs=None)  # roberta-large
                _rep_res = switch_replace_neg(pos_candi, neg, rep_pairs=_cur_aug_rep_pairs)  # albert-xx-large-v2 in server 220.
                if len(_rep_res) > 0:
                    neg_res.extend(_rep_res)
            _pos_aug += len(neg_res)

            # Add negative samples from the initial ``rest_sentences``.
            for neg in neg_candidates:
                _rep_res = replace_neg(pos_candi, neg, rep_pairs=_cur_aug_rep_pairs)
                if _rep_res:
                    neg_res.extend(_rep_res)
                # if len(neg_res) >= MAX_NEG_SAMPLE_NUM:
                #     break
            _res_aug += max(len(neg_res) - _pos_aug, 0)

            # if len(neg_res) > MAX_NEG_SAMPLE_NUM:
            #     neg_res = neg_res[:MAX_NEG_SAMPLE_NUM]

            # Add simple negative samples from the positive samples of the sampled data item.
            if len(neg_res) < MAX_NEG_SAMPLE_NUM:
                for neg in sampled_neg_candidates:
                    # Add the sampled negative candidate since it contains the replaced head/tail entity already.
                    # _rep_res = " ".join(neg["sent"])
                    # neg_res.append(_rep_res)
                    _rep_res = annotate_entity(neg)
                    neg_res.append(_rep_res)
                    if len(neg_res) >= MAX_NEG_SAMPLE_NUM:
                        break

            # Add simple negative samples from the ``rest_sentences`` from the sampled data item.
            if len(neg_res) < MAX_NEG_SAMPLE_NUM:
                for neg in _all_neg_candidates[_sampled_neg_item_key]:
                    # _rep_res = replace_neg(sampled_neg_candidates[0], neg, rep_pairs=_cur_aug_rep_pairs, out_of_domain=False)
                    _rep_res, _ = context_replace_neg(sampled_neg_candidates[0], neg, rep_pairs=_cur_aug_rep_pairs, out_of_domain=False)
                    if _rep_res:
                        neg_res.extend(_rep_res)
                    if len(neg_res) >= MAX_NEG_SAMPLE_NUM:
                        break

            # if len(neg_res) > max_neg_num:
            #     neg_res = neg_res[:max_neg_num]

            # Add simple negative samples for padding
            while len(neg_res) < MAX_NEG_SAMPLE_NUM:
                neg_data_item_id = random.choice(list(_all_neg_candidates.keys()))
                while neg_data_item_id == item["id"] or neg_data_item_id == _sampled_neg_item_key:
                    neg_data_item_id = random.choice(list(_all_neg_candidates.keys()))
                neg = random.choice(_all_neg_candidates[neg_data_item_id])

                # _rep_res = replace_neg(sampled_neg_candidates[0], neg, rep_pairs=_cur_aug_rep_pairs, out_of_domain=False)
                _rep_res, _ = context_replace_neg(sampled_neg_candidates[0], neg, rep_pairs=_cur_aug_rep_pairs, out_of_domain=False)
                if _rep_res:
                    neg_res.extend(_rep_res)

            _sim_aug += max(len(neg_res) - _pos_aug - _res_aug, 0)

            # if len(neg_res) > max_neg_num:
            #     neg_res = neg_res[:max_neg_num]
            assert len(neg_res) >= MAX_NEG_SAMPLE_NUM
            neg_res = random.sample(neg_res, max_neg_num)

            for _neg in neg_res:
                if "h" in _neg:
                    # assert _neg["h"] == pos_candi["h"], (_neg["h"], pos_candi["h"])
                    # assert _neg["t"] == pos_candi["t"], (_neg["t"], pos_candi["t"])
                    continue
                _neg["h"] = pos_candi["h"]
                _neg["t"] = pos_candi["t"]

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
                    "positive": new_pos_candi_sent,
                    "orig_id": item["id"],
                    "h": _cur_aug_rep_pairs[h_t_ent_ids[0]] if h_t_ent_ids[0] in _cur_aug_rep_pairs else _h_str,
                    "t": _cur_aug_rep_pairs[h_t_ent_ids[1]] if h_t_ent_ids[1] in _cur_aug_rep_pairs else _t_str,
                    "pos_aug_num": _pos_aug,
                    "res_aug_num": _res_aug,
                    "sim_aug_num": _sim_aug,
                    "rep_ent_num": _sampled_ent_num,
                    "path_s_ids": shuffled_s_ids,
                    "path_s_ids_order": rel_s_ids_order,
                    "pos_id": pos_idx,
                    "ent_mentions": path_ent_str_ls,
                })

            # ============= context replaced-based examples ==================== #
            if len(path_sent2rep) == 0:
                continue

            neg_ctx_sent = []
            _ctx_pos_aug = 0
            _ctx_res_aug = 0
            _ctx_sim_aug = 0

            for neg in mutual_samples + neg_candidates:
                tgt_ctx_sent_id, tgt_ctx_sent = random.choice(path_sent2rep)
                _rep_res, _rep_flag = context_replace_neg(tgt_ctx_sent, neg, rep_pairs=_cur_aug_rep_pairs, out_of_domain=False)
                _rep_res = [(tgt_ctx_sent_id, _tmp_res) for _tmp_res in _rep_res]

                if _rep_res:
                    _left_num = min(MAX_NEG_SAMPLE_NUM, len(_rep_res))
                    neg_ctx_sent.extend(_rep_res[:_left_num])
                    if _rep_flag == 0:
                        _ctx_pos_aug += _left_num
                    else:
                        _ctx_res_aug += _left_num

                # if len(neg_ctx_sent) >= MAX_NEG_SAMPLE_NUM:
                #     neg_ctx_sent = neg_ctx_sent[:MAX_NEG_SAMPLE_NUM]
                #     break

            if len(neg_ctx_sent) < MAX_NEG_SAMPLE_NUM:
                for neg in sampled_neg_candidates + _all_neg_candidates[_sampled_neg_item_key]:
                    tgt_ctx_sent_id, tgt_ctx_sent = random.choice(path_sent2rep)
                    # _rep_res, _ = context_replace_neg(tgt_ctx_sent, neg, rep_pairs=None, out_of_domain=True)
                    # V9_Fix: add ``_cur_aug_rep_pairs`` although ``out_of_domain=True``
                    # to replace the head/tail entity in the context sentence.
                    _rep_res, _ = context_replace_neg(tgt_ctx_sent, neg, rep_pairs=_cur_aug_rep_pairs, out_of_domain=False)
                    _rep_res = [(tgt_ctx_sent_id, _tmp_res) for _tmp_res in _rep_res]

                    if _rep_res:
                        _left_num = min(MAX_NEG_SAMPLE_NUM, len(_rep_res))
                        neg_ctx_sent.extend(_rep_res[:_left_num])
                        _ctx_sim_aug += _left_num

                    if len(neg_ctx_sent) >= MAX_NEG_SAMPLE_NUM:
                        neg_ctx_sent = neg_ctx_sent[:MAX_NEG_SAMPLE_NUM]
                        break

            while len(neg_ctx_sent) < MAX_NEG_SAMPLE_NUM:
                neg_data_item_id = random.choice(list(_all_neg_candidates.keys()))
                while neg_data_item_id == item["id"]:
                    neg_data_item_id = random.choice(list(_all_neg_candidates.keys()))
                neg = random.choice(_all_neg_candidates[neg_data_item_id])

                tgt_ctx_sent_id, tgt_ctx_sent = random.choice(path_sent2rep)
                # _rep_res, _ = context_replace_neg(tgt_ctx_sent, neg, rep_pairs=None, out_of_domain=True)
                # V9_Fix: add ``_cur_aug_rep_pairs`` although ``out_of_domain=True``
                # to replace the head/tail entity in the context sentence.
                _rep_res, _ = context_replace_neg(tgt_ctx_sent, neg, rep_pairs=_cur_aug_rep_pairs, out_of_domain=False)
                _rep_res = [(tgt_ctx_sent_id, _tmp_res) for _tmp_res in _rep_res]

                if _rep_res:
                    _left_num = MAX_NEG_SAMPLE_NUM - len(neg_ctx_sent)
                    neg_ctx_sent.extend(_rep_res[:_left_num])
                    _ctx_sim_aug += min(_left_num, len(_rep_res))

            assert len(neg_ctx_sent) >= MAX_NEG_SAMPLE_NUM
            neg_ctx_sent = random.sample(neg_ctx_sent, max_neg_num)

            # To avoid shuffling, re-generate the new sentences
            new_sentences = dict()
            for s_id, sent in selected_sentences.items():
                new_sentences[s_id] = _replace_entities_w_str(sent, _cur_aug_rep_pairs, enforce_h_t=True)
                new_sentences[s_id]["h"] = sent["h"]
                new_sentences[s_id]["t"] = sent["t"]

            negative_context = []
            for _neg_sent in neg_ctx_sent:
                _new_context = copy.deepcopy(new_sentences)
                _neg_sent[1]["h"] = _new_context[_neg_sent[0]]["h"]
                _neg_sent[1]["t"] = _new_context[_neg_sent[0]]["t"]
                _new_context[_neg_sent[0]] = _neg_sent[1]
                _new_context_sentences = list(_new_context.values())
                if shuffle_context:
                    random.shuffle(_new_context_sentences)
                # negative_context.append(" ".join(_new_context_sentences))
                negative_context.append(add_noise_sentence(noise_sent_num, item, _new_context_sentences, rep_pairs=_cur_aug_rep_pairs,
                                                           shuffle=True))

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
                    "pos_aug_num": _ctx_pos_aug,
                    "res_aug_num": _ctx_res_aug,
                    "sim_aug_num": _ctx_sim_aug,
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
                  max_neg_samples_num: int = 8,
                  num_workers: int = 48):
    logger.info(f"Loading raw examples from {file_path}...")
    raw_data = pickle.load(open(file_path, "rb"))
    data = raw_data["examples"]
    raw_texts = raw_data["raw_texts"]

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
    with Pool(num_workers, initializer=_initializer,
              initargs=(entity_pool, negative_pool, all_neg_candidates, all_path_sentences, geometric_dist, max_neg_samples_num)) as p:
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

    _pos_aug = 0
    _res_aug = 0
    _sim_aug = 0
    _rep_ent_num = 0
    _rep_aug_num = 0
    simple_exp_num = 0
    for e in examples:
        _pos_aug += e.pop("pos_aug_num")
        _res_aug += e.pop("res_aug_num")
        _sim_aug += e.pop("sim_aug_num")
        if "rep_ent_num" in e:
            _rep_ent_num += e.pop("rep_ent_num")
            _rep_aug_num += 1
        else:
            simple_exp_num += 1

    _ctx_pos_aug = 0
    _ctx_res_aug = 0
    _ctx_sim_aug = 0
    simple_ctx_num = 0
    for e in context_examples:
        _ctx_pos_aug += e.pop("pos_aug_num")
        _ctx_res_aug += e.pop("res_aug_num")
        _ctx_sim_aug += e.pop("sim_aug_num")
        if "rep_ent_num" not in e:
            simple_ctx_num += 1

    if len(examples) > 0:
        logger.info(f"Augmentation statistics: ")
        logger.info(f"Augmentation from positive candidates: {_pos_aug} || {_pos_aug * 1.0 / len(examples)}")
        logger.info(f"Augmentation from rest sentences: {_res_aug} || {_res_aug * 1.0 / len(examples)}")
        logger.info(f"Augmentation from simple sentences: {_sim_aug} || {_sim_aug * 1.0 / len(examples)}")
        if _rep_aug_num:
            logger.info(f"Averaged replaced entity num: {_rep_ent_num} / {_rep_aug_num} || {_rep_ent_num * 1.0 / _rep_aug_num}")
        else:
            logger.info(f"Averaged replaced entity num: [0].")
        logger.info(f"Simple examples ratio over all examples: {simple_exp_num} / {len(examples)} = "
                    f"{simple_exp_num * 1.0 / len(examples)}.")

    if len(context_examples) > 0:
        logger.info(f"Context examples statistics: ")
        logger.info(f"Augmentation from positive candidates: {_ctx_pos_aug} || {_ctx_pos_aug * 1.0 / len(context_examples)}")
        logger.info(f"Augmentation from rest sentences: {_ctx_res_aug} || {_ctx_res_aug * 1.0 / len(context_examples)}")
        logger.info(f"Augmentation from simple sentences: {_ctx_sim_aug} || {_ctx_sim_aug * 1.0 / len(context_examples)}")
        logger.info(f"Simple context examples ratio over all context examples: {simple_ctx_num} / {len(context_examples)} = "
                    f"{simple_ctx_num * 1.0 / len(context_examples)}")

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

    # for span, h_spans, t_spans in zip(ctx_sentence_spans[0], ctx_h_spans[0], ctx_t_spans[0]):
    #     print(_tokenizer.convert_tokens_to_string(ctx_tokens[0][span[0]: span[1]]))
    #     print("============================")
    #     for span in h_spans + t_spans:
    #         print(_tokenizer.convert_tokens_to_string(ctx_tokens[0][span[0]: span[1]]))
    #     print("+++++++++++++++++++++++++++++++")

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


def convert_examples_into_features(file_path: str, tokenizer: PreTrainedTokenizer, id2rel_path_decode_id_file: str, rel_vocab: str,
                                   shuffle_context: bool = False, max_neg_num: int = 3, aug_num: int = 10,
                                   max_seq_length: int = 512, geo_p: float = 0.5, min_rep_num: int = 1,
                                   deduct_ratio: float = 1.0, context_ratio: float = 1.0, noise_sent_ratio: float = 0.5,
                                   remove_deduct: bool = False, remove_context: bool = False,
                                   max_neg_samples_num: int = 8, num_workers=48,
                                   add_cf_pair_data: bool = True):
    tokenizer_name = tokenizer.__class__.__name__
    tokenizer_name = tokenizer_name.replace('TokenizerFast', '')
    tokenizer_name = tokenizer_name.replace('Tokenizer', '').lower()

    file_suffix = f"{tokenizer_name}_{shuffle_context}_{max_neg_num}_{aug_num}_" \
                  f"{max_seq_length}_{geo_p}_{min_rep_num}_" \
                  f"{deduct_ratio}_{context_ratio}_{noise_sent_ratio}_{max_neg_samples_num}_" \
                  f"{'' if not remove_context else 'no-ctx-ex_'}{'' if not remove_deduct else 'no-duc-ex_'}path_v9.1.2"

    cached_file_path = f"{file_path}_{file_suffix}"
    if os.path.exists(cached_file_path):
        logger.info(f"Loading cached file from {cached_file_path}")
        all_examples, raw_texts = torch.load(cached_file_path)

        # if isinstance(all_examples[0]["tokens"][1], list):
        #     for sample in tqdm(all_examples, total=len(all_examples)):
        #         ctx_tokens = sample["tokens"]
        #         for op_id, op_tokens in enumerate(ctx_tokens):
        #             if op_id == 0:
        #                 continue
        #             op_string = tokenizer.convert_tokens_to_string(op_tokens)
        #             ctx_tokens[op_id] = op_string
        #     torch.save((all_examples, raw_texts), cached_file_path)

        dataset = WikiPathDatasetRelGenerateV1(all_examples, raw_texts, id2rel_path_decode_id_file, rel_vocab)

        return dataset

    examples, context_examples, raw_texts = read_examples(file_path, shuffle_context=shuffle_context, max_neg_num=max_neg_num,
                                                          aug_num=aug_num, geo_p=geo_p, min_rep_num=min_rep_num,
                                                          deduct_ratio=deduct_ratio, context_ratio=context_ratio,
                                                          noise_sent_ratio=noise_sent_ratio,
                                                          remove_deduct=remove_deduct, remove_context=remove_context,
                                                          max_neg_samples_num=max_neg_samples_num,
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

    return WikiPathDatasetRelGenerateV1(all_examples, raw_texts, id2rel_path_decode_id_file, rel_vocab)


# def convert_examples_into_features_v3_pair(file_path: str, tokenizer: PreTrainedTokenizer, id2rel_path_decode_id_file: str,
#                                            shuffle_context: bool = False, max_neg_num: int = 3, aug_num: int = 10,
#                                            max_seq_length: int = 512, geo_p: float = 0.5, min_rep_num: int = 1,
#                                            deduct_ratio: float = 1.0, context_ratio: float = 1.0, noise_sent_ratio: float = 0.5,
#                                            remove_deduct: bool = False, remove_context: bool = False,
#                                            max_neg_samples_num: int = 8, num_workers=48,
#                                            ):
#     tokenizer_name = tokenizer.__class__.__name__
#     tokenizer_name = tokenizer_name.replace('TokenizerFast', '')
#     tokenizer_name = tokenizer_name.replace('Tokenizer', '').lower()
#
#     file_suffix = f"{tokenizer_name}_{shuffle_context}_{max_neg_num}_{aug_num}_" \
#                   f"{max_seq_length}_{geo_p}_{min_rep_num}_" \
#                   f"{deduct_ratio}_{context_ratio}_{noise_sent_ratio}_{max_neg_samples_num}_" \
#                   f"{'' if not remove_context else 'no-ctx-ex_'}{'' if not remove_deduct else 'no-duc-ex_'}path_v9.1_split_vqvae_v3"
#
#     cached_file_path = f"{file_path}_{file_suffix}"
#     if os.path.exists(cached_file_path):
#         logger.info(f"Loading cached file from {cached_file_path}")
#         all_examples, raw_texts = torch.load(cached_file_path)
#         dataset = WikiPathDatasetPatternPairVQVAE(all_examples, raw_texts, id2rel_path_decode_id_file)
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
#     return WikiPathDatasetPatternPairVQVAE(all_examples, raw_texts, id2rel_path_decode_id_file)
#
#
# def convert_examples_into_features_v3(file_path: str, tokenizer: PreTrainedTokenizer, id2rel_path_decode_id_file: str,
#                                       shuffle_context: bool = False, max_neg_num: int = 3, aug_num: int = 10,
#                                       max_seq_length: int = 512, geo_p: float = 0.5, min_rep_num: int = 1,
#                                       deduct_ratio: float = 1.0, context_ratio: float = 1.0, noise_sent_ratio: float = 0.5,
#                                       remove_deduct: bool = False, remove_context: bool = False,
#                                       max_neg_samples_num: int = 8, num_workers=48,
#                                       generative_order: bool = False, rel_vocab: str = None, rel_vocab_size: int = None,
#                                       ):
#     tokenizer_name = tokenizer.__class__.__name__
#     tokenizer_name = tokenizer_name.replace('TokenizerFast', '')
#     tokenizer_name = tokenizer_name.replace('Tokenizer', '').lower()
#
#     file_suffix = f"{tokenizer_name}_{shuffle_context}_{max_neg_num}_{aug_num}_" \
#                   f"{max_seq_length}_{geo_p}_{min_rep_num}_" \
#                   f"{deduct_ratio}_{context_ratio}_{noise_sent_ratio}_{max_neg_samples_num}_" \
#                   f"{'' if not remove_context else 'no-ctx-ex_'}{'' if not remove_deduct else 'no-duc-ex_'}path_v9.1_split_vqvae_v3"
#
#     cached_file_path = f"{file_path}_{file_suffix}"
#     if os.path.exists(cached_file_path):
#         logger.info(f"Loading cached file from {cached_file_path}")
#         all_examples, raw_texts = torch.load(cached_file_path)
#         dataset = WikiPathDatasetRelGenerateV3(all_examples, raw_texts, id2rel_path_decode_id_file, rel_vocab_size, rel_vocab,
#                                                generative_order)
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
#     return WikiPathDatasetRelGenerateV3(all_examples, raw_texts, id2rel_path_decode_id_file, rel_vocab_size, rel_vocab, generative_order)
#
#
# def convert_examples_into_features_v3_ent(file_path: str, tokenizer: PreTrainedTokenizer, id2rel_path_decode_id_file: str,
#                                           shuffle_context: bool = False, max_neg_num: int = 3, aug_num: int = 10,
#                                           max_seq_length: int = 512, geo_p: float = 0.5, min_rep_num: int = 1,
#                                           deduct_ratio: float = 1.0, context_ratio: float = 1.0, noise_sent_ratio: float = 0.5,
#                                           remove_deduct: bool = False, remove_context: bool = False,
#                                           max_neg_samples_num: int = 8, num_workers=48,
#                                           generative_order: bool = False, rel_vocab: str = None, rel_vocab_size: int = None,
#                                           code_drop_threshold: int = 0, remove_cf_data_decoding: bool = False,
#                                           ):
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
#         dataset = WikiPathDatasetRelGenerateV3(all_examples, raw_texts, id2rel_path_decode_id_file, rel_vocab_size, rel_vocab,
#                                                generative_order, code_drop_threshold, remove_cf_data_decoding)
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
#     return WikiPathDatasetRelGenerateV3(all_examples, raw_texts, id2rel_path_decode_id_file, rel_vocab_size, rel_vocab, generative_order,
#                                         code_drop_threshold, remove_cf_data_decoding)
#
#
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