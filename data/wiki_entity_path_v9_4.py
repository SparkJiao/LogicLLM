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

from data.collators.wiki import WikiPathDatasetV6wPatternPair, WikiPathDatasetV6wPatternPairFull, WikiPathDatasetRelGenerateV1, \
    WikiPathDatasetV6wPatternPairsKMeans, WikiPathDatasetV5, WikiPathDatasetRelGenerateV2
from data.data_utils import get_all_permutation
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
"""

logger = get_child_logger("Wiki.Entity.Path.V9.1")

_entity_pool: Dict
_negative_pool: Dict
_all_neg_candidates: Dict
_all_path_sentences: Dict
_sent_label2sent: Dict[int, Dict]
_geometric_dist: torch.distributions.Distribution

_permutation_sample_num: int = 6
MAX_NEG_SAMPLE_NUM: int = 8


def _replace_entities_w_str(candidate, rep_pairs: Dict[int, str]):
    ent_to_rep = []

    ent_vis = set()

    for ent_id in candidate["ent"]:
        if ent_id in rep_pairs:
            for r in candidate["ent"][ent_id]:
                r["tgt"] = rep_pairs[ent_id]
                ent_to_rep.append(r)
            assert ent_id not in ent_vis
            ent_vis.add(ent_id)

    re = sorted(ent_to_rep, key=lambda x: x["pos"][0])
    # Non-overlapping check.
    for _tmp_id, _tmp in enumerate(re):
        if _tmp_id == 0:
            continue
        assert _tmp["pos"][0] >= re[_tmp_id - 1]["pos"][1]

    new_tokens = []
    _last_e = 0
    for r in re:
        s, e = r["pos"]
        new_tokens.extend(candidate["sent"][_last_e: s])
        new_tokens.append(r["tgt"])
        _last_e = e

    new_tokens.extend(candidate["sent"][_last_e:])
    return " ".join(new_tokens)


def get_ent_str(candi, ent_id):
    tokens = candi["sent"]
    # ent_mentions = list(candi["ent"][ent_id].values())
    ent_mentions = candi["ent"][ent_id]

    mention = random.choice(ent_mentions)
    return pos2str(mention["pos"][0], mention["pos"][1], tokens)


def pos2str(ent_s, ent_e, tokens):
    return " ".join(tokens[ent_s: ent_e])


def prepare_sent_label(examples):
    label2sent = collections.defaultdict(list)
    for example in tqdm(examples, total=len(examples), desc="preparing sent labels"):
        exp_id = example["id"]
        for s_id, sent in example["selected_sentences"].items():
            if sent["edge_label"] == -1:
                continue

            label2sent[sent["edge_label"]].append((exp_id, "path", s_id, sent))

        for pos_id, pos in enumerate(example["pos"]):
            if pos["edge_label"] == -1:
                continue

            label2sent[pos["edge_label"]].append((exp_id, "pos", pos_id, pos))

    return label2sent


def check_candidate(anchor, candidate):
    if candidate["h"] == anchor["h"] and candidate["t"] == anchor["t"]:
        return False
    if candidate["edge_label"] == anchor["edge_label"]:
        return False
    return True


def sample_sent_candidate(anchor, exp_id):
    edge_labels = list(_sent_label2sent.keys())

    _label = random.choice(edge_labels)
    while _label == anchor["edge_label"]:
        _label = random.choice(edge_labels)

    candidates = _sent_label2sent[_label]
    candidate = random.choice(candidates)
    cnt = 0
    # while candidate[0] == exp_id and (not check_candidate(anchor, candidate[3])):  # 挑负样本可以用来自同一个example的，这样反而更强
    # FIXED on 2022/12/10: while candidate[0] == exp_id or (not check_candidate(anchor, candidate[3])):
    while not check_candidate(anchor, candidate[3]):
        candidate = random.choice(candidates)
        cnt += 1
        if cnt > 10:
            return None
    return candidate


def sample_pos_candidate(anchor, exp_id):
    _label = anchor["edge_label"]
    # assert _label != -1
    if _label == -1:
        return None

    candidates = _sent_label2sent[_label]
    candidate = random.choice(candidates)
    cnt = 0
    while candidate[0] == exp_id or (anchor["h"] == candidate[3]["h"] and anchor["t"] == candidate[3]["t"]):  # 挑正样本尽量不用来自同一个example的
        candidate = random.choice(candidates)
        cnt += 1
        if cnt > 20:
            return None
    return candidate


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


def rep_context_sent(selected_sentences, tgt_sent_id, rep_tgt_sent, return_sentences: bool = False):
    context_sentences = {s_id: " ".join(s["sent"]) for s_id, s in selected_sentences.items()}
    context_sentences[tgt_sent_id] = rep_tgt_sent

    if return_sentences:
        return list(context_sentences.values())

    return " ".join(list(context_sentences.values()))


def entity_replace(pos, neg_candidate, rep_pairs: Dict[int, str] = None):
    pos_h, pos_t = pos["h"], pos["t"]
    neg_h, neg_t = neg_candidate["h"], neg_candidate["t"]
    pos_h_str = get_ent_str(pos, pos["h"])
    pos_t_str = get_ent_str(pos, pos["t"])

    if rep_pairs is None:
        rep_pairs = {}

    # neg(neg_h, neg_t) -> neg(pos_h, pos_t)
    rep_pairs_copy = copy.deepcopy(rep_pairs)
    if pos_h in rep_pairs:
        rep_pairs_copy[neg_h] = rep_pairs[pos_h]
    else:
        rep_pairs_copy[neg_h] = pos_h_str
    if pos_t in rep_pairs:
        rep_pairs_copy[neg_t] = rep_pairs[pos_t]
    else:
        rep_pairs_copy[neg_t] = pos_t_str
    return _replace_entities_w_str(neg_candidate, rep_pairs_copy)


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


def add_noise_sentence(num: int, item: Dict, orig_sentences: List[str], rep_pairs: Dict[int, str] = None, shuffle: bool = False) -> str:
    if num == 0:
        return " ".join(orig_sentences)

    noise_src_sent = sample_context_sentence(num, item)
    noise_tgt_sent = random.sample(list(item["selected_sentences"].values()), num)
    noise_sent = []
    for noise_tgt, noise_s in zip(noise_tgt_sent, noise_src_sent):
        # _res = replace_neg(noise_tgt, noise_s, rep_pairs=rep_pairs, out_of_domain=False)
        _res, _ = context_replace_neg(noise_tgt, noise_s, rep_pairs=rep_pairs, out_of_domain=False)
        if len(_res) > 0:
            noise_sent.append(_res[0])

    noise_ctx = insert_sentences(orig_sentences, noise_sent, shuffle=shuffle)
    return " ".join(noise_ctx)


def generate_ent_rep_pairs(pos_candi, neg_candi, non_path_ent_ids: set):
    rep_pairs = {}
    non_path_ent_ids_vis = copy.deepcopy(non_path_ent_ids)
    for neg_eng_id in neg_candi["ent"].keys():
        if neg_eng_id in [neg_candi["h"], neg_candi["t"]]:
            continue
        if neg_eng_id in pos_candi["ent"]:
            continue

        if len(non_path_ent_ids_vis) == 0:
            return None

        tgt_ent_id = random.choice(list(non_path_ent_ids_vis))
        tgt_ent_str = random.choice(list(_entity_pool[tgt_ent_id]))
        rep_pairs[neg_eng_id] = tgt_ent_str
        non_path_ent_ids_vis.remove(tgt_ent_id)
    return rep_pairs


def _initializer(entity_pool: Dict, negative_pool: Dict, all_neg_candidates: Dict, all_path_sentences: Dict, sent_label2sent: Dict,
                 geometric_dist: torch.distributions.Distribution,
                 max_neg_samples_num: int):
    global _entity_pool
    global _negative_pool
    global _all_neg_candidates
    global _all_path_sentences
    global _sent_label2sent
    global _geometric_dist
    global MAX_NEG_SAMPLE_NUM

    _entity_pool = entity_pool
    _negative_pool = negative_pool
    _all_neg_candidates = all_neg_candidates
    _all_path_sentences = all_path_sentences
    _sent_label2sent = sent_label2sent
    _geometric_dist = geometric_dist
    MAX_NEG_SAMPLE_NUM = max_neg_samples_num


def _process_single_item(item, max_neg_num: int, aug_num: int, min_rep_num: int, shuffle_context: bool,
                         deduct_ratio: float = 1.0, context_ratio: float = 1.0, noise_sent_ratio: float = 0.5,
                         remove_deduct: bool = False, remove_context: bool = False, pos_aug_ratio: float = 0.5):
    examples = []
    context_examples = []

    noise_sent_num = int(len(item["selected_sentences"]) * noise_sent_ratio)

    selected_sentences = item["selected_sentences"]
    if len(selected_sentences) == 0:
        return []
    context = " ".join([" ".join(s["sent"]) for s_id, s in selected_sentences.items()])
    orig_sentences = [" ".join(s["sent"]) for s_id, s in selected_sentences.items()]

    path_sent2rep = [(s_id, s) for s_id, s in selected_sentences.items() if s["h"] != -1 and s["t"] != -1]
    path_ent_ids = set()
    for s in selected_sentences.values():
        path_ent_ids.add(s["h"])
        path_ent_ids.add(s["t"])
    item_ent_ids = set([ent_id for ent_id in item["entity"] if ent_id not in path_ent_ids])

    path2sent_ids = []
    for i, x in enumerate(item["path"]):
        if i == 0:
            continue
        path2sent_ids.append(x[1])

    # neg_candidates = [x for x in item["rest_sentences"].values() if len(x["ent"]) > 1]

    for pos_idx, pos_candi in enumerate(item["pos"]):
        if pos_candi["edge_label"] == -1:
            continue
        # Statistics
        neg_res = []
        neg_edge_labels = []
        candidate_id_vis = set()

        cnt = 0
        while len(neg_res) < max_neg_num:
            cnt += 1
            neg_candidate = sample_sent_candidate(pos_candi, item["id"])
            if neg_candidate is None:
                if cnt > 30:
                    logger.warn("Stuck here!!!!!!!!!!!!!!!!!!")
                continue

            neg_candidate_id = neg_candidate[:3]
            if neg_candidate_id in candidate_id_vis:
                continue
            candidate_id_vis.add(neg_candidate_id)
            neg_candidate = neg_candidate[3]

            # rep_pairs = generate_ent_rep_pairs(pos_candi, neg_candidate, item_ent_ids - {pos_candi["h"], pos_candi["t"]})
            # if rep_pairs is None:
            #     continue
            #
            # neg = entity_replace(pos_candi, neg_candidate, rep_pairs=rep_pairs)
            neg = entity_replace(pos_candi, neg_candidate, rep_pairs=None)
            neg_res.append(neg)
            neg_edge_labels.append(neg_candidate["edge_label"])

        assert len(neg_res) >= max_neg_num

        _r = random.random()
        pos_aug_flag = False
        if not remove_deduct and _r < deduct_ratio:
            _r = random.random()
            aug_pos_candi = None
            if _r < pos_aug_ratio:
                sampled_pos_candi = sample_pos_candidate(pos_candi, item["id"])
                if sampled_pos_candi is not None:
                    aug_pos_candi = entity_replace(pos_candi, sampled_pos_candi[3], rep_pairs=None)
                    pos_aug_flag = True

            examples.append({
                "context": add_noise_sentence(num=noise_sent_num, item=item, orig_sentences=orig_sentences, rep_pairs=None, shuffle=True),
                "negative": neg_res,
                "positive": " ".join(pos_candi["sent"]) if aug_pos_candi is None else aug_pos_candi,
                "orig_id": item["id"],
                "pos_edge_label": pos_candi["edge_label"],
                "neg_edge_label": neg_edge_labels,
                "path_edge_labels": [selected_sentences[s_id]["edge_label"] for s_id in path2sent_ids],
                "pos_aug_flag": pos_aug_flag
            })

        # ============= context replaced-based examples ==================== #
        if len(path_sent2rep) == 0:
            continue

        neg_ctx_sent = []
        neg_ctx_edge_labels = []
        candidate_id_vis = set()

        cnt = 0
        while len(neg_ctx_sent) < max_neg_num:
            cnt += 1
            if cnt > 30:
                logger.warn("Stuck here!!!!!!!!!!!!!!!!!!")

            tgt_ctx_sent_id, tgt_ctx_sent = random.choice(path_sent2rep)

            neg_candidate = sample_sent_candidate(tgt_ctx_sent, item["id"])
            if neg_candidate is None:
                continue

            neg_candidate_id = neg_candidate[:3]
            if neg_candidate_id in candidate_id_vis:
                continue
            candidate_id_vis.add(neg_candidate_id)
            neg_candidate = neg_candidate[3]

            # rep_pairs = generate_ent_rep_pairs(tgt_ctx_sent, neg_candidate, item_ent_ids - {tgt_ctx_sent["h"], tgt_ctx_sent["t"]})
            # if rep_pairs is None:
            #     continue
            #
            # neg = entity_replace(tgt_ctx_sent, neg_candidate, rep_pairs=rep_pairs)
            neg = entity_replace(tgt_ctx_sent, neg_candidate, rep_pairs=None)
            neg_ctx_sent.append((tgt_ctx_sent_id, neg))
            neg_path_edge_labels = [selected_sentences[s_id]["edge_label"] if s_id != tgt_ctx_sent_id
                                    else neg_candidate["edge_label"] for s_id in path2sent_ids]
            neg_ctx_edge_labels.append(neg_path_edge_labels)

        assert len(neg_ctx_sent) >= max_neg_num

        _r = random.random()
        if not remove_context and _r < context_ratio:
            negative_context = [
                rep_context_sent(selected_sentences, _neg_sent[0], _neg_sent[1], return_sentences=True) for _neg_sent in neg_ctx_sent
            ]
            negative_noise_context = [
                add_noise_sentence(noise_sent_num, item, neg_ctx, rep_pairs=None, shuffle=True) for neg_ctx in negative_context
            ]

            _r = random.random()
            aug_context = None
            pos_aug_flag = False
            if _r < pos_aug_ratio:
                tgt_ctx_sent_id, tgt_ctx_sent = random.choice(path_sent2rep)
                sampled_ctx_pos_candi = sample_pos_candidate(tgt_ctx_sent, item["id"])
                if sampled_ctx_pos_candi is not None:
                    aug_context = {s_id: " ".join(s["sent"]) for s_id, s in selected_sentences.items()}
                    aug_context[tgt_ctx_sent_id] = entity_replace(tgt_ctx_sent, sampled_ctx_pos_candi[3], rep_pairs=None)
                    aug_context = list(aug_context.values())
                    pos_aug_flag = True

            context_examples.append({
                "context": add_noise_sentence(num=noise_sent_num, item=item,
                                              orig_sentences=orig_sentences if aug_context is None else aug_context,
                                              rep_pairs=None, shuffle=True),
                "condition": " ".join(pos_candi["sent"]),
                "negative_context": negative_noise_context,
                "orig_id": item["id"],
                "pos_edge_label": pos_candi["edge_label"],
                "neg_path_edge_labels": neg_ctx_edge_labels,
                "path_edge_labels": [selected_sentences[s_id]["edge_label"] for s_id in path2sent_ids],
                "pos_aug_flag": pos_aug_flag
            })

    # Augment the context
    # 1. ~~Choose the head entity or the tail entity as the target entity.~~
    #    Randomly sample some entities in the path to be replaced.
    # 2. Randomly sample other entities from the entity pool.
    # 3. Replace the target entity in the context with the sampled entity.
    # 4. Replace the target entity in negative samples with the sampled entity.

    # Gather the entity ids in the meta-path for sampling.
    # path_ent_ids = set([p_ent_id for p_ent_id, p_sent_id in item["path"]])
    h_t_ent_ids = [item["pos"][0]["h"], item["pos"][0]["t"]]
    # for x in h_t_ent_ids:
    #     assert x in path_ent_ids
    #     path_ent_ids.remove(x)
    # if "relation_connect_ent" in item:
    #     for x in item["relation_connect_ent"]:
    #         if x in h_t_ent_ids:
    #             continue
    #         path_ent_ids.remove(x)
    item_ent_ids = set(list(item["entity"].keys()))
    for x in h_t_ent_ids:
        assert x in item_ent_ids
        item_ent_ids.remove(x)

    _h_str = get_ent_str(item["pos"][0], h_t_ent_ids[0])
    _t_str = get_ent_str(item["pos"][0], h_t_ent_ids[1])

    for _ in range(aug_num):  # Repeat for augmentation

        for pos_idx, pos_candi in enumerate(item["pos"]):
            if pos_candi["edge_label"] == -1:
                continue

            # TODO: 反事实替换实体这一部分，是否也能够根据`edge label`
            # Sample the amount of entities to be replaced from the geometric distribution.
            if min_rep_num >= len(item_ent_ids) + len(h_t_ent_ids):
                _sampled_ent_num = len(item_ent_ids) + len(h_t_ent_ids)
            else:
                if _geometric_dist is not None:
                    _sampled_ent_num = int(_geometric_dist.sample().item()) + min_rep_num
                else:
                    _sampled_ent_num = min_rep_num
                cnt = 0
                while _sampled_ent_num >= (len(item_ent_ids) + len(h_t_ent_ids)):
                    cnt += 1
                    _sampled_ent_num = int(_geometric_dist.sample().item()) + min_rep_num
                    if cnt > 1000:
                        logger.warning("Wrong here.")
                        raise RuntimeError()
                assert min_rep_num <= _sampled_ent_num < (len(item_ent_ids) + len(h_t_ent_ids))

            # Make sure the head/tail entity in the entities to be replaced.
            if _sampled_ent_num <= 2:
                sampled_ent_ids = random.sample(h_t_ent_ids, _sampled_ent_num)
            else:
                sampled_ent_ids = h_t_ent_ids + random.sample(list(item_ent_ids), _sampled_ent_num - 2)

            target_ent_str = sample_entity(_entity_pool, item_ent_ids | set(h_t_ent_ids), _sampled_ent_num)
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

            new_sentences = []
            for _, sent in selected_sentences.items():
                new_sentences.append(_replace_entities_w_str(sent, _cur_aug_rep_pairs))

            new_pos_candi_sent = _replace_entities_w_str(pos_candi, _cur_aug_rep_pairs)
            _r = random.random()
            pos_aug_flag = False
            if _r < pos_aug_ratio:
                sampled_pos_candi = sample_pos_candidate(pos_candi, item["id"])
                if sampled_pos_candi is not None:
                    new_pos_candi_sent = entity_replace(pos_candi, sampled_pos_candi[3], rep_pairs=_cur_aug_rep_pairs)
                    pos_aug_flag = True

            neg_res = []
            neg_edge_labels = []
            candidate_id_vis = set()

            cnt = 0
            while len(neg_res) < max_neg_num:
                cnt += 1
                neg_candidate = sample_sent_candidate(pos_candi, item["id"])
                if neg_candidate is None:
                    if cnt > 30:
                        logger.warn("Stuck here!!!!!!!!!!!!!!!!!!")
                    continue

                neg_candidate_id = neg_candidate[:3]
                if neg_candidate_id in candidate_id_vis:
                    continue
                candidate_id_vis.add(neg_candidate_id)
                neg_candidate = neg_candidate[3]

                # rep_pairs = generate_ent_rep_pairs(pos_candi, neg_candidate, item_ent_ids - {pos_candi["h"], pos_candi["t"]})
                # if rep_pairs is None:
                #     continue
                # rep_pairs.update(_cur_aug_rep_pairs)
                #
                # neg = entity_replace(pos_candi, neg_candidate, rep_pairs=rep_pairs)
                neg = entity_replace(pos_candi, neg_candidate, rep_pairs=_cur_aug_rep_pairs)
                neg_res.append(neg)
                neg_edge_labels.append(neg_candidate["edge_label"])

            assert len(neg_res) >= max_neg_num

            if shuffle_context:
                random.shuffle(new_sentences)

            new_context = add_noise_sentence(noise_sent_num, item, new_sentences, rep_pairs=_cur_aug_rep_pairs, shuffle=True)

            if not remove_deduct:
                examples.append({
                    "context": new_context,
                    "negative": neg_res,
                    "positive": new_pos_candi_sent,
                    "orig_id": item["id"],
                    "h": _cur_aug_rep_pairs[h_t_ent_ids[0]] if h_t_ent_ids[0] in _cur_aug_rep_pairs else _h_str,
                    "t": _cur_aug_rep_pairs[h_t_ent_ids[1]] if h_t_ent_ids[1] in _cur_aug_rep_pairs else _t_str,
                    "rep_ent_num": _sampled_ent_num,
                    "pos_edge_label": pos_candi["edge_label"],
                    "neg_edge_label": neg_edge_labels,
                    "path_edge_labels": [selected_sentences[s_id]["edge_label"] for s_id in path2sent_ids],
                    "pos_aug_flag": pos_aug_flag
                })

            # ============= context replaced-based examples ==================== #
            if len(path_sent2rep) == 0:
                continue

            neg_ctx_sent = []
            neg_ctx_edge_labels = []
            candidate_id_vis = set()

            while len(neg_ctx_sent) < max_neg_num:
                cnt += 1
                if cnt > 30:
                    logger.warn("Stuck here!!!!!!!!!!!!!!!!!!")

                tgt_ctx_sent_id, tgt_ctx_sent = random.choice(path_sent2rep)

                neg_candidate = sample_sent_candidate(tgt_ctx_sent, item["id"])
                if neg_candidate is None:
                    continue

                neg_candidate_id = neg_candidate[:3]
                if neg_candidate_id in candidate_id_vis:
                    continue
                candidate_id_vis.add(neg_candidate_id)
                neg_candidate = neg_candidate[3]

                # rep_pairs = generate_ent_rep_pairs(tgt_ctx_sent, neg_candidate, item_ent_ids - {tgt_ctx_sent["h"], tgt_ctx_sent["t"]})
                # if rep_pairs is None:
                #     continue
                # rep_pairs.update(_cur_aug_rep_pairs)
                #
                # neg = entity_replace(tgt_ctx_sent, neg_candidate, rep_pairs=rep_pairs)
                neg = entity_replace(tgt_ctx_sent, neg_candidate, rep_pairs=_cur_aug_rep_pairs)
                neg_ctx_sent.append((tgt_ctx_sent_id, neg))
                neg_path_edge_labels = [selected_sentences[s_id]["edge_label"] if s_id != tgt_ctx_sent_id
                                        else neg_candidate["edge_label"] for s_id in path2sent_ids]
                neg_ctx_edge_labels.append(neg_path_edge_labels)

            # To avoid shuffling, re-generate the new sentences
            new_sentences = dict()
            for s_id, sent in selected_sentences.items():
                new_sentences[s_id] = _replace_entities_w_str(sent, _cur_aug_rep_pairs)

            _r = random.random()
            pos_aug_flag = False
            if _r < pos_aug_ratio:
                tgt_ctx_sent_id, tgt_ctx_sent = random.choice(path_sent2rep)
                sampled_ctx_pos_candi = sample_pos_candidate(tgt_ctx_sent, item["id"])
                if sampled_ctx_pos_candi is not None:
                    new_sentences[tgt_ctx_sent_id] = entity_replace(tgt_ctx_sent, sampled_ctx_pos_candi[3], rep_pairs=_cur_aug_rep_pairs)
                    pos_aug_flag = True

            negative_context = []
            for _neg_sent in neg_ctx_sent:
                _new_context = copy.deepcopy(new_sentences)
                _new_context[_neg_sent[0]] = _neg_sent[1]
                _new_context_sentences = list(_new_context.values())
                if shuffle_context:
                    random.shuffle(_new_context_sentences)
                negative_context.append(add_noise_sentence(noise_sent_num, item, _new_context_sentences, rep_pairs=_cur_aug_rep_pairs,
                                                           shuffle=True))

            new_sentences = list(new_sentences.values())
            if shuffle_context:
                random.shuffle(new_sentences)
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
                    "pos_edge_label": pos_candi["edge_label"],
                    "neg_path_edge_labels": neg_ctx_edge_labels,
                    "path_edge_labels": [selected_sentences[s_id]["edge_label"] for s_id in path2sent_ids],
                    "pos_aug_flag": pos_aug_flag
                })

    return examples, context_examples


def read_examples(file_path: str, shuffle_context: bool = False,
                  max_neg_num: int = 3, aug_num: int = 10,
                  geo_p: float = 0.5, min_rep_num: int = 1,
                  deduct_ratio: float = 1.0, context_ratio: float = 1.0, noise_sent_ratio: float = 0.5,
                  remove_deduct: bool = False, remove_context: bool = False, pos_aug_ratio: float = 0.5,
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

    sent_label2sent = prepare_sent_label(data)

    examples = []
    context_examples = []
    with Pool(num_workers, initializer=_initializer,
              initargs=(entity_pool, negative_pool, all_neg_candidates, all_path_sentences,
                        sent_label2sent, geometric_dist, max_neg_samples_num)) as p:
        _annotate = partial(_process_single_item,
                            max_neg_num=max_neg_num, aug_num=aug_num, min_rep_num=min_rep_num, shuffle_context=shuffle_context,
                            deduct_ratio=deduct_ratio, context_ratio=context_ratio, noise_sent_ratio=noise_sent_ratio,
                            remove_deduct=remove_deduct, remove_context=remove_context, pos_aug_ratio=pos_aug_ratio)
        _results = list(tqdm(
            p.imap(_annotate, data, chunksize=32),
            total=len(data),
            desc="Reading examples"
        ))

    pos_aug_num = 0
    for _res, _context_res in _results:
        if _res:
            examples.extend(_res)
            if examples[-1]["pos_aug_flag"]:
                pos_aug_num += 1
        if _context_res:
            context_examples.extend(_context_res)
            if context_examples[-1]["pos_aug_flag"]:
                pos_aug_num += 1

    logger.info(f"{len(examples)} examples are loaded from {file_path}.")
    logger.info(f"{len(context_examples)} context examples are loaded from {file_path}.")
    logger.info(f"Positive candidate augmented example ratio: "
                f"{pos_aug_num} / {len(examples) + len(context_examples)} = {pos_aug_num / (len(examples) + len(context_examples))}")

    return examples, context_examples, raw_texts


_tokenizer: PreTrainedTokenizer


def length_filter_init(tokenizer: PreTrainedTokenizer):
    global _tokenizer
    _tokenizer = tokenizer


def length_filter(sample, max_seq_length: int):
    if "negative_context" in sample:
        tokens_b = _tokenizer.tokenize(sample["condition"])
        for ctx in sample["negative_context"] + [sample["context"]]:
            tokens_a = _tokenizer.tokenize(ctx)
            tokens = _tokenizer.build_inputs_with_special_tokens(tokens_a, tokens_b)
            if len(tokens) > max_seq_length:
                return False
    else:
        tokens_a = _tokenizer.tokenize(sample["context"])
        for option in sample["negative"] + [sample["positive"]]:
            tokens_b = _tokenizer.tokenize(option)
            tokens = _tokenizer.build_inputs_with_special_tokens(tokens_a, tokens_b)
            if len(tokens) > max_seq_length:
                return False
    return True


# def convert_examples_into_features_ctr(file_path: str, tokenizer: PreTrainedTokenizer,
#                                        shuffle_context: bool = False, max_neg_num: int = 3, aug_num: int = 10,
#                                        max_seq_length: int = 512, geo_p: float = 0.5, min_rep_num: int = 1,
#                                        deduct_ratio: float = 1.0, context_ratio: float = 1.0, noise_sent_ratio: float = 0.5,
#                                        remove_deduct: bool = False, remove_context: bool = False,
#                                        max_neg_samples_num: int = 8, num_workers=48):
#     tokenizer_name = tokenizer.__class__.__name__
#     tokenizer_name = tokenizer_name.replace('TokenizerFast', '')
#     tokenizer_name = tokenizer_name.replace('Tokenizer', '').lower()
#
#     file_suffix = f"{tokenizer_name}_{shuffle_context}_{max_neg_num}_{aug_num}_" \
#                   f"{max_seq_length}_{geo_p}_{min_rep_num}_" \
#                   f"{deduct_ratio}_{context_ratio}_{noise_sent_ratio}_{max_neg_samples_num}_" \
#                   f"{'' if not remove_context else 'no-ctx-ex_'}{'' if not remove_deduct else 'no-duc-ex_'}path_v9.4"
#
#     cached_file_path = f"{file_path}_{file_suffix}"
#     if os.path.exists(cached_file_path):
#         logger.info(f"Loading cached file from {cached_file_path}")
#         all_examples, raw_texts = torch.load(cached_file_path)
#         dataset = WikiPathDatasetV5(all_examples, raw_texts)
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
#     with Pool(num_workers, initializer=length_filter_init, initargs=(tokenizer,)) as p:
#         _annotate = partial(length_filter, max_seq_length=max_seq_length)
#         flags = list(tqdm(
#             p.imap(_annotate, examples + context_examples, chunksize=32),
#             total=(len(examples) + len(context_examples)),
#             desc="filtering examples by length",
#         ))
#     all_examples = []
#     for flag, exp in zip(flags, examples + context_examples):
#         if flag:
#             all_examples.append(exp)
#
#     # Save
#     logger.info(f"Saving processed features into {cached_file_path}.")
#     torch.save((all_examples, raw_texts), cached_file_path)
#
#     return WikiPathDatasetV5(all_examples, raw_texts)
#
#
# def convert_examples_into_features(file_path: str, tokenizer: PreTrainedTokenizer, pattern_pair_file: str,
#                                    shuffle_context: bool = False, max_neg_num: int = 3, aug_num: int = 10,
#                                    max_seq_length: int = 512, geo_p: float = 0.5, min_rep_num: int = 1,
#                                    deduct_ratio: float = 1.0, context_ratio: float = 1.0, noise_sent_ratio: float = 0.5,
#                                    remove_deduct: bool = False, remove_context: bool = False,
#                                    max_neg_samples_num: int = 8, num_workers=48,
#                                    add_cf_pair_data: bool = True):
#     tokenizer_name = tokenizer.__class__.__name__
#     tokenizer_name = tokenizer_name.replace('TokenizerFast', '')
#     tokenizer_name = tokenizer_name.replace('Tokenizer', '').lower()
#
#     file_suffix = f"{tokenizer_name}_{shuffle_context}_{max_neg_num}_{aug_num}_" \
#                   f"{max_seq_length}_{geo_p}_{min_rep_num}_" \
#                   f"{deduct_ratio}_{context_ratio}_{noise_sent_ratio}_{max_neg_samples_num}_" \
#                   f"{'' if not remove_context else 'no-ctx-ex_'}{'' if not remove_deduct else 'no-duc-ex_'}path_v9.4"
#
#     cached_file_path = f"{file_path}_{file_suffix}"
#     if os.path.exists(cached_file_path):
#         logger.info(f"Loading cached file from {cached_file_path}")
#         all_examples, raw_texts = torch.load(cached_file_path)
#         dataset = WikiPathDatasetV6wPatternPair(all_examples, raw_texts, pattern_pair_file, add_cf_pair_data=add_cf_pair_data)
#         return dataset
#
#     examples, context_examples, raw_texts = read_examples(file_path, shuffle_context=shuffle_context, max_neg_num=max_neg_num,
#                                                           aug_num=aug_num, geo_p=geo_p, min_rep_num=min_rep_num,
#                                                           deduct_ratio=deduct_ratio, context_ratio=context_ratio,
#                                                           noise_sent_ratio=noise_sent_ratio,
#                                                           remove_deduct=remove_deduct, remove_context=remove_context,
#                                                           max_neg_samples_num=max_neg_samples_num,
#                                                           num_workers=num_workers)
#     all_examples = examples + context_examples
#
#     # Save
#     logger.info(f"Saving processed features into {cached_file_path}.")
#     torch.save((all_examples, raw_texts), cached_file_path)
#
#     return WikiPathDatasetV6wPatternPair(all_examples, raw_texts, pattern_pair_file, add_cf_pair_data=add_cf_pair_data)
#
#
# def convert_examples_into_features_v2(file_path: str, tokenizer: PreTrainedTokenizer, pattern_pair_file: str,
#                                       shuffle_context: bool = False, max_neg_num: int = 3, aug_num: int = 10,
#                                       max_seq_length: int = 512, geo_p: float = 0.5, min_rep_num: int = 1,
#                                       deduct_ratio: float = 1.0, context_ratio: float = 1.0, noise_sent_ratio: float = 0.5,
#                                       remove_deduct: bool = False, remove_context: bool = False,
#                                       max_neg_samples_num: int = 8, num_workers=48):
#     tokenizer_name = tokenizer.__class__.__name__
#     tokenizer_name = tokenizer_name.replace('TokenizerFast', '')
#     tokenizer_name = tokenizer_name.replace('Tokenizer', '').lower()
#
#     file_suffix = f"{tokenizer_name}_{shuffle_context}_{max_neg_num}_{aug_num}_" \
#                   f"{max_seq_length}_{geo_p}_{min_rep_num}_" \
#                   f"{deduct_ratio}_{context_ratio}_{noise_sent_ratio}_{max_neg_samples_num}_" \
#                   f"{'' if not remove_context else 'no-ctx-ex_'}{'' if not remove_deduct else 'no-duc-ex_'}path_v9.4"
#
#     cached_file_path = f"{file_path}_{file_suffix}"
#     if os.path.exists(cached_file_path):
#         logger.info(f"Loading cached file from {cached_file_path}")
#         all_examples, raw_texts = torch.load(cached_file_path)
#         dataset = WikiPathDatasetV6wPatternPairFull(all_examples, raw_texts, pattern_pair_file)
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
#         _annotate = partial(length_filter, max_seq_length=max_seq_length)
#         flags = list(tqdm(
#             p.imap(_annotate, examples + context_examples, chunksize=32),
#             total=(len(examples) + len(context_examples)),
#             desc="filtering examples by length",
#         ))
#     all_examples = []
#     for flag, exp in zip(flags, examples + context_examples):
#         if flag:
#             all_examples.append(exp)
#
#     # Save
#     logger.info(f"Saving processed features into {cached_file_path}.")
#     torch.save((all_examples, raw_texts), cached_file_path)
#
#     return WikiPathDatasetV6wPatternPairFull(all_examples, raw_texts, pattern_pair_file)


def convert_examples_into_features(file_path: str, tokenizer: PreTrainedTokenizer, rel_vocab: str,
                                   shuffle_context: bool = False, max_neg_num: int = 3, aug_num: int = 10,
                                   max_seq_length: int = 512, geo_p: float = 0.5, min_rep_num: int = 1,
                                   deduct_ratio: float = 1.0, context_ratio: float = 1.0, noise_sent_ratio: float = 0.5,
                                   remove_deduct: bool = False, remove_context: bool = False, pos_aug_ratio: float = 0.5,
                                   max_neg_samples_num: int = 8, num_workers=48,
                                   ):
    tokenizer_name = tokenizer.__class__.__name__
    tokenizer_name = tokenizer_name.replace('TokenizerFast', '')
    tokenizer_name = tokenizer_name.replace('Tokenizer', '').lower()

    file_suffix = f"{tokenizer_name}_{shuffle_context}_{max_neg_num}_{aug_num}_" \
                  f"{max_seq_length}_{geo_p}_{min_rep_num}_" \
                  f"{deduct_ratio}_{context_ratio}_{noise_sent_ratio}_{pos_aug_ratio}_{max_neg_samples_num}_" \
                  f"{'' if not remove_context else 'no-ctx-ex_'}{'' if not remove_deduct else 'no-duc-ex_'}path_v9.4"

    cached_file_path = f"{file_path}_{file_suffix}"
    if os.path.exists(cached_file_path):
        logger.info(f"Loading cached file from {cached_file_path}")
        all_examples, raw_texts = torch.load(cached_file_path)
        dataset = WikiPathDatasetRelGenerateV2(all_examples, raw_texts, rel_vocab)
        return dataset

    examples, context_examples, raw_texts = read_examples(file_path, shuffle_context=shuffle_context, max_neg_num=max_neg_num,
                                                          aug_num=aug_num, geo_p=geo_p, min_rep_num=min_rep_num,
                                                          deduct_ratio=deduct_ratio, context_ratio=context_ratio,
                                                          noise_sent_ratio=noise_sent_ratio,
                                                          remove_deduct=remove_deduct, remove_context=remove_context,
                                                          max_neg_samples_num=max_neg_samples_num, pos_aug_ratio=pos_aug_ratio,
                                                          num_workers=num_workers)
    with Pool(num_workers, initializer=length_filter_init, initargs=(tokenizer,)) as p:
        _annotate = partial(length_filter, max_seq_length=max_seq_length)
        flags = list(tqdm(
            p.imap(_annotate, examples + context_examples, chunksize=32),
            total=(len(examples) + len(context_examples)),
            desc="filtering examples by length",
        ))
    all_examples = []
    for flag, exp in zip(flags, examples + context_examples):
        if flag:
            all_examples.append(exp)

    # Save
    logger.info(f"Saving processed features into {cached_file_path}.")
    torch.save((all_examples, raw_texts), cached_file_path)

    return WikiPathDatasetRelGenerateV2(all_examples, raw_texts, rel_vocab)


# def convert_examples_into_features_v4(file_path: str, tokenizer: PreTrainedTokenizer, rel_path_set_file: str,
#                                       shuffle_context: bool = False, max_neg_num: int = 3, aug_num: int = 10,
#                                       max_seq_length: int = 512, geo_p: float = 0.5, min_rep_num: int = 1,
#                                       deduct_ratio: float = 1.0, context_ratio: float = 1.0, noise_sent_ratio: float = 0.5,
#                                       remove_deduct: bool = False, remove_context: bool = False,
#                                       max_neg_samples_num: int = 8, num_workers=48):
#     tokenizer_name = tokenizer.__class__.__name__
#     tokenizer_name = tokenizer_name.replace('TokenizerFast', '')
#     tokenizer_name = tokenizer_name.replace('Tokenizer', '').lower()
#
#     file_suffix = f"{tokenizer_name}_{shuffle_context}_{max_neg_num}_{aug_num}_" \
#                   f"{max_seq_length}_{geo_p}_{min_rep_num}_" \
#                   f"{deduct_ratio}_{context_ratio}_{noise_sent_ratio}_{max_neg_samples_num}_" \
#                   f"{'' if not remove_context else 'no-ctx-ex_'}{'' if not remove_deduct else 'no-duc-ex_'}path_v9.4"
#
#     cached_file_path = f"{file_path}_{file_suffix}"
#     if os.path.exists(cached_file_path):
#         logger.info(f"Loading cached file from {cached_file_path}")
#         all_examples, raw_texts = torch.load(cached_file_path)
#         dataset = WikiPathDatasetV6wPatternPairsKMeans(all_examples, raw_texts, rel_path_set_file)
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
#         _annotate = partial(length_filter, max_seq_length=max_seq_length)
#         flags = list(tqdm(
#             p.imap(_annotate, examples + context_examples, chunksize=32),
#             total=(len(examples) + len(context_examples)),
#             desc="filtering examples by length",
#         ))
#     all_examples = []
#     for flag, exp in zip(flags, examples + context_examples):
#         if flag:
#             all_examples.append(exp)
#
#     # Save
#     logger.info(f"Saving processed features into {cached_file_path}.")
#     torch.save((all_examples, raw_texts), cached_file_path)
#
#     return WikiPathDatasetV6wPatternPairsKMeans(all_examples, raw_texts, rel_path_set_file)


def _quick_loading(file_path: str, pattern_pair_file: str, **kwargs):
    logger.info(f"Quickly loading cached file from {file_path}")
    all_examples, raw_texts = torch.load(file_path)
    dataset = WikiPathDatasetV6wPatternPair(all_examples, raw_texts, pattern_pair_file)
    return dataset
