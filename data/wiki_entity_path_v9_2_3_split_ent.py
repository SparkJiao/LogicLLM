import collections
import copy
import os
import pickle
import random
from functools import partial
from multiprocessing import Pool
from typing import Dict, List

import torch
import math
from torch.distributions.geometric import Geometric
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from data.collators.wiki import WikiPathDatasetV6wPatternPair, WikiPathDatasetV6wPatternPairFull
from data.data_utils import get_all_permutation
from general_util.logger import get_child_logger
from transformers.tokenization_utils import TruncationStrategy, PaddingStrategy

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
Version 9.1:
    1.  Use the unique entity id in Wikidata to avoid repeat entity replacement.
Version split:
    1.  Don't combine the sentences together.
Version 9.2:
    1.  Replace the entities that are not included in the path as different entities of different mentions. Applied only to context.
Version 9.2.1:
    1.  The above replacement is also applied to the positive candidate. 
Version 9.2.2:
    1.  The negative candidates from the same passage are also applied.
Version 9.2.3:
    1.  考虑到如果仅仅通过替换成任意的实体的话，虽然可能会出现重复，但没有保证，导致可能会出现shortcut，即模型只需要检查每个实体出现的次数，若仅出现了一次，
        则该实体一定不会在推理路径中。因此，更进一步地，我们可能可以首先采样一个相对比较大的实体集合，然后对于每个句子，除了那些应当被排除出去的实体外，
        对于剩下的实体，随机从这个集合中取出一部分实体，然后进行替换。
"""

logger = get_child_logger("Wiki.Entity.Path.V9.2.3.split")

_entity_pool: Dict
_negative_pool: Dict
_all_neg_candidates: Dict
_all_path_sentences: Dict
_geometric_dist: torch.distributions.Distribution
_tokenizer: PreTrainedTokenizer

_permutation_sample_num: int = 6
MAX_NEG_SAMPLE_NUM: int = 8
MAX_ENT_NUM_PER_SENTENCE: int = 5


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
    res_ent_ids = []
    pool_vis = copy.deepcopy(path_ent_ids)
    cnt = 0
    for _ in range(k):
        ent_id = random.choice(all_ent_id_ls)
        while ent_id in pool_vis:
            cnt += 1
            ent_id = random.choice(all_ent_id_ls)
            if cnt > 100:
                raise RuntimeError(k)

        pool_vis.add(ent_id)
        res_ent_ids.append(ent_id)

        entity_str = random.choice(list(pool[ent_id]))
        res.append(entity_str)
    return res, res_ent_ids


def rep_context_sent(selected_sentences, tgt_sent_id, rep_tgt_sent, return_sentences: bool = False):
    context_sentences = {s_id: " ".join(s["sent"]) for s_id, s in selected_sentences.items()}
    context_sentences[tgt_sent_id] = rep_tgt_sent

    if return_sentences:
        return list(context_sentences.values())

    return " ".join(list(context_sentences.values()))


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


def extract_bias_entity(sentences):
    """
    1.  Select all entities not included in the path but has appeared more than once across different sentences.
    2.  Generate a series pair of <sentence id, entity id> for following replacement.
        Note that there should be at least one mention of a specific entity that is kept.
    3.  Replace the entity mentions with randomly sampled tgg
    """
    pass
    #   一些思考：
    #   这个问题的核心在于，如果一个entity在某个example里出现了很多次，那么对于每一个mention我们希望替换成不同的entity，来避免潜在的边。
    #   具体实现上，我们需要对于每一个sentence在处理的时候，动态的generate一个`rep_pairs`。
    #
    #   此处需要考虑常识问题，我们其实是希望在下游任务里保留一定的implicit的常识知识的，（但具体怎样做不清楚）。比如要求被替换的实体在wikidata里不能和
    #   已有的entity存在relation（可能在实现上有点复杂）；再比如如果某个实体和path entities中的某个entity存在Wikidata上的relation，我们需要考虑替换掉他。
    #
    #   现阶段，我希望直接替换掉全部的非path上的entity试试，同时保证不同的mention替换的效果不同。


def bias_entity_generate_rep_pairs(sentence, existing_ent_ids: set, exclude: bool = True, candidate_entity_pool=None):
    src_ent_ids = []
    if exclude:
        for ent_id in sentence["ent"]:
            if ent_id not in [sentence["h"], sentence["t"]]:
                src_ent_ids.append(ent_id)
    else:
        src_ent_ids = sentence["ent"]

    if candidate_entity_pool is None:
        tgt_ent_str, tgt_ent_ids = sample_entity(_entity_pool, existing_ent_ids, len(src_ent_ids))
    else:
        tgt_ent_str, tgt_ent_ids = sample_entity(candidate_entity_pool, existing_ent_ids, len(src_ent_ids))

    return {src_ent_id: tgt_str for src_ent_id, tgt_str in zip(src_ent_ids, tgt_ent_str)}


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


def _process_single_item(item, max_neg_num: int, aug_num: int, min_rep_num: int, shuffle_context: bool,
                         deduct_ratio: float = 1.0, context_ratio: float = 1.0, noise_sent_ratio: float = 0.5,
                         remove_deduct: bool = False, remove_context: bool = False):
    examples = []
    context_examples = []

    noise_sent_num = int(len(item["selected_sentences"]) * noise_sent_ratio)

    selected_sentences = item["selected_sentences"]
    if len(selected_sentences) == 0:
        return []
    context = " ".join([" ".join(s["sent"]) for s_id, s in selected_sentences.items()])
    orig_sentences = [" ".join(s["sent"]) for s_id, s in selected_sentences.items()]

    path_sent2rep = [(s_id, s) for s_id, s in selected_sentences.items() if s["h"] != -1 and s["t"] != -1]

    neg_candidates = [x for x in item["rest_sentences"].values() if len(x["ent"]) > 1]

    path_ent_ids = []
    for sent in selected_sentences.values():
        path_ent_ids.extend([sent["h"], sent["t"]])
    path_ent_ids = set(path_ent_ids)
    assert -1 not in path_ent_ids

    for pos_idx, pos_candi in enumerate(item["pos"]):
        # Statistics
        _res_aug = 0
        _pos_aug = 0
        _sim_aug = 0

        neg_res = []

        # Sample a sub-entity pool to avoid new short-cut through entity amount
        item_ent = item["entity"]
        _, sampled_sub_ent_pool_ids = sample_entity(_entity_pool, set(list(item_ent.keys())),
                                                    max(MAX_ENT_NUM_PER_SENTENCE, math.ceil((len(item_ent) - len(path_ent_ids)) * 1.5)))
        sampled_sub_ent_pool = {_ent_id: copy.deepcopy(_entity_pool[_ent_id]) for _ent_id in sampled_sub_ent_pool_ids}

        # Other positive candidates
        mutual_samples = [candi for candi_idx, candi in enumerate(item["pos"]) if candi_idx != pos_idx]
        for neg in mutual_samples:
            if len(neg["ent"]) > MAX_ENT_NUM_PER_SENTENCE:
                continue

            # TODO: `switch`的一个问题在于，引入了本句中另外的一个实体的关系，可能出现一些side information来辅助模型判断。
            #   当然这个问题在使用随机采样的句子的时候也会出现。因为随机采样的句子也可能包含新的实体信息。
            #   有可能解决吗？只交换两个实体的位置？可能会导致换了又没换
            augmented_rep_pairs = bias_entity_generate_rep_pairs(neg, existing_ent_ids=path_ent_ids, exclude=False,
                                                                 candidate_entity_pool=sampled_sub_ent_pool)
            _rep_res = switch_replace_neg(pos_candi, neg, rep_pairs=augmented_rep_pairs)
            if len(_rep_res) > 0:
                neg_res.extend(_rep_res)
        _pos_aug += len(neg_res)

        # Easy samples.
        for neg in neg_candidates:
            if len(neg["ent"]) > MAX_ENT_NUM_PER_SENTENCE:
                continue
            augmented_rep_pairs = bias_entity_generate_rep_pairs(neg, existing_ent_ids=path_ent_ids, exclude=False,
                                                                 candidate_entity_pool=sampled_sub_ent_pool)
            _rep_res = replace_neg(pos_candi, neg, rep_pairs=augmented_rep_pairs)
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
            if len(neg["ent"]) > MAX_ENT_NUM_PER_SENTENCE:
                continue

            # _rep_res = replace_neg(pos_candi, neg, rep_pairs=None, out_of_domain=False)
            augmented_rep_pairs = bias_entity_generate_rep_pairs(neg, existing_ent_ids=path_ent_ids, exclude=False,
                                                                 candidate_entity_pool=sampled_sub_ent_pool)
            _rep_res, _ = context_replace_neg(pos_candi, neg, rep_pairs=augmented_rep_pairs, out_of_domain=False)
            if _rep_res:
                neg_res.extend(_rep_res)

        _sim_aug = max(len(neg_res) - _res_aug - _pos_aug, 0)

        assert len(neg_res) >= max_neg_num

        path_ent_str_ls = []
        for sent in selected_sentences.values():
            path_ent_str_ls.extend(extract_replaced_path_entity(sent))
        path_ent_str_ls = list(set(path_ent_str_ls))

        _r = random.random()
        if not remove_deduct and _r < deduct_ratio:
            # replace the mentions of non-path-entities with randomly sampled entities.
            augmented_sentences = []
            for sent in selected_sentences.values():
                augmented_rep_pairs = bias_entity_generate_rep_pairs(sent, path_ent_ids,
                                                                     candidate_entity_pool=sampled_sub_ent_pool)
                augmented_sentences.append(_replace_entities_w_str(sent, augmented_rep_pairs))

            # replace all entities except the head-tail entities in positive candidate.
            augmented_pos_candi_rep_pairs = bias_entity_generate_rep_pairs(pos_candi, path_ent_ids,
                                                                           candidate_entity_pool=sampled_sub_ent_pool)
            augmented_pos_candi = _replace_entities_w_str(pos_candi, augmented_pos_candi_rep_pairs)

            examples.append({
                # "context": context,
                # "context": add_noise_sentence(num=noise_sent_num, item=item, orig_sentences=orig_sentences, rep_pairs=None, shuffle=True),
                "context": add_noise_sentence(num=noise_sent_num, item=item, orig_sentences=augmented_sentences,
                                              rep_pairs=None, shuffle=True),
                "negative": random.sample(neg_res, max_neg_num),
                # "positive": " ".join(pos_candi["sent"]),  # We didn't augment the non-path-entities in the positive candidate anymore.
                "positive": augmented_pos_candi,
                "orig_id": item["id"],
                "pos_aug_num": _pos_aug,
                "res_aug_num": _res_aug,
                "sim_aug_num": _sim_aug,
                "path_ent_str": path_ent_str_ls
            })

        # ============= context replaced-based examples ==================== #
        if len(path_sent2rep) == 0:
            continue

        neg_ctx_sent = []
        _ctx_pos_aug = 0
        _ctx_res_aug = 0
        _ctx_sim_aug = 0

        for neg in mutual_samples + neg_candidates:
            if len(neg["ent"]) > MAX_ENT_NUM_PER_SENTENCE:
                continue

            augmented_rep_pairs = bias_entity_generate_rep_pairs(neg, existing_ent_ids=path_ent_ids, exclude=False,
                                                                 candidate_entity_pool=sampled_sub_ent_pool)
            tgt_ctx_sent_id, tgt_ctx_sent = random.choice(path_sent2rep)
            _rep_res, _rep_flag = context_replace_neg(tgt_ctx_sent, neg, rep_pairs=augmented_rep_pairs, out_of_domain=False)
            _rep_res = [(tgt_ctx_sent_id, _tmp_res) for _tmp_res in _rep_res]

            if _rep_res:
                _left_num = min(MAX_NEG_SAMPLE_NUM, len(_rep_res))
                neg_ctx_sent.extend(_rep_res[:_left_num])
                if _rep_flag == 0:
                    _ctx_pos_aug += _left_num
                else:
                    _ctx_res_aug += _left_num

            # Comment the following code to generate more examples based on the original context.
            # if len(neg_ctx_sent) >= MAX_NEG_SAMPLE_NUM:
            #     neg_ctx_sent = neg_ctx_sent[:MAX_NEG_SAMPLE_NUM]
            #     break

        while len(neg_ctx_sent) < MAX_NEG_SAMPLE_NUM:
            neg_data_item_id = random.choice(list(_all_neg_candidates.keys()))
            while neg_data_item_id == item["id"]:
                neg_data_item_id = random.choice(list(_all_neg_candidates.keys()))
            neg = random.choice(_all_neg_candidates[neg_data_item_id])

            if len(neg["ent"]) > MAX_ENT_NUM_PER_SENTENCE:
                continue

            # _rep_res = replace_neg(tgt_ctx_sent, neg, rep_pairs=None, out_of_domain=True)
            augmented_rep_pairs = bias_entity_generate_rep_pairs(neg, existing_ent_ids=path_ent_ids, exclude=False,
                                                                 candidate_entity_pool=sampled_sub_ent_pool)
            tgt_ctx_sent_id, tgt_ctx_sent = random.choice(path_sent2rep)
            _rep_res, _ = context_replace_neg(tgt_ctx_sent, neg, rep_pairs=augmented_rep_pairs, out_of_domain=False)
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
                rep_context_sent(selected_sentences, _neg_sent[0], _neg_sent[1], return_sentences=True) for _neg_sent in neg_ctx_sent
            ]
            negative_noise_context = [
                add_noise_sentence(noise_sent_num, item, neg_ctx, rep_pairs=None, shuffle=True) for neg_ctx in negative_context
            ]

            # replace the mentions of non-path-entities with randomly sampled entities.
            augmented_sentences = []
            for sent in selected_sentences.values():
                augmented_rep_pairs = bias_entity_generate_rep_pairs(sent, path_ent_ids,
                                                                     candidate_entity_pool=sampled_sub_ent_pool)
                augmented_sentences.append(_replace_entities_w_str(sent, augmented_rep_pairs))

            # replace all entities except the head-tail entities in positive candidate.
            augmented_pos_candi_rep_pairs = bias_entity_generate_rep_pairs(pos_candi, path_ent_ids,
                                                                           candidate_entity_pool=sampled_sub_ent_pool)
            augmented_pos_candi = _replace_entities_w_str(pos_candi, augmented_pos_candi_rep_pairs)

            context_examples.append({
                # "context": context,
                # "context": add_noise_sentence(num=noise_sent_num, item=item, orig_sentences=orig_sentences, rep_pairs=None, shuffle=True),
                "context": add_noise_sentence(num=noise_sent_num, item=item, orig_sentences=augmented_sentences,
                                              rep_pairs=None, shuffle=True),
                # "condition": " ".join(pos_candi["sent"]),
                "condition": augmented_pos_candi,
                # "negative_context": [rep_context_sent(selected_sentences, _neg_sent[0], _neg_sent[1]) for _neg_sent in neg_ctx_sent],
                "negative_context": negative_noise_context,
                "orig_id": item["id"],
                "pos_aug_num": _ctx_pos_aug,
                "res_aug_num": _ctx_res_aug,
                "sim_aug_num": _ctx_sim_aug,
                "path_ent_str": path_ent_str_ls,
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

            target_ent_str, target_ent_ids = sample_entity(_entity_pool, path_ent_ids | set(h_t_ent_ids), _sampled_ent_num)
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
            existing_ent_ids = path_ent_ids | set(h_t_ent_ids) | set(target_ent_ids) | {sampled_neg_candidates[0]["h"],
                                                                                        sampled_neg_candidates[0]["t"]}

            # Sample a sub-entity pool to avoid new short-cut through entity amount
            item_ent = item["entity"]
            _, sampled_sub_ent_pool_ids = sample_entity(_entity_pool, set(list(item_ent.keys())) | existing_ent_ids,
                                                        max(MAX_ENT_NUM_PER_SENTENCE, math.ceil((len(item_ent) - len(path_ent_ids)) * 1.5)))
            sampled_sub_ent_pool = {_ent_id: _entity_pool[_ent_id] for _ent_id in sampled_sub_ent_pool_ids}

            path_ent_str_ls = []
            for _, sent in selected_sentences.items():
                augmented_rep_pairs = bias_entity_generate_rep_pairs(sent, existing_ent_ids,
                                                                     candidate_entity_pool=sampled_sub_ent_pool)
                assert len(set(list(augmented_rep_pairs.keys())) & set(list(_cur_aug_rep_pairs))) == 0
                augmented_rep_pairs.update(_cur_aug_rep_pairs)
                new_sentences.append(_replace_entities_w_str(sent, augmented_rep_pairs))

                path_ent_str_ls.extend(extract_replaced_path_entity(sent, _cur_aug_rep_pairs))
            path_ent_str_ls = list(set(path_ent_str_ls))

            # new_pos_candi_sent = _replace_entities_w_str(pos_candi, _cur_aug_rep_pairs)
            new_pos_candi_sent = _replace_entities_w_str(pos_candi,
                                                         bias_entity_generate_rep_pairs(pos_candi, existing_ent_ids,
                                                                                        candidate_entity_pool=sampled_sub_ent_pool))

            # Statistics
            _res_aug = 0
            _pos_aug = 0
            _sim_aug = 0

            neg_res = []

            # Other positive candidates
            mutual_samples = [candi for candi_idx, candi in enumerate(item["pos"]) if candi_idx != pos_idx]
            for neg in mutual_samples:
                if len(neg["ent"]) > MAX_ENT_NUM_PER_SENTENCE:
                    continue

                # _rep_res = switch_replace_neg(pos_candi, neg, rep_pairs=None)  # roberta-large
                if len(neg["ent"]) > 6:
                    continue
                augmented_rep_pairs = bias_entity_generate_rep_pairs(neg, existing_ent_ids, exclude=False,
                                                                     candidate_entity_pool=sampled_sub_ent_pool)
                augmented_rep_pairs.update(_cur_aug_rep_pairs)
                _rep_res = switch_replace_neg(pos_candi, neg, rep_pairs=augmented_rep_pairs)  # albert-xx-large-v2 in server 220.
                if len(_rep_res) > 0:
                    neg_res.extend(_rep_res)
            _pos_aug += len(neg_res)

            # Add negative samples from the initial ``rest_sentences``.
            for neg in neg_candidates:
                if len(neg["ent"]) > MAX_ENT_NUM_PER_SENTENCE:
                    continue

                augmented_rep_pairs = bias_entity_generate_rep_pairs(neg, existing_ent_ids, exclude=False,
                                                                     candidate_entity_pool=sampled_sub_ent_pool)
                augmented_rep_pairs.update(_cur_aug_rep_pairs)
                _rep_res = replace_neg(pos_candi, neg, rep_pairs=augmented_rep_pairs)
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
                    if len(neg["ent"]) > MAX_ENT_NUM_PER_SENTENCE:
                        continue

                    augmented_rep_pairs = bias_entity_generate_rep_pairs(neg, existing_ent_ids,
                                                                         candidate_entity_pool=sampled_sub_ent_pool)
                    augmented_rep_pairs.update(_cur_aug_rep_pairs)
                    # Add the sampled negative candidate since it contains the replaced head/tail entity already.
                    # _rep_res = " ".join(neg["sent"])
                    _rep_res = _replace_entities_w_str(neg, augmented_rep_pairs)
                    neg_res.append(_rep_res)
                    if len(neg_res) >= MAX_NEG_SAMPLE_NUM:
                        break

            # Add simple negative samples from the ``rest_sentences`` from the sampled data item.
            if len(neg_res) < MAX_NEG_SAMPLE_NUM:
                for neg in _all_neg_candidates[_sampled_neg_item_key]:
                    if len(neg["ent"]) > MAX_ENT_NUM_PER_SENTENCE:
                        continue

                    # _rep_res = replace_neg(sampled_neg_candidates[0], neg, rep_pairs=_cur_aug_rep_pairs, out_of_domain=False)
                    augmented_rep_pairs = bias_entity_generate_rep_pairs(neg, existing_ent_ids, exclude=False,
                                                                         candidate_entity_pool=sampled_sub_ent_pool)
                    augmented_rep_pairs.update(_cur_aug_rep_pairs)
                    _rep_res, _ = context_replace_neg(sampled_neg_candidates[0], neg, rep_pairs=augmented_rep_pairs, out_of_domain=False)
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

                if len(neg["ent"]) > MAX_ENT_NUM_PER_SENTENCE:
                    continue

                augmented_rep_pairs = bias_entity_generate_rep_pairs(neg, existing_ent_ids, exclude=False,
                                                                     candidate_entity_pool=sampled_sub_ent_pool)
                augmented_rep_pairs.update(_cur_aug_rep_pairs)
                # _rep_res = replace_neg(sampled_neg_candidates[0], neg, rep_pairs=_cur_aug_rep_pairs, out_of_domain=False)
                _rep_res, _ = context_replace_neg(sampled_neg_candidates[0], neg, rep_pairs=augmented_rep_pairs, out_of_domain=False)
                if _rep_res:
                    neg_res.extend(_rep_res)

            _sim_aug += max(len(neg_res) - _pos_aug - _res_aug, 0)

            # if len(neg_res) > max_neg_num:
            #     neg_res = neg_res[:max_neg_num]
            assert len(neg_res) >= MAX_NEG_SAMPLE_NUM
            neg_res = random.sample(neg_res, max_neg_num)

            if shuffle_context:
                random.shuffle(new_sentences)
            # new_context = " ".join(new_sentences)

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
                    "path_ent_str": path_ent_str_ls,
                })

            # ============= context replaced-based examples ==================== #
            # TODO: Check这里的``rep_pairs``参数是否有问题
            if len(path_sent2rep) == 0:
                continue

            neg_ctx_sent = []
            _ctx_pos_aug = 0
            _ctx_res_aug = 0
            _ctx_sim_aug = 0

            new_pos_candi_sent = _replace_entities_w_str(pos_candi,
                                                         bias_entity_generate_rep_pairs(pos_candi, existing_ent_ids,
                                                                                        candidate_entity_pool=sampled_sub_ent_pool))

            for neg in mutual_samples + neg_candidates:
                if len(neg["ent"]) > MAX_ENT_NUM_PER_SENTENCE:
                    continue

                augmented_rep_pairs = bias_entity_generate_rep_pairs(neg, existing_ent_ids, exclude=False,
                                                                     candidate_entity_pool=sampled_sub_ent_pool)
                augmented_rep_pairs.update(_cur_aug_rep_pairs)

                tgt_ctx_sent_id, tgt_ctx_sent = random.choice(path_sent2rep)
                _rep_res, _rep_flag = context_replace_neg(tgt_ctx_sent, neg, rep_pairs=augmented_rep_pairs, out_of_domain=False)
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
                    if len(neg["ent"]) > MAX_ENT_NUM_PER_SENTENCE:
                        continue

                    tgt_ctx_sent_id, tgt_ctx_sent = random.choice(path_sent2rep)
                    # _rep_res, _ = context_replace_neg(tgt_ctx_sent, neg, rep_pairs=None, out_of_domain=True)
                    # V9_Fix: add ``_cur_aug_rep_pairs`` although ``out_of_domain=True``
                    # to replace the head/tail entity in the context sentence.
                    augmented_rep_pairs = bias_entity_generate_rep_pairs(neg, existing_ent_ids, exclude=False,
                                                                         candidate_entity_pool=sampled_sub_ent_pool)
                    augmented_rep_pairs.update(_cur_aug_rep_pairs)
                    _rep_res, _ = context_replace_neg(tgt_ctx_sent, neg, rep_pairs=augmented_rep_pairs, out_of_domain=False)
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

                if len(neg["ent"]) > MAX_ENT_NUM_PER_SENTENCE:
                    continue

                tgt_ctx_sent_id, tgt_ctx_sent = random.choice(path_sent2rep)
                # _rep_res, _ = context_replace_neg(tgt_ctx_sent, neg, rep_pairs=None, out_of_domain=True)
                # V9_Fix: add ``_cur_aug_rep_pairs`` although ``out_of_domain=True``
                # to replace the head/tail entity in the context sentence.
                augmented_rep_pairs = bias_entity_generate_rep_pairs(neg, existing_ent_ids, exclude=False,
                                                                     candidate_entity_pool=sampled_sub_ent_pool)
                augmented_rep_pairs.update(_cur_aug_rep_pairs)
                _rep_res, _ = context_replace_neg(tgt_ctx_sent, neg, rep_pairs=augmented_rep_pairs, out_of_domain=False)
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
                augmented_rep_pairs = bias_entity_generate_rep_pairs(sent, existing_ent_ids,
                                                                     candidate_entity_pool=sampled_sub_ent_pool)
                assert len(set(list(augmented_rep_pairs.keys())) & set(list(_cur_aug_rep_pairs))) == 0
                augmented_rep_pairs.update(_cur_aug_rep_pairs)
                new_sentences[s_id] = _replace_entities_w_str(sent, augmented_rep_pairs)

            negative_context = []
            for _neg_sent in neg_ctx_sent:  # 负样本应该也不需要对head-tail entity以外的entities做特殊处理？
                _new_context = copy.deepcopy(new_sentences)
                _new_context[_neg_sent[0]] = _neg_sent[1]
                _new_context_sentences = list(_new_context.values())
                if shuffle_context:
                    random.shuffle(_new_context_sentences)
                # negative_context.append(" ".join(_new_context_sentences))
                negative_context.append(add_noise_sentence(noise_sent_num, item, _new_context_sentences, rep_pairs=_cur_aug_rep_pairs,
                                                           shuffle=True))

            new_sentences = list(new_sentences.values())
            if shuffle_context:
                random.shuffle(new_sentences)
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
                    "path_ent_str": path_ent_str_ls,
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


def _annotation_init(tokenizer: PreTrainedTokenizer):
    global _tokenizer
    _tokenizer = tokenizer


def annotate_ent_str_per_example_for_tagging(example, max_seq_length: int):
    input_a = " ".join(example["context"])
    input_b = example["positive"] if "positive" in example else example["condition"]
    path_ent_str = set(example["path_ent_str"])

    max_ent_token_len = max([len(_tokenizer.tokenize(ent)) for ent in path_ent_str])

    _tokenize_kwargs = {
        "padding": PaddingStrategy.MAX_LENGTH,
        "truncation": TruncationStrategy.LONGEST_FIRST,
        "max_length": max_seq_length,
    }

    tokenizer_outputs = _tokenizer(input_a, input_b, **_tokenize_kwargs)
    tokens = _tokenizer.convert_ids_to_tokens(tokenizer_outputs["input_ids"])
    labels = [0] * len(tokens)
    # ent_str_to_pos = collections.defaultdict(list)
    all_pos = []

    for i in range(len(tokens)):
        if tokens[i] == _tokenizer.pad_token:
            break
        for j in range(i, len(tokens)):
            if j - i + 1 > max_ent_token_len + 3:  # extra length for safety.
                break
            tmp = _tokenizer.convert_tokens_to_string(tokens[i: j + 1]).strip()
            if tmp in path_ent_str:
                # ent_str_to_pos[tmp].append((i, j + 1))
                all_pos.append((i, j + 1))
                labels[i: j + 1] = [1] * (j - i + 1)

    sorted_pos = sorted(all_pos, key=lambda x: x[0])
    overlapped_cnt = 0
    # Non-overlapping check.
    for _tmp_id, _tmp in enumerate(sorted_pos):
        if _tmp_id == 0:
            continue
        if _tmp[0] >= sorted_pos[_tmp_id - 1][1]:
            # print("warning: overlapped path entity position")
            overlapped_cnt += 1

    example["tagging_label"] = labels
    example["overlapped_cnt"] = overlapped_cnt
    example["token_num"] = sum([1 if x != _tokenizer.pad_token else 0 for x in tokens])
    example["tag_num"] = sum(labels)

    return example


def convert_examples_into_features(file_path: str, tokenizer: PreTrainedTokenizer, pattern_pair_file: str,
                                   shuffle_context: bool = False, max_neg_num: int = 3, aug_num: int = 10,
                                   max_seq_length: int = 512, geo_p: float = 0.5, min_rep_num: int = 1,
                                   deduct_ratio: float = 1.0, context_ratio: float = 1.0, noise_sent_ratio: float = 0.5,
                                   remove_deduct: bool = False, remove_context: bool = False,
                                   max_neg_samples_num: int = 8, num_workers=48):
    tokenizer_name = tokenizer.__class__.__name__
    tokenizer_name = tokenizer_name.replace('TokenizerFast', '')
    tokenizer_name = tokenizer_name.replace('Tokenizer', '').lower()

    file_suffix = f"{tokenizer_name}_{shuffle_context}_{max_neg_num}_{aug_num}_" \
                  f"{max_seq_length}_{geo_p}_{min_rep_num}_" \
                  f"{deduct_ratio}_{context_ratio}_{noise_sent_ratio}_{max_neg_samples_num}_" \
                  f"{'' if not remove_context else 'no-ctx-ex_'}{'' if not remove_deduct else 'no-duc-ex_'}path_v9.2.3_split_ent"

    cached_file_path = f"{file_path}_{file_suffix}"
    if os.path.exists(cached_file_path):
        logger.info(f"Loading cached file from {cached_file_path}")
        all_examples, raw_texts = torch.load(cached_file_path)
        dataset = WikiPathDatasetV6wPatternPairFull(all_examples, raw_texts, pattern_pair_file)
        return dataset

    examples, context_examples, raw_texts = read_examples(file_path, shuffle_context=shuffle_context, max_neg_num=max_neg_num,
                                                          aug_num=aug_num, geo_p=geo_p, min_rep_num=min_rep_num,
                                                          deduct_ratio=deduct_ratio, context_ratio=context_ratio,
                                                          noise_sent_ratio=noise_sent_ratio,
                                                          remove_deduct=remove_deduct, remove_context=remove_context,
                                                          max_neg_samples_num=max_neg_samples_num,
                                                          num_workers=num_workers)
    all_examples = examples + context_examples

    with Pool(num_workers, initializer=_annotation_init, initargs=(tokenizer,)) as p:
        _annotate = partial(annotate_ent_str_per_example_for_tagging, max_seq_length=max_seq_length)
        _results = list(tqdm(
            p.imap(_annotate, all_examples, chunksize=32),
            total=len(all_examples),
            desc="Reading examples"
        ))

    overlapped_cnt = 0
    tag_ratio_per_sample = 0
    for _res in _results:
        _res_overlap = _res.pop("overlapped_cnt")
        overlapped_cnt += _res_overlap

        _token_num = _res.pop("token_num")
        _tag_num = _res.pop("tag_num")
        tag_ratio_per_sample += _tag_num / _token_num

    all_examples = _results

    logger.info(f"Overlapped instances: {overlapped_cnt} / {len(all_examples)} = {overlapped_cnt / len(all_examples)}")
    logger.info(f"Average tagged token num: {tag_ratio_per_sample} / {len(all_examples)} = {tag_ratio_per_sample / len(all_examples)}")

    # Save
    logger.info(f"Saving processed features into {cached_file_path}.")
    torch.save((all_examples, raw_texts), cached_file_path)

    return WikiPathDatasetV6wPatternPairFull(all_examples, raw_texts, pattern_pair_file)


def _quick_loading(file_path: str, pattern_pair_file: str, **kwargs):
    logger.info(f"Quickly loading cached file from {file_path}")
    all_examples, raw_texts = torch.load(file_path)
    dataset = WikiPathDatasetV6wPatternPair(all_examples, raw_texts, pattern_pair_file)
    return dataset
