import collections
import copy
import os
import pickle
import random
from tqdm import tqdm
from collections import Counter, defaultdict
from typing import Tuple, Optional, List, Dict, Any
from multiprocessing import Pool
import itertools

import torch
import transformers
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
from transformers import AutoTokenizer, PreTrainedTokenizer, T5Tokenizer, BartTokenizer
from transformers.tokenization_utils import PaddingStrategy, TruncationStrategy

from general_util.logger import get_child_logger

logger = get_child_logger("Wiki.Entity.Path")


class WikiPathDatasetV5(Dataset):
    def __init__(self, examples, raw_texts):
        self.examples = examples

        _aligned_texts = []
        while len(_aligned_texts) < len(examples):
            diff = len(examples) - len(_aligned_texts)
            if diff < len(raw_texts):
                _aligned_texts.extend(random.sample(raw_texts, diff))
            else:
                _aligned_texts.extend(raw_texts[:])
        assert len(_aligned_texts) == len(self.examples)

        # cnt = Counter(list(map(lambda x: len(x["negative"]) if "negative" in x else len(x["negative_context"]), examples)))
        # assert len(cnt) == 1, cnt

        self.raw_texts = _aligned_texts

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index) -> T_co:
        example = self.examples[index]
        text = self.raw_texts[index]
        return {
            "example": example,
            "text": text,
            "index": index,
        }


def map_counterfactual_data(examples: List[Dict[str, Any]]):
    exp_orig_id2cate = {}
    for exp in tqdm(examples, desc="Categorizing examples", total=len(examples)):
        orig_id = exp["orig_id"]
        if orig_id not in exp_orig_id2cate:
            exp_orig_id2cate[orig_id] = {
                "cf": {
                    "opt_based": [],
                    "ctx_based": [],
                },
                "non_cf": {
                    "opt_based": [],
                    "ctx_based": [],
                },
            }

        if "h" in exp:  # counterfactual example
            assert "t" in exp
            if "positive" in exp:  # option-based example
                exp_orig_id2cate[orig_id]["cf"]["opt_based"].append(exp)
            else:
                assert "negative_context" in exp
                exp_orig_id2cate[orig_id]["cf"]["ctx_based"].append(exp)
        else:  # non-counterfactual example
            if "positive" in exp:  # option-based example
                exp_orig_id2cate[orig_id]["non_cf"]["opt_based"].append(exp)
            else:
                assert "negative_context" in exp
                exp_orig_id2cate[orig_id]["non_cf"]["ctx_based"].append(exp)

    # Some mapping relations:
    #   1. option -> ctx
    #       1.1. option (non-cf) -> ctx (non-cf)
    #       1.2. option (cf)     -> ctx (cf)
    #       1.3. option (cf)     -> ctx (non-cf)
    #       1.4. option (non-cf) -> ctx (cf)
    #   2. ctx -> option
    #       2.1. ctx (non-cf) -> option (non-cf)
    #       2.2. ctx (cf)     -> option (cf)
    #       2.3. ctx (cf)     -> option (non-cf)
    #       2.4. ctx (non-cf) -> option (cf)
    # For 1.1, 1.2, 2.1, 2.2, these samples can be processed by the original collator by simply transforming into seq2seq format.
    # For 1.3, 1.4, 2.3, 2.4, we need to add paired examples.

    new_examples = []
    for orig_id in tqdm(exp_orig_id2cate, desc="Generating paired examples", total=len(exp_orig_id2cate)):
        if len(exp_orig_id2cate[orig_id]["cf"]["opt_based"]) and len(exp_orig_id2cate[orig_id]["non_cf"]["opt_based"]):
            # `context` (cf) -> `positive` (non-cf)
            base_exp = exp_orig_id2cate[orig_id]["cf"]["opt_based"][0]  # since `context` part across different samples are the same.
            for target_exp in exp_orig_id2cate[orig_id]["non_cf"]["opt_based"]:
                new_exp = copy.deepcopy(base_exp)
                new_exp["positive"] = target_exp["positive"]
                new_exp["negative"] = target_exp["negative"]
                new_examples.append(new_exp)

            # `context` (non-cf) -> `positive` (cf)
            base_exp = exp_orig_id2cate[orig_id]["non_cf"]["opt_based"][0]
            for target_exp in exp_orig_id2cate[orig_id]["cf"]["opt_based"]:
                new_exp = copy.deepcopy(base_exp)
                new_exp["positive"] = target_exp["positive"]
                new_exp["negative"] = target_exp["negative"]
                new_examples.append(new_exp)

        if len(exp_orig_id2cate[orig_id]["cf"]["ctx_based"]) and len(exp_orig_id2cate[orig_id]["non_cf"]["ctx_based"]):
            # `option` (cf) -> `context` (non-cf)
            for base_exp in exp_orig_id2cate[orig_id]["cf"]["ctx_based"]:
                target_exp = exp_orig_id2cate[orig_id]["non_cf"]["ctx_based"][0]
                new_exp = copy.deepcopy(base_exp)
                new_exp["context"] = target_exp["context"]
                new_exp["negative_context"] = target_exp["negative_context"]
                new_examples.append(new_exp)

            # `option` (non-cf) -> `context` (cf)
            for base_exp in exp_orig_id2cate[orig_id]["non_cf"]["ctx_based"]:
                target_exp = exp_orig_id2cate[orig_id]["cf"]["ctx_based"][0]
                new_exp = copy.deepcopy(base_exp)
                new_exp["context"] = target_exp["context"]
                new_exp["negative_context"] = target_exp["negative_context"]
                new_examples.append(new_exp)

    return new_examples


def _map_counterfactual_data_mp_single_process(sample: Dict):
    _new_examples = []
    if len(sample["cf"]["opt_based"]) and len(sample["non_cf"]["opt_based"]):
        # `context` (cf) -> `positive` (non-cf)
        base_exp = sample["cf"]["opt_based"][0]  # since `context` part across different samples are the same.
        for target_exp in sample["non_cf"]["opt_based"]:
            new_exp = copy.deepcopy(base_exp)
            new_exp["positive"] = target_exp["positive"]
            new_exp["negative"] = target_exp["negative"]
            _new_examples.append(new_exp)

        # `context` (non-cf) -> `positive` (cf)
        base_exp = sample["non_cf"]["opt_based"][0]
        for target_exp in sample["cf"]["opt_based"]:
            new_exp = copy.deepcopy(base_exp)
            new_exp["positive"] = target_exp["positive"]
            new_exp["negative"] = target_exp["negative"]
            _new_examples.append(new_exp)

    if len(sample["cf"]["ctx_based"]) and len(sample["non_cf"]["ctx_based"]):
        # `option` (cf) -> `context` (non-cf)
        for base_exp in sample["cf"]["ctx_based"]:
            target_exp = sample["non_cf"]["ctx_based"][0]
            new_exp = copy.deepcopy(base_exp)
            new_exp["context"] = target_exp["context"]
            new_exp["negative_context"] = target_exp["negative_context"]
            _new_examples.append(new_exp)

        # `option` (non-cf) -> `context` (cf)
        for base_exp in sample["non_cf"]["ctx_based"]:
            target_exp = sample["cf"]["ctx_based"][0]
            new_exp = copy.deepcopy(base_exp)
            new_exp["context"] = target_exp["context"]
            new_exp["negative_context"] = target_exp["negative_context"]
            _new_examples.append(new_exp)

    return _new_examples


def map_counterfactual_data_mp(examples: List[Dict[str, Any]], num_workers: int = 32):
    exp_orig_id2cate = {}
    for exp in tqdm(examples, desc="Categorizing examples", total=len(examples)):
        orig_id = exp["orig_id"]
        if orig_id not in exp_orig_id2cate:
            exp_orig_id2cate[orig_id] = {
                "cf": {
                    "opt_based": [],
                    "ctx_based": [],
                },
                "non_cf": {
                    "opt_based": [],
                    "ctx_based": [],
                },
            }

        if "h" in exp:  # counterfactual example
            assert "t" in exp
            if "positive" in exp:  # option-based example
                exp_orig_id2cate[orig_id]["cf"]["opt_based"].append(exp)
            else:
                assert "negative_context" in exp
                exp_orig_id2cate[orig_id]["cf"]["ctx_based"].append(exp)
        else:  # non-counterfactual example
            if "positive" in exp:  # option-based example
                exp_orig_id2cate[orig_id]["non_cf"]["opt_based"].append(exp)
            else:
                assert "negative_context" in exp
                exp_orig_id2cate[orig_id]["non_cf"]["ctx_based"].append(exp)

    # Some mapping relations:
    #   1. option -> ctx
    #       1.1. option (non-cf) -> ctx (non-cf)
    #       1.2. option (cf)     -> ctx (cf)
    #       1.3. option (cf)     -> ctx (non-cf)
    #       1.4. option (non-cf) -> ctx (cf)
    #   2. ctx -> option
    #       2.1. ctx (non-cf) -> option (non-cf)
    #       2.2. ctx (cf)     -> option (cf)
    #       2.3. ctx (cf)     -> option (non-cf)
    #       2.4. ctx (non-cf) -> option (cf)
    # For 1.1, 1.2, 2.1, 2.2, these samples can be processed by the original collator by simply transforming into seq2seq format.
    # For 1.3, 1.4, 2.3, 2.4, we need to add paired examples.

    with Pool(num_workers) as p:
        new_examples = list(
            tqdm(
                p.imap(_map_counterfactual_data_mp_single_process, exp_orig_id2cate.values()),
                total=len(exp_orig_id2cate),
            )
        )
    new_examples = list(itertools.chain.from_iterable(new_examples))

    return new_examples


class WikiPathDatasetV5ComplexMap(Dataset):
    """
    We add different mapping relation in this version of Dataset.
    """

    def __init__(self, examples, raw_texts):
        self.examples = examples
        num_orig_examples = len(examples)
        logger.info(f"Number of original examples: {num_orig_examples}")
        new_examples = map_counterfactual_data(examples)
        num_ext_examples = len(new_examples)
        logger.info(f"Number of extended examples: {num_ext_examples}")

        self.examples.extend(new_examples)

        _aligned_texts = []
        while len(_aligned_texts) < len(self.examples):
            diff = len(self.examples) - len(_aligned_texts)
            if diff < len(raw_texts):
                _aligned_texts.extend(random.sample(raw_texts, diff))
            else:
                _aligned_texts.extend(raw_texts[:])
        assert len(_aligned_texts) == len(self.examples)

        self.raw_texts = _aligned_texts

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index) -> T_co:
        example = self.examples[index]
        text = self.raw_texts[index]
        return {
            "example": example,
            "text": text,
            "index": index,
        }


class WikiPathDatasetV6wPatternPair(WikiPathDatasetV5):
    def __init__(self, examples, raw_texts, pattern_pair_file: str, add_cf_pair_data: bool = True):
        super().__init__(examples, raw_texts)

        self.id2exp = collections.defaultdict(list)
        for exp_id, exp in enumerate(self.examples):
            self.id2exp[exp["orig_id"]].append(exp_id)

        self.pattern_pairs = pickle.load(open(pattern_pair_file, "rb"))

        self.add_cf_pair_data = add_cf_pair_data

    def __getitem__(self, index) -> T_co:
        item = super().__getitem__(index)

        paired_example_ids = []
        if item["example"]["orig_id"] in self.pattern_pairs and len(self.pattern_pairs[item["example"]["orig_id"]]):
            paired_example_orig_ids = self.pattern_pairs[item["example"]["orig_id"]]
            for paired_exp_id in paired_example_orig_ids:
                paired_example_ids.extend(self.id2exp[paired_exp_id])
        else:
            paired_example_orig_ids = []
            if self.add_cf_pair_data:
                # If there is no paired examples, we add corresponding original examples of itself as the paired examples.
                # The only thing to check is to avoid the specific example itself.
                paired_example_ids.extend(self.id2exp[item["example"]["orig_id"]])

        paired_example_ids = list(set(paired_example_ids) - {index})

        if len(paired_example_ids) > 0:
            paired_example = self.examples[random.choice(paired_example_ids)]
        else:
            paired_example = None

        item["paired_example_orig_ids"] = paired_example_orig_ids
        item["paired_example"] = paired_example

        return item


class WikiPathDatasetV6wPatternPairFull(WikiPathDatasetV5):
    def __init__(self, examples, raw_texts, pattern_pair_file: str):
        super().__init__(examples, raw_texts)

        logger.info(f"Concatenating example sentences...")
        for exp in self.examples:
            exp["context_sentences"] = exp["context"]
            exp["context"] = " ".join(exp["context"])
            if "negative_context" in exp:
                exp["negative_context"] = [" ".join(neg_ctx) for neg_ctx in exp["negative_context"]]

        self.id2exp = collections.defaultdict(list)
        for exp_id, exp in enumerate(self.examples):
            self.id2exp[exp["orig_id"]].append(exp_id)

        pattern_pairs = pickle.load(open(pattern_pair_file, "rb"))
        pattern_pair_cnt = 0
        self.pattern_pairs = {}
        for orig_id in pattern_pairs:
            if len(pattern_pairs[orig_id]):
                self.pattern_pairs[orig_id] = pattern_pairs[orig_id]
                pattern_pair_cnt += len(pattern_pairs[orig_id])
        self.pattern_pair_keys = list(self.pattern_pairs.keys())
        logger.info(f"Pattern pair keys: {len(self.pattern_pair_keys)}")
        logger.info(f"Total pattern pairs: {pattern_pair_cnt}")

    def __getitem__(self, index) -> T_co:
        item = super().__getitem__(index)

        # paired_example_ids = []
        # if item["example"]["orig_id"] in self.pattern_pairs and len(self.pattern_pairs[item["example"]["orig_id"]]):
        #     paired_example_orig_ids = self.pattern_pairs[item["example"]["orig_id"]]
        #     for paired_exp_id in paired_example_orig_ids:
        #         paired_example_ids.extend(self.id2exp[paired_exp_id])
        # else:
        #     paired_example_orig_ids = []
        #     if self.add_cf_pair_data:
        #         # If there is no paired examples, we add corresponding original examples of itself as the paired examples.
        #         # The only thing to check is to avoid the specific example itself.
        #         paired_example_ids.extend(self.id2exp[item["example"]["orig_id"]])
        #
        # paired_example_ids = list(set(paired_example_ids) - {index})
        #
        # if len(paired_example_ids) > 0:
        #     paired_example = self.examples[random.choice(paired_example_ids)]
        # else:
        #     paired_example = None
        if index < len(self.pattern_pairs):
            _idx = index
        else:
            _idx = random.choice(list(range(len(self.pattern_pair_keys))))
        pair_example = self.examples[random.choice(self.id2exp[self.pattern_pair_keys[_idx]])]
        assert pair_example["orig_id"] == self.pattern_pair_keys[_idx]

        paired_example_orig_ids = self.pattern_pairs[pair_example["orig_id"]]
        paired_example_ids = []
        for paired_exp_orig_id in paired_example_orig_ids:
            paired_example_ids.extend(self.id2exp[paired_exp_orig_id])
        assert len(paired_example_ids)

        paired_example = self.examples[random.choice(paired_example_ids)]
        assert paired_example["orig_id"] in paired_example_orig_ids

        item["pair_q"] = pair_example
        item["pair_k_orig_ids"] = paired_example_orig_ids
        item["pair_k"] = paired_example

        return item


class WikiPathDatasetV6wPatternPairsKMeans(WikiPathDatasetV5):
    def __init__(self, examples, raw_texts, rel_path_set_file: str):
        super().__init__(examples, raw_texts)

        self.id2exp = collections.defaultdict(list)
        for exp_id, exp in enumerate(self.examples):
            self.id2exp[exp["orig_id"]].append(exp_id)
        self.exp_orig_id_set = set(list(self.id2exp.keys()))

        rel_path_set = pickle.load(open(rel_path_set_file, "rb"))
        cnt = 0
        rel_path_groups = []
        for pattern, pattern_exp_ls in rel_path_set.items():
            pattern_exp_ls = [exp_id for exp_id in pattern_exp_ls if exp_id in self.exp_orig_id_set]
            if len(pattern_exp_ls) > 1:
                rel_path_groups.append(pattern_exp_ls)
                cnt += len(pattern_exp_ls)
        self.rel_path_groups = rel_path_groups
        logger.info(f"Number of Pattern groups: {len(rel_path_groups)}")
        logger.info(f"Average number of examples per group: {cnt / len(rel_path_groups)}")

    def __getitem__(self, index) -> T_co:
        item = super().__getitem__(index)

        if index < len(self.rel_path_groups):
            _idx = index
        else:
            _idx = random.choice(list(range(len(self.rel_path_groups))))

        pair_example_orig_ids = random.sample(self.rel_path_groups[_idx], k=2)
        pair_example_q = self.examples[random.choice(self.id2exp[pair_example_orig_ids[0]])]
        pair_example_k = self.examples[random.choice(self.id2exp[pair_example_orig_ids[1]])]
        assert pair_example_q["orig_id"] in self.rel_path_groups[_idx]
        assert pair_example_k["orig_id"] in self.rel_path_groups[_idx]

        item["pair_q"] = pair_example_q
        item["pair_group_id"] = _idx
        item["pair_k"] = pair_example_k

        return item


def get_sentence_level_label(example, rel_ids):
    path_s_ids_order = example["path_s_ids_order"]
    path_s_ids = example["path_s_ids"]

    assert len(path_s_ids_order) == len(path_s_ids) == len(rel_ids["input_a"])
    s_id2rel_id = {s_id: rel_id for s_id, rel_id in zip(path_s_ids_order, rel_ids["input_a"])}
    rel_ids_input_order = [s_id2rel_id[s_id] for s_id in path_s_ids] + [rel_ids["input_b"]]
    return rel_ids_input_order


class WikiPathDatasetPatternPairKMeansLocal(WikiPathDatasetV6wPatternPairsKMeans):
    """
    Add sentence-level ids for local sentence-level contrastive learning.

    Examples come from `wiki_entity_path_v9_1_split_vqvae`.
    """

    def __init__(self, examples, raw_texts, rel_path_set_file: str, id2rel_path_decode_id_file: str, code_drop_threshold: int = 0):
        super().__init__(examples, raw_texts, rel_path_set_file)

        self.id2rel_path_decode_ids = filter_long_tail_clusters(pickle.load(open(id2rel_path_decode_id_file, "rb")), code_drop_threshold)

    def __getitem__(self, index):
        result = super().__getitem__(index)

        if result["pair_q"]["orig_id"] in self.id2rel_path_decode_ids:
            result["pair_q"]["rel_ids_input_order"] = get_sentence_level_label(result["pair_q"],
                                                                               self.id2rel_path_decode_ids[result["pair_q"]["orig_id"]])
        else:
            logger.warning(f"Not found in `id2rel_path_decode_ids` with example_id: {result['pair_q']['orig_id']}")
            result["pair_q"]["rel_ids_input_order"] = [-1] * len(result["pair_q"]["sentence_spans"])

        if result["pair_k"]["orig_id"] in self.id2rel_path_decode_ids:
            result["pair_k"]["rel_ids_input_order"] = get_sentence_level_label(result["pair_k"],
                                                                               self.id2rel_path_decode_ids[result["pair_k"]["orig_id"]])
        else:
            logger.warning(f"Not found in `id2rel_path_decode_ids` with example_id: {result['pair_k']['orig_id']}")
            result["pair_k"]["rel_ids_input_order"] = [-1] * len(result["pair_k"]["sentence_spans"])

        # assert sorted(result["pair_q"]["rel_ids_input_order"]) == sorted(result["pair_k"]["rel_ids_input_order"]), (
        #     self.id2rel_path_decode_ids[result["pair_q"]["orig_id"]],
        #     self.id2rel_path_decode_ids[result["pair_k"]["orig_id"]])
        # Here we have found an assertion error here:  ([270, 511, 550, 511], [511, 270, 550, 520]).
        # The problem is caused by that the one single relation path may have multiple outcome/direct-path.
        # So `self.id2rel_path_decode_ids[result["pair_q"]["orig_id"]]["input_b"]` can be different with
        # `self.id2rel_path_decode_ids[result["pair_k"]["orig_id"]]["input_b"]`

        return result


def filter_long_tail_clusters(id2rep_path_decode_ids, code_drop_threshold: int = 0):
    all_labels = []
    for res in id2rep_path_decode_ids.values():
        all_labels.extend(res["input_a"])
        all_labels.append(res["input_b"])

    cnt = Counter(all_labels)
    code_drop_set = set([label for label, label_num in cnt.items() if label_num <= code_drop_threshold])

    tmp = 0
    for k, v in id2rep_path_decode_ids.items():
        tmp_a = 0
        for x in id2rep_path_decode_ids[k]["input_a"] + [id2rep_path_decode_ids[k]["input_b"]]:
            if x == -1:
                tmp_a += 1

        id2rep_path_decode_ids[k]["input_a"] = [x if x not in code_drop_set else -1 for x in v["input_a"]]
        id2rep_path_decode_ids[k]["input_b"] = v["input_b"] if v["input_b"] not in code_drop_set else -1

        tmp_b = 0
        for x in id2rep_path_decode_ids[k]["input_a"] + [id2rep_path_decode_ids[k]["input_b"]]:
            if x == -1:
                tmp_b += 1

        tmp += tmp_b - tmp_a

    logger.info(f"Remove {tmp} long-tail codes.")
    return id2rep_path_decode_ids


class WikiPathDatasetRelGenerateV1(WikiPathDatasetV5):
    def __init__(self, examples, raw_texts, id2rel_path_decode_id_file: str, rel_vocab: str, code_drop_threshold: int = 0,
                 remove_cf_data_decoding: bool = False):
        super().__init__(examples, raw_texts)

        self.id2rel_path_decode_ids = filter_long_tail_clusters(pickle.load(open(id2rel_path_decode_id_file, "rb")), code_drop_threshold)
        rel_vocab = pickle.load(open(rel_vocab, "rb"))
        self.rel_vocab_size = len(set(list(rel_vocab.values())))
        self.eos_token_id = self.rel_vocab_size
        self.pad_token_id = self.rel_vocab_size + 1
        self.remove_cf_data_decoding = remove_cf_data_decoding

    def __getitem__(self, index):
        item = super().__getitem__(index)

        # if self.remove_cf_data_decoding and "rep_ent_num" in item["example"]:
        if self.remove_cf_data_decoding and "h" in item["example"]:
            item["rel_labels"] = [-1]
            return item

        example_id = item["example"]["orig_id"]

        if example_id in self.id2rel_path_decode_ids:
            path_decode_input_a = self.id2rel_path_decode_ids[example_id]["input_a"]
            path_decode_input_b = self.id2rel_path_decode_ids[example_id]["input_b"]

            # if path_decode_input_b == -1:
            #     logger.info("Inconsistency checked.")
            #     item["rel_labels"] = path_decode_input_a + [self.eos_token_id]  # as </s> token
            # else:
            item["rel_labels"] = path_decode_input_a + [path_decode_input_b, self.eos_token_id]

            if -1 in item["rel_labels"]:
                item["rel_labels"] = [-1]
        else:
            item["rel_labels"] = [-1]

        return item


class WikiPathDatasetRelGenerateV2(WikiPathDatasetV5):
    def __init__(self, examples, raw_texts, rel_vocab: str):
        super().__init__(examples, raw_texts)

        rel_vocab = pickle.load(open(rel_vocab, "rb"))
        self.rel_vocab_size = len(set(list(rel_vocab.values())))
        self.eos_token_id = self.rel_vocab_size
        self.pad_token_id = self.rel_vocab_size + 1

    def __getitem__(self, index):
        item = super().__getitem__(index)

        example = item["example"]
        rel_labels = [example["path_edge_labels"] + [example["pos_edge_label"], self.eos_token_id]]
        if "neg_edge_label" in example:
            for neg_edge_label in example["neg_edge_label"]:
                rel_labels.append(example["path_edge_labels"] + [neg_edge_label, self.eos_token_id])
        else:
            for ctx_neg_edge_label in example["neg_path_edge_labels"]:
                rel_labels.append(ctx_neg_edge_label + [example["pos_edge_label"], self.eos_token_id])

        for op_id, op_rel_labels in enumerate(rel_labels):
            if -1 in op_rel_labels:
                rel_labels[op_id] = [-1]

        item["rel_labels"] = rel_labels

        return item


def parse_wiki_path_predictions(predictions, code_drop_threshold: int = 0):
    indices = predictions["indices"]
    codes = predictions["codes"]
    codes = list(map(lambda x: x.item() if isinstance(x, torch.Tensor) else x, codes))

    code_cnt = Counter(codes)
    code_drop_set = set([code for code, code_num in code_cnt.items() if code_num <= code_drop_threshold])

    results = {}
    tmp = 0
    for index, code in zip(indices, codes):
        if code in code_drop_set:
            code = -1
            tmp += 1

        example_id, sent_type, sent_id = index.split("-")

        if example_id not in results:
            results[example_id] = {
                "path": {},
                "pos": {}
            }

        results[example_id][sent_type][sent_id] = code
    logger.info(f"Remove {tmp} long-tail codes.")
    return results


class WikiPathDatasetRelGenerateV3(WikiPathDatasetV5):
    def __init__(self, examples, raw_texts, id2rel_path_decode_id_file: str, rel_vocab_size: int = None, rel_vocab: str = None,
                 generative_order: bool = False, code_drop_threshold: int = 0, remove_cf_data_decoding: bool = False):
        super().__init__(examples, raw_texts)

        logger.info(f"{rel_vocab_size}\t{rel_vocab}\t{generative_order}\t{code_drop_threshold}\t{remove_cf_data_decoding}")

        self.id2rel_path_decode_ids = parse_wiki_path_predictions(torch.load(id2rel_path_decode_id_file, map_location="cpu"),
                                                                  code_drop_threshold=code_drop_threshold)
        if rel_vocab is not None:
            rel_vocab = torch.load(rel_vocab, map_location="cpu")
            self.rel_vocab_size = len(rel_vocab)
        else:
            assert rel_vocab_size is not None and rel_vocab_size > 0
            self.rel_vocab_size = rel_vocab_size
        self.eos_token_id = self.rel_vocab_size
        self.pad_token_id = self.rel_vocab_size + 1
        self.generative_order = generative_order
        self.remove_cf_data_decoding = remove_cf_data_decoding

    def __getitem__(self, index):
        item = super().__getitem__(index)

        example_id = item["example"]["orig_id"]

        if self.remove_cf_data_decoding and "rep_ent_num" in item["example"]:
            item["rel_labels"] = [-1]
            return item

        if example_id in self.id2rel_path_decode_ids:
            rel_labels = []
            path_s_ids = item["example"]["path_s_ids"] if not self.generative_order else item["example"]["path_s_ids_order"]

            for s_id in path_s_ids:
                if str(s_id) not in self.id2rel_path_decode_ids[example_id]["path"]:
                    rel_labels.append(-1)
                else:
                    rel_labels.append(self.id2rel_path_decode_ids[example_id]["path"][str(s_id)])

            if str(item["example"]["pos_id"]) not in self.id2rel_path_decode_ids[example_id]["pos"]:
                rel_labels.append(-1)
            else:
                rel_labels.append(self.id2rel_path_decode_ids[example_id]["pos"][str(item["example"]["pos_id"])])

            rel_labels.append(self.eos_token_id)

            if -1 in rel_labels:
                item["rel_labels"] = [-1]
            else:
                item["rel_labels"] = rel_labels
        else:
            item["rel_labels"] = [-1]

        return item


def obtain_path_id_sequence(example, id2labels):
    order = example["path_s_ids_order"]
    example_id = example["orig_id"]
    rel_labels = []

    if example_id not in id2labels:
        return [-1]

    for s_id in order:
        if str(s_id) not in id2labels[example_id]["path"]:
            rel_labels.append(-1)
        else:
            rel_labels.append(id2labels[example_id]["path"][str(s_id)])

    if str(example["pos_id"]) not in id2labels[example_id]["pos"]:
        rel_labels.append(-1)
    else:
        rel_labels.append(id2labels[example_id]["pos"][str(example["pos_id"])])

    if -1 in rel_labels:
        rel_labels = [-1]
    return rel_labels


class WikiPathDatasetPatternPairVQVAE(WikiPathDatasetV5):
    def __init__(self, examples, raw_texts, id2rel_path_decode_id_file: str):
        super().__init__(examples, raw_texts)

        self.id2exp = collections.defaultdict(list)
        for exp_id, exp in enumerate(self.examples):
            self.id2exp[exp["orig_id"]].append(exp_id)
        self.exp_orig_id_set = set(list(self.id2exp.keys()))

        self.id2rel_path_decode_ids = parse_wiki_path_predictions(torch.load(id2rel_path_decode_id_file, map_location="cpu"))

        rel_path_groups = defaultdict(set)
        for exp in self.examples:
            exp_id = exp["orig_id"]
            pattern = obtain_path_id_sequence(exp, self.id2rel_path_decode_ids)
            if pattern[0] == -1:
                continue
            pattern = "\t".join(list(map(str, pattern)))
            rel_path_groups[pattern].add(exp_id)

        self.rel_path_groups = {}
        cnt = 0
        for pattern, pattern_group in rel_path_groups.items():
            if len(pattern_group) > 1:
                self.rel_path_groups[pattern] = list(pattern_group)
                cnt += len(pattern_group)

        self.rel_path_keys = list(self.rel_path_groups.keys())

        logger.info(f"Number of Pattern groups: {len(self.rel_path_groups)}")
        logger.info(f"Average number of examples per group: {cnt / len(self.rel_path_groups)}")

    def __getitem__(self, index):
        item = super().__getitem__(index)

        if index < len(self.rel_path_keys):
            key = self.rel_path_keys[index]
        else:
            # _idx = random.choice(list(range(len(self.rel_path_groups))))
            key = random.choice(self.rel_path_keys)

        pair_example_orig_ids = random.sample(self.rel_path_groups[key], k=2)
        pair_example_q = self.examples[random.choice(self.id2exp[pair_example_orig_ids[0]])]
        pair_example_k = self.examples[random.choice(self.id2exp[pair_example_orig_ids[1]])]
        assert pair_example_q["orig_id"] in self.rel_path_groups[key]
        assert pair_example_k["orig_id"] in self.rel_path_groups[key]

        item["pair_q"] = pair_example_q
        item["pair_group_id"] = key
        item["pair_k"] = pair_example_k

        return item


class WikiPathDatasetGenerate(Dataset):
    """
    It seems that the dataset class is not relevant to ``generation``.
    """

    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index) -> T_co:
        example = self.examples[index]
        return {
            "example": example,
            "index": index,
        }


class WikiPathDatasetCollator:
    def __init__(self, max_seq_length: int, tokenizer: str, mlm_probability: float = 0.15, max_option_num: int = 4):
        self.max_seq_length = max_seq_length
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast=False)
        self.mlm_probability = mlm_probability
        self.max_option_num = max_option_num

        transformers.logging.set_verbosity_error()  # Avoid truncation warning.

    def __call__(self, batch):
        # examples, texts = list(zip(*batch))
        examples, texts = [], []
        for b in batch:
            # assert list(b.keys()) == ["example", "text"], b.keys()
            examples.append(b.pop("example"))
            texts.append(b.pop("text"))
            # assert isinstance(texts[-1], str), texts[-1]
        del batch

        sentences = []
        options = []
        for e in examples:
            op = ([e["positive"]] + e["negative"])[:self.max_option_num]
            options.extend(op)
            sentences.extend([e["context"]] * len(op))
        batch_size = len(examples)
        # option_num = len(examples[0]["negative"]) + 1
        option_num = min(len(examples[0]["negative"]) + 1, self.max_option_num)

        tokenizer_outputs = self.tokenizer(sentences, options, padding=PaddingStrategy.MAX_LENGTH,
                                           truncation=TruncationStrategy.LONGEST_FIRST, max_length=self.max_seq_length,
                                           return_tensors="pt")
        input_ids = tokenizer_outputs["input_ids"]
        attention_mask = tokenizer_outputs["attention_mask"]

        mlm_tokenize_outputs = self.tokenizer(texts, padding=PaddingStrategy.MAX_LENGTH,
                                              truncation=TruncationStrategy.LONGEST_FIRST, max_length=self.max_seq_length,
                                              return_tensors="pt")
        mlm_input_ids = mlm_tokenize_outputs["input_ids"]
        mlm_attention_mask = mlm_tokenize_outputs["attention_mask"]

        mlm_input_ids, mlm_labels = self.mask_tokens(mlm_input_ids)

        res = {
            "input_ids": input_ids.reshape(batch_size, option_num, self.max_seq_length),
            "attention_mask": attention_mask.reshape(batch_size, option_num, self.max_seq_length),
            "labels": torch.zeros(batch_size, dtype=torch.long),
            "mlm_input_ids": mlm_input_ids,
            "mlm_attention_mask": mlm_attention_mask,
            "mlm_labels": mlm_labels
        }
        if "token_type_ids" in tokenizer_outputs:
            res["token_type_ids"] = tokenizer_outputs["token_type_ids"]
        return res

    def mask_tokens(
            self, inputs: torch.Tensor, special_tokens_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
            # Remove padding.
            special_tokens_mask = special_tokens_mask | (labels == self.tokenizer.pad_token_id)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -1  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels


class WikiPathDatasetCollatorOnlyMLM(WikiPathDatasetCollator):
    def __call__(self, batch):
        texts = []
        for b in batch:
            texts.append(b.pop("text"))
        del batch

        mlm_tokenize_outputs = self.tokenizer(texts, padding=PaddingStrategy.MAX_LENGTH,
                                              truncation=TruncationStrategy.LONGEST_FIRST, max_length=self.max_seq_length,
                                              return_tensors="pt")
        mlm_input_ids = mlm_tokenize_outputs["input_ids"]
        mlm_attention_mask = mlm_tokenize_outputs["attention_mask"]

        mlm_input_ids, mlm_labels = self.mask_tokens(mlm_input_ids)

        return {
            "input_ids": mlm_input_ids,
            "attention_mask": mlm_attention_mask,
            "labels": mlm_labels
        }


class WikiPathDatasetCollatorWithContext(WikiPathDatasetCollator):
    def __init__(self, max_seq_length: int, tokenizer: str, mlm_probability: float = 0.15, max_option_num: int = 4, swap: bool = False):
        super().__init__(max_seq_length, tokenizer, mlm_probability, max_option_num)
        self.swap = swap

    def __call__(self, batch):
        # FIXME:
        #   这里的代码有问题，所有继承该类的代码需要重新检查（主要涉及path generation相关的code中，通过super()调用__call__()方法的代码段）
        #   错误的原因在于，这里对batch内部的顺序做了重新排列，导致后续按照原有的顺序构造label的时候对应不上（主要是生成任务）
        # examples, texts = list(zip(*batch))
        # op_examples, ctx_examples, texts = [], [], []
        # for b in batch:
        #     example = b.pop("example")
        #     if "negative_context" in example:
        #         ctx_examples.append(example)
        #     else:
        #         op_examples.append(example)
        #     # examples.append(b.pop("example"))
        #     texts.append(b.pop("text"))
        #     # assert isinstance(texts[-1], str), texts[-1]
        # del batch
        # batch_size = len(op_examples) + len(ctx_examples)
        # assert batch_size == len(texts)
        #
        # # TODO: Check if other possible input formats are ok, e.g., <rest context> <sep> <pseudo/ground truth edge> <sep> <sep> <option>
        #
        # input_a = []
        # input_b = []
        # option_num = -1
        #
        # for e in op_examples:
        #     op = ([e["positive"]] + e["negative"])[:self.max_option_num]
        #     if self.swap:
        #         input_a.extend([e["context"]] * len(op))
        #         input_b.extend(op)
        #     else:
        #         input_a.extend(op)
        #         input_b.extend([e["context"]] * len(op))
        #     if option_num == -1:
        #         option_num = len(op)
        #     else:
        #         assert option_num == len(op)
        #
        # for e in ctx_examples:
        #     positive_context = e.pop("context")
        #     negative_context = e.pop("negative_context")
        #     op = e.pop("condition")
        #     input_a.extend([positive_context] + negative_context)
        #     input_b.extend([op] * (len(negative_context) + 1))
        #     if option_num == -1:
        #         option_num = len(negative_context) + 1
        #     else:
        #         assert option_num == len(negative_context) + 1, (option_num, len(negative_context))

        # FIXED by 2022/11/27
        input_a, input_b, texts = [], [], []
        for b in batch:
            example = b.pop("example")
            if "negative_context" in example:
                positive_context = example.pop("context")
                negative_context = example.pop("negative_context")
                op = example.pop("condition")
                input_a.extend([positive_context] + negative_context)
                input_b.extend([op] * (len(negative_context) + 1))
                assert self.max_option_num == len(negative_context) + 1, len(negative_context)
            else:
                op = ([example["positive"]] + example["negative"])[:self.max_option_num]
                if self.swap:
                    input_a.extend([example["context"]] * len(op))
                    input_b.extend(op)
                else:
                    input_a.extend(op)
                    input_b.extend([example["context"]] * len(op))

            texts.append(b.pop("text"))
        batch_size = len(texts)
        option_num = self.max_option_num

        tokenizer_outputs = self.tokenizer(input_a, input_b, padding=PaddingStrategy.LONGEST,
                                           truncation=TruncationStrategy.LONGEST_FIRST, max_length=self.max_seq_length,
                                           return_tensors="pt")
        input_ids = tokenizer_outputs["input_ids"]
        attention_mask = tokenizer_outputs["attention_mask"]

        mlm_tokenize_outputs = self.tokenizer(texts, padding=PaddingStrategy.LONGEST,
                                              truncation=TruncationStrategy.LONGEST_FIRST, max_length=self.max_seq_length,
                                              return_tensors="pt")
        mlm_input_ids = mlm_tokenize_outputs["input_ids"]
        mlm_attention_mask = mlm_tokenize_outputs["attention_mask"]

        mlm_input_ids, mlm_labels = self.mask_tokens(mlm_input_ids)
        seq_len = input_ids.size(1)

        res = {
            "input_ids": input_ids.reshape(batch_size, option_num, seq_len),
            "attention_mask": attention_mask.reshape(batch_size, option_num, seq_len),
            "labels": torch.zeros(batch_size, dtype=torch.long),
            "mlm_input_ids": mlm_input_ids,
            "mlm_attention_mask": mlm_attention_mask,
            "mlm_labels": mlm_labels
        }
        if "token_type_ids" in tokenizer_outputs:
            res["token_type_ids"] = tokenizer_outputs["token_type_ids"]
        return res


class WikiPathDatasetCollatorWithContextAndPair(WikiPathDatasetCollator):
    def __init__(self, max_seq_length: int, tokenizer: str, mlm_probability: float = 0.15, max_option_num: int = 4, swap: bool = False):
        super().__init__(max_seq_length, tokenizer, mlm_probability, max_option_num)
        self.swap = swap

    def prepare_single_example(self, item):
        input_a = []
        input_b = []
        if "negative_context" in item["example"]:
            input_a.extend(([item["example"]["context"]] + item["example"]["negative_context"])[:self.max_option_num])
            input_b.extend([item["example"]["condition"]] * self.max_option_num)
        else:
            op = ([item["example"]["positive"]] + item["example"]["negative"])[:self.max_option_num]
            if self.swap:
                input_a.extend([item["example"]["context"]] * len(op))
                input_b.extend(op)
            else:
                input_a.extend(op)
                input_b.extend([item["example"]["context"]] * len(op))

        assert len(input_a) == len(input_b)

        paired_input_a = self.tokenizer.pad_token
        paired_input_b = self.tokenizer.pad_token
        paired_example = item["paired_example"]
        paired_example_id = -1
        if paired_example is not None:
            paired_example_id = paired_example["orig_id"]
            if "negative_context" in paired_example:
                paired_input_a = paired_example["context"]
                paired_input_b = paired_example["condition"]
            else:
                if self.swap:
                    paired_input_a = paired_example["context"]
                    paired_input_b = paired_example["positive"]
                else:
                    paired_input_a = paired_example["positive"]
                    paired_input_b = paired_example["context"]

        return input_a, input_b, paired_input_a, paired_input_b, item["example"]["orig_id"], paired_example_id, \
            item["paired_example_orig_ids"]

    def __call__(self, batch):
        input_a, input_b, texts = [], [], []
        paired_input_a, paired_input_b, orig_ids, paired_orig_ids, paired_orig_id_list = [], [], [], [], []
        for b in batch:
            b_input_a, b_input_b, b_paired_input_a, b_paired_input_b, b_orig_id, b_paired_orig_id, b_pair_id_list = \
                self.prepare_single_example(b)

            input_a.extend(b_input_a)
            input_b.extend(b_input_b)
            paired_input_a.append(b_paired_input_a)
            paired_input_b.append(b_paired_input_b)
            orig_ids.append(b_orig_id)
            paired_orig_ids.append(b_paired_orig_id)
            paired_orig_id_list.append(b_pair_id_list)
            texts.append(b["text"])

        del batch

        tokenizer_outputs = self.tokenizer(input_a, input_b, padding=PaddingStrategy.LONGEST,
                                           truncation=TruncationStrategy.LONGEST_FIRST, max_length=self.max_seq_length,
                                           return_tensors="pt")
        input_ids = tokenizer_outputs["input_ids"]
        attention_mask = tokenizer_outputs["attention_mask"]

        pair_tokenizer_outputs = self.tokenizer(paired_input_a, paired_input_b, padding=PaddingStrategy.LONGEST,
                                                truncation=TruncationStrategy.LONGEST_FIRST, max_length=self.max_seq_length,
                                                return_tensors="pt")
        pair_input_ids = pair_tokenizer_outputs["input_ids"]
        pair_attention_mask = pair_tokenizer_outputs["attention_mask"]

        # prepare pair labels and mask
        pair_align_mask = []  # `1` for mask and `0` for true value.
        pair_align_labels = []  # `-1` for non-pair examples.
        assert len(paired_orig_id_list) == len(orig_ids) == len(paired_orig_ids)
        for a_id, (a_orig_id, a_paired_orig_id_list) in enumerate(zip(orig_ids, paired_orig_id_list)):
            # pair dot product
            pair_mask = []
            for b_id, b_paired_orig_id in enumerate(paired_orig_ids):
                if a_id == b_id:
                    if b_paired_orig_id != -1:
                        pair_align_labels.append(b_id)
                        assert b_paired_orig_id in a_paired_orig_id_list or b_paired_orig_id == a_orig_id
                    else:
                        pair_align_labels.append(-1)
                    pair_mask.append(0)
                else:
                    if b_paired_orig_id in a_paired_orig_id_list or b_paired_orig_id == -1 or b_paired_orig_id == a_orig_id:
                        pair_mask.append(1)
                    else:
                        pair_mask.append(0)

            # self dot product
            for c_id, c_orig_id in enumerate(orig_ids):
                if c_id == a_id:
                    pair_mask.append(1)
                else:
                    if c_orig_id in a_paired_orig_id_list or c_orig_id == a_orig_id:
                        pair_mask.append(1)
                    else:
                        pair_mask.append(0)

            # assert sum(pair_mask) + 2 <= len(pair_mask), pair_mask
            assert len(pair_mask) == len(orig_ids) * 2
            pair_align_mask.append(pair_mask)

        pair_align_mask = torch.tensor(pair_align_mask, dtype=torch.int)
        pair_align_labels = torch.tensor(pair_align_labels, dtype=torch.long)

        mlm_tokenize_outputs = self.tokenizer(texts, padding=PaddingStrategy.LONGEST,
                                              truncation=TruncationStrategy.LONGEST_FIRST, max_length=self.max_seq_length,
                                              return_tensors="pt")
        mlm_input_ids = mlm_tokenize_outputs["input_ids"]
        mlm_attention_mask = mlm_tokenize_outputs["attention_mask"]

        mlm_input_ids, mlm_labels = self.mask_tokens(mlm_input_ids)

        batch_size = len(pair_input_ids)
        option_num = self.max_option_num

        res = {
            "input_ids": input_ids.reshape(batch_size, option_num, -1),
            "attention_mask": attention_mask.reshape(batch_size, option_num, -1),
            "labels": torch.zeros(batch_size, dtype=torch.long),
            "mlm_input_ids": mlm_input_ids,
            "mlm_attention_mask": mlm_attention_mask,
            "mlm_labels": mlm_labels,
            "pair_input_ids": pair_input_ids,
            "pair_attention_mask": pair_attention_mask,
            "pair_mask": pair_align_mask,
            "pair_labels": pair_align_labels,
            "pair_label_num": (pair_align_labels > -1).sum(),
            "pair_value_num": (1 - pair_align_mask).sum(dim=-1)[pair_align_labels > -1].sum() / batch_size,
        }
        if "token_type_ids" in tokenizer_outputs:
            res["token_type_ids"] = tokenizer_outputs["token_type_ids"]
        return res


class WikiPathDatasetCollatorWithContextAndPairComplete(WikiPathDatasetCollator):
    def __init__(self, max_seq_length: int, tokenizer: str, mlm_probability: float = 0.15, max_option_num: int = 4, swap: bool = False,
                 option_dropout: float = 0.0, k_option_dropout: float = 0.0):
        super().__init__(max_seq_length, tokenizer, mlm_probability, max_option_num)
        self.swap = swap
        self.option_dropout = option_dropout
        self.k_option_dropout = k_option_dropout

    @staticmethod
    def prepare_single_example_positive(example):
        if "negative_context" in example:
            input_a = example["context"]
            input_b = example["condition"]
        else:
            # Remove `swap` for easy implementation of `option` dropout.
            # if self.swap:
            #     input_a = example["context"]
            #     input_b = example["positive"]
            # else:
            #     input_a = example["positive"]
            #     input_b = example["context"]
            input_a = example["context"]
            input_b = example["positive"]
        return input_a, input_b

    def prepare_single_example(self, item):
        input_a = []
        input_b = []
        if "negative_context" in item["example"]:
            input_a.extend(([item["example"]["context"]] + item["example"]["negative_context"])[:self.max_option_num])
            input_b.extend([item["example"]["condition"]] * self.max_option_num)
        else:
            op = ([item["example"]["positive"]] + item["example"]["negative"])[:self.max_option_num]
            if self.swap:
                input_a.extend([item["example"]["context"]] * len(op))
                input_b.extend(op)
            else:
                input_a.extend(op)
                input_b.extend([item["example"]["context"]] * len(op))

        assert len(input_a) == len(input_b) == self.max_option_num

        pair_q_input_a, pair_q_input_b = self.prepare_single_example_positive(item["pair_q"])
        pair_k_input_a, pair_k_input_b = self.prepare_single_example_positive(item["pair_k"])

        return input_a, input_b, item["pair_q"]["orig_id"], pair_q_input_a, pair_q_input_b, \
            item["pair_k"]["orig_id"], pair_k_input_a, pair_k_input_b, \
            item["pair_k_orig_ids"]

    def __call__(self, batch):
        input_a, input_b, texts = [], [], []
        pair_q_orig_ids, pair_q_a, pair_q_b, pair_k_orig_ids, pair_k_a, pair_k_b, all_paired_orig_ids = [], [], [], [], [], [], []
        dropped_op_cnt = 0
        for b in batch:
            b_input_a, b_input_b, b_pair_q_orig_id, b_pair_q_a, b_pair_q_b, \
                b_pair_k_orig_id, b_pair_k_a, b_pair_k_b, \
                b_pair_k_orig_ids = self.prepare_single_example(b)

            input_a.extend(b_input_a)
            input_b.extend(b_input_b)
            pair_q_orig_ids.append(b_pair_q_orig_id)
            pair_q_a.append(b_pair_q_a)

            _r = random.random()
            if _r < self.option_dropout:
                pair_q_b.append("")
                dropped_op_cnt += 1
            else:
                pair_q_b.append(b_pair_q_b)

            pair_k_orig_ids.append(b_pair_k_orig_id)
            pair_k_a.append(b_pair_k_a)

            _r = random.random()
            if _r < self.k_option_dropout:
                pair_k_b.append("")
            else:
                pair_k_b.append(b_pair_k_b)

            all_paired_orig_ids.append(b_pair_k_orig_ids)
            texts.append(b["text"])

        del batch
        _tokenize_kwargs = {
            "padding": PaddingStrategy.LONGEST,
            "truncation": TruncationStrategy.LONGEST_FIRST,
            "max_length": self.max_seq_length,
            "return_tensors": "pt"
        }

        tokenizer_outputs = self.tokenizer(input_a, input_b, **_tokenize_kwargs)
        input_ids = tokenizer_outputs["input_ids"]
        attention_mask = tokenizer_outputs["attention_mask"]

        pair_q_tokenizer_outputs = self.tokenizer(pair_q_a, pair_q_b, **_tokenize_kwargs)
        pair_q_input_ids = pair_q_tokenizer_outputs["input_ids"]
        pair_q_attention_mask = pair_q_tokenizer_outputs["attention_mask"]

        pair_k_tokenizer_outputs = self.tokenizer(pair_k_a, pair_k_b, **_tokenize_kwargs)
        pair_k_input_ids = pair_k_tokenizer_outputs["input_ids"]
        pair_k_attention_mask = pair_k_tokenizer_outputs["attention_mask"]

        # prepare pair labels and mask
        pair_align_mask = []  # `1` for mask and `0` for true value.
        pair_align_labels = []  # `-1` for non-pair examples.
        assert len(pair_q_orig_ids) == len(pair_k_orig_ids) == len(all_paired_orig_ids)
        for q_id, (q_orig_id, q_paired_orig_id_list) in enumerate(zip(pair_q_orig_ids, all_paired_orig_ids)):
            # pair dot product
            pair_mask = []
            for k_id, k_orig_id in enumerate(pair_k_orig_ids):
                if q_id == k_id:
                    assert k_orig_id in q_paired_orig_id_list
                    pair_mask.append(0)
                else:
                    if k_orig_id in q_paired_orig_id_list or k_orig_id == q_orig_id:
                        pair_mask.append(1)
                    else:
                        pair_mask.append(0)
            if sum(pair_mask) + 1 == len(pair_k_orig_ids):
                pair_align_labels.append(-1)
            else:
                pair_align_labels.append(q_id)

            # TODO: Add self-product for more negative samples? Then the option dropout should be modified.

            pair_align_mask.append(pair_mask)

        pair_align_mask = torch.tensor(pair_align_mask, dtype=torch.int)
        pair_align_labels = torch.tensor(pair_align_labels, dtype=torch.long)
        num_labels = (pair_align_labels > -1).sum()

        mlm_tokenize_outputs = self.tokenizer(texts, **_tokenize_kwargs)
        mlm_input_ids = mlm_tokenize_outputs["input_ids"]
        mlm_attention_mask = mlm_tokenize_outputs["attention_mask"]

        mlm_input_ids, mlm_labels = self.mask_tokens(mlm_input_ids)

        batch_size = len(pair_q_input_ids)
        option_num = self.max_option_num

        res = {
            "input_ids": input_ids.reshape(batch_size, option_num, -1),
            "attention_mask": attention_mask.reshape(batch_size, option_num, -1),
            "labels": torch.zeros(batch_size, dtype=torch.long),
            "mlm_input_ids": mlm_input_ids,
            "mlm_attention_mask": mlm_attention_mask,
            "mlm_labels": mlm_labels,
            "pair_q_input_ids": pair_q_input_ids,
            "pair_q_attention_mask": pair_q_attention_mask,
            "pair_k_input_ids": pair_k_input_ids,
            "pair_k_attention_mask": pair_k_attention_mask,
            "pair_mask": pair_align_mask,
            "pair_labels": pair_align_labels,
            "pair_label_num": num_labels,
            "pair_value_num": (1 - pair_align_mask).sum(dim=-1)[pair_align_labels > -1].sum() / num_labels,
            "dropped_op_cnt": torch.tensor([dropped_op_cnt])
        }
        if "token_type_ids" in tokenizer_outputs:
            res["token_type_ids"] = tokenizer_outputs["token_type_ids"]
        # print("pair_value_num", res["pair_value_num"])
        return res


class WikiPathDatasetCollatorWithContextAndPairCompleteKMeans(WikiPathDatasetCollatorWithContextAndPairComplete):
    def __init__(self, max_seq_length: int, tokenizer: str, mlm_probability: float = 0.15, max_option_num: int = 4, swap: bool = False,
                 option_dropout: float = 0.0, k_option_dropout: float = 0.0):
        super().__init__(max_seq_length, tokenizer, mlm_probability, max_option_num, swap, option_dropout, k_option_dropout)

    def prepare_single_example(self, item):
        input_a = []
        input_b = []
        if "negative_context" in item["example"]:
            input_a.extend(([item["example"]["context"]] + item["example"]["negative_context"])[:self.max_option_num])
            input_b.extend([item["example"]["condition"]] * self.max_option_num)
        else:
            op = ([item["example"]["positive"]] + item["example"]["negative"])[:self.max_option_num]
            if self.swap:
                input_a.extend([item["example"]["context"]] * len(op))
                input_b.extend(op)
            else:
                input_a.extend(op)
                input_b.extend([item["example"]["context"]] * len(op))

        assert len(input_a) == len(input_b) == self.max_option_num

        pair_q_input_a, pair_q_input_b = self.prepare_single_example_positive(item["pair_q"])
        pair_k_input_a, pair_k_input_b = self.prepare_single_example_positive(item["pair_k"])

        return input_a, input_b, pair_q_input_a, pair_q_input_b, pair_k_input_a, pair_k_input_b, item["pair_group_id"]

    def __call__(self, batch):
        input_a, input_b, texts = [], [], []
        pair_q_a, pair_q_b, pair_k_a, pair_k_b, pair_group_ids = [], [], [], [], []
        dropped_op_cnt = 0
        for b in batch:
            b_input_a, b_input_b, b_pair_q_a, b_pair_q_b, b_pair_k_a, b_pair_k_b, b_pair_group_id = self.prepare_single_example(b)

            input_a.extend(b_input_a)
            input_b.extend(b_input_b)

            pair_q_a.append(b_pair_q_a)

            _r = random.random()
            if _r < self.option_dropout:
                pair_q_b.append("")
                dropped_op_cnt += 1
            else:
                pair_q_b.append(b_pair_q_b)

            pair_k_a.append(b_pair_k_a)

            _r = random.random()
            if _r < self.k_option_dropout:
                pair_k_b.append("")
            else:
                pair_k_b.append(b_pair_k_b)

            pair_group_ids.append(b_pair_group_id)
            texts.append(b["text"])

        del batch
        _tokenize_kwargs = {
            "padding": PaddingStrategy.LONGEST,
            "truncation": TruncationStrategy.LONGEST_FIRST,
            "max_length": self.max_seq_length,
            "return_tensors": "pt"
        }

        tokenizer_outputs = self.tokenizer(input_a, input_b, **_tokenize_kwargs)
        input_ids = tokenizer_outputs["input_ids"]
        attention_mask = tokenizer_outputs["attention_mask"]

        pair_q_tokenizer_outputs = self.tokenizer(pair_q_a, pair_q_b, **_tokenize_kwargs)
        pair_q_input_ids = pair_q_tokenizer_outputs["input_ids"]
        pair_q_attention_mask = pair_q_tokenizer_outputs["attention_mask"]

        pair_k_tokenizer_outputs = self.tokenizer(pair_k_a, pair_k_b, **_tokenize_kwargs)
        pair_k_input_ids = pair_k_tokenizer_outputs["input_ids"]
        pair_k_attention_mask = pair_k_tokenizer_outputs["attention_mask"]

        # prepare pair labels and mask
        pair_align_mask = []  # `1` for mask and `0` for true value.
        pair_align_labels = []  # `-1` for non-pair examples.
        for q_id, q_group_id in enumerate(pair_group_ids):
            # pair dot product
            pair_mask = []
            for k_id, k_group_id in enumerate(pair_group_ids):
                if q_id == k_id:
                    assert q_group_id == k_group_id
                    pair_mask.append(0)
                else:
                    if q_group_id == k_group_id:
                        pair_mask.append(1)
                    else:
                        pair_mask.append(0)
            if sum(pair_mask) + 1 == len(pair_group_ids):
                pair_align_labels.append(-1)
            else:
                pair_align_labels.append(q_id)

            pair_align_mask.append(pair_mask)

        pair_align_mask = torch.tensor(pair_align_mask, dtype=torch.int)
        pair_align_labels = torch.tensor(pair_align_labels, dtype=torch.long)
        num_labels = (pair_align_labels > -1).sum()

        mlm_tokenize_outputs = self.tokenizer(texts, **_tokenize_kwargs)
        mlm_input_ids = mlm_tokenize_outputs["input_ids"]
        mlm_attention_mask = mlm_tokenize_outputs["attention_mask"]

        mlm_input_ids, mlm_labels = self.mask_tokens(mlm_input_ids)

        batch_size = len(pair_q_input_ids)
        option_num = self.max_option_num

        res = {
            "input_ids": input_ids.reshape(batch_size, option_num, -1),
            "attention_mask": attention_mask.reshape(batch_size, option_num, -1),
            "labels": torch.zeros(batch_size, dtype=torch.long),
            "mlm_input_ids": mlm_input_ids,
            "mlm_attention_mask": mlm_attention_mask,
            "mlm_labels": mlm_labels,
            "pair_q_input_ids": pair_q_input_ids,
            "pair_q_attention_mask": pair_q_attention_mask,
            "pair_k_input_ids": pair_k_input_ids,
            "pair_k_attention_mask": pair_k_attention_mask,
            "pair_mask": pair_align_mask,
            "pair_labels": pair_align_labels,
            "pair_label_num": num_labels,
            "pair_value_num": (1 - pair_align_mask).sum(dim=-1)[pair_align_labels > -1].sum() / num_labels,
            "dropped_op_cnt": torch.tensor([dropped_op_cnt])
        }
        if "token_type_ids" in tokenizer_outputs:
            res["token_type_ids"] = tokenizer_outputs["token_type_ids"]
        return res


def sentence_spans_list_to_tensor(examples):
    max_sent_num = max(map(lambda x: len(x["sentence_spans"]), examples))
    max_sent_len = 0
    for exp in examples:
        max_sent_len = max(max_sent_len, max(map(lambda x: x[1] - x[0], exp["sentence_spans"])))

    sent_token_index = torch.zeros(len(examples), max_sent_num, max_sent_len, dtype=torch.long)
    sent_token_mask = torch.zeros(len(examples), max_sent_num, max_sent_len)
    sent_index = []
    for exp_id, exp in enumerate(examples):
        assert len(exp["sentence_spans"]) == len(exp["rel_ids_input_order"]) or (exp["rel_ids_input_order"] == [-1])
        for s_id, (s, e) in enumerate(exp["sentence_spans"]):
            sent_token_index[exp_id, s_id, :(e - s)] = torch.arange(s, e, dtype=torch.long)
            sent_token_mask[exp_id, s_id, :(e - s)] = 1
            sent_index.append(exp_id * max_sent_num + s_id)

    sent_index = torch.tensor(sent_index, dtype=torch.long)  # [batch_sent_num], used for torch.gather

    return sent_token_index, sent_token_mask, sent_index


class WikiPathDatasetCollatorWithContextAndPairCompleteKMeansLocal(WikiPathDatasetCollatorWithContextAndPairCompleteKMeans):
    def __init__(self, max_seq_length: int, tokenizer: str, mlm_probability: float = 0.15, max_option_num: int = 4, swap: bool = False,
                 option_dropout: float = 0.0, k_option_dropout: float = 0.0):
        super().__init__(max_seq_length, tokenizer, mlm_probability, max_option_num, swap, option_dropout, k_option_dropout)

    def previous_forward(self, batch):
        input_a, input_b, texts = [], [], []
        pair_q_a, pair_q_b, pair_k_a, pair_k_b, pair_group_ids = [], [], [], [], []
        pair_q_orig_b, pair_k_orig_b = [], []
        dropped_op_cnt = 0
        for b in batch:
            b_input_a, b_input_b, b_pair_q_a, b_pair_q_b, b_pair_k_a, b_pair_k_b, b_pair_group_id = self.prepare_single_example(b)

            input_a.extend(b_input_a)
            input_b.extend(b_input_b)

            pair_q_a.append(b_pair_q_a)

            _r = random.random()
            if _r < self.option_dropout:
                pair_q_b.append("")
                dropped_op_cnt += 1
            else:
                pair_q_b.append(b_pair_q_b)
            pair_q_orig_b.append(b_pair_q_b)

            pair_k_a.append(b_pair_k_a)

            _r = random.random()
            if _r < self.k_option_dropout:
                pair_k_b.append("")
            else:
                pair_k_b.append(b_pair_k_b)
            pair_k_orig_b.append(b_pair_k_b)

            pair_group_ids.append(b_pair_group_id)
            texts.append(b["text"])

        del batch
        _tokenize_kwargs = {
            "padding": PaddingStrategy.LONGEST,
            "truncation": TruncationStrategy.LONGEST_FIRST,
            "max_length": self.max_seq_length,
            "return_tensors": "pt"
        }

        tokenizer_outputs = self.tokenizer(input_a, input_b, **_tokenize_kwargs)
        input_ids = tokenizer_outputs["input_ids"]
        attention_mask = tokenizer_outputs["attention_mask"]

        pair_q_tokenizer_outputs = self.tokenizer(pair_q_a, pair_q_b, **_tokenize_kwargs)
        pair_q_input_ids = pair_q_tokenizer_outputs["input_ids"]
        pair_q_attention_mask = pair_q_tokenizer_outputs["attention_mask"]

        pair_k_tokenizer_outputs = self.tokenizer(pair_k_a, pair_k_b, **_tokenize_kwargs)
        pair_k_input_ids = pair_k_tokenizer_outputs["input_ids"]
        pair_k_attention_mask = pair_k_tokenizer_outputs["attention_mask"]

        # prepare pair labels and mask
        pair_align_mask = []  # `1` for mask and `0` for true value.
        pair_align_labels = []  # `-1` for non-pair examples.
        for q_id, q_group_id in enumerate(pair_group_ids):
            # pair dot product
            pair_mask = []
            for k_id, k_group_id in enumerate(pair_group_ids):
                if q_id == k_id:
                    assert q_group_id == k_group_id
                    pair_mask.append(0)
                else:
                    if q_group_id == k_group_id:
                        pair_mask.append(1)
                    else:
                        pair_mask.append(0)
            if sum(pair_mask) + 1 == len(pair_group_ids):
                pair_align_labels.append(-1)
            else:
                pair_align_labels.append(q_id)

            pair_align_mask.append(pair_mask)

        pair_align_mask = torch.tensor(pair_align_mask, dtype=torch.int)
        pair_align_labels = torch.tensor(pair_align_labels, dtype=torch.long)
        num_labels = (pair_align_labels > -1).sum()

        mlm_tokenize_outputs = self.tokenizer(texts, **_tokenize_kwargs)
        mlm_input_ids = mlm_tokenize_outputs["input_ids"]
        mlm_attention_mask = mlm_tokenize_outputs["attention_mask"]

        mlm_input_ids, mlm_labels = self.mask_tokens(mlm_input_ids)

        batch_size = len(pair_q_input_ids)
        option_num = self.max_option_num

        res = {
            "input_ids": input_ids.reshape(batch_size, option_num, -1),
            "attention_mask": attention_mask.reshape(batch_size, option_num, -1),
            "labels": torch.zeros(batch_size, dtype=torch.long),
            "mlm_input_ids": mlm_input_ids,
            "mlm_attention_mask": mlm_attention_mask,
            "mlm_labels": mlm_labels,
            "pair_q_input_ids": pair_q_input_ids,
            "pair_q_attention_mask": pair_q_attention_mask,
            "pair_k_input_ids": pair_k_input_ids,
            "pair_k_attention_mask": pair_k_attention_mask,
            "pair_mask": pair_align_mask,
            "pair_labels": pair_align_labels,
            "pair_label_num": num_labels,
            "pair_value_num": (1 - pair_align_mask).sum(dim=-1)[pair_align_labels > -1].sum() / num_labels,
            "dropped_op_cnt": torch.tensor([dropped_op_cnt]),
        }
        # if "token_type_ids" in tokenizer_outputs:
        #     res["token_type_ids"] = tokenizer_outputs["token_type_ids"]

        if self.option_dropout:
            pair_q_orig_outputs = self.tokenizer(pair_q_a, pair_q_orig_b, **_tokenize_kwargs)
            pair_q_orig_input_ids = pair_q_orig_outputs["input_ids"]
            pair_q_orig_attention_mask = pair_q_orig_outputs["attention_mask"]
            res["pair_q_orig_input_ids"] = pair_q_orig_input_ids
            res["pair_q_orig_attention_mask"] = pair_q_orig_attention_mask

        if self.k_option_dropout:
            pair_k_orig_outputs = self.tokenizer(pair_k_a, pair_k_orig_b, **_tokenize_kwargs)
            pair_k_orig_input_ids = pair_k_orig_outputs["input_ids"]
            pair_k_orig_attention_mask = pair_k_orig_outputs["attention_mask"]
            res["pair_k_orig_input_ids"] = pair_k_orig_input_ids
            res["pair_k_orig_attention_mask"] = pair_k_orig_attention_mask

        return res

    def __call__(self, batch):
        pair_q_examples = [b["pair_q"] for b in batch]
        pair_k_examples = [b["pair_k"] for b in batch]

        # batch, max_sent_num, max_sent_len
        q_sent_token_index, q_sent_token_mask, q_sent_index = sentence_spans_list_to_tensor(pair_q_examples)
        k_sent_token_index, k_sent_token_mask, k_sent_index = sentence_spans_list_to_tensor(pair_k_examples)

        q_flat_rel_ids = []
        for q_id, q_exp in enumerate(pair_q_examples):
            q_flat_rel_ids.extend(q_exp["rel_ids_input_order"])
        assert len(q_flat_rel_ids) == q_sent_index.size(0)
        # tmp = [q_exp["rel_ids_input_order"] for q_exp in pair_q_examples]
        # logger.info(f"Q unflatten rel ids: {tmp}")

        k_flat_rel_ids = []
        for k_id, k_exp in enumerate(pair_k_examples):
            k_flat_rel_ids.extend(k_exp["rel_ids_input_order"])
        assert len(k_flat_rel_ids) == k_sent_index.size(0)
        # tmp = [k_exp["rel_ids_input_order"] for k_exp in pair_k_examples]
        # logger.info(f"K unflatten rel ids: {tmp}")

        k_label2index = defaultdict(list)
        k_flat_rel_ids_mask = torch.ones(len(k_flat_rel_ids))
        for k_idx, k_rel_id in enumerate(k_flat_rel_ids):
            if k_rel_id == -1:
                k_flat_rel_ids_mask[k_idx] = 0
                continue
            k_label2index[k_rel_id].append(k_idx)

        # logger.info(f"Q flat rel ids: {q_flat_rel_ids}")
        # logger.info(f"K flat rel ids: {k_flat_rel_ids}")

        local_ctr_labels = []
        local_ctr_score_mask = torch.ones(len(q_flat_rel_ids), len(k_flat_rel_ids))
        for q_idx, q_rel_id in enumerate(q_flat_rel_ids):
            local_ctr_score_mask[q_idx].copy_(k_flat_rel_ids_mask)

            if q_rel_id == -1:
                local_ctr_labels.append(-1)
                continue

            if q_rel_id not in k_label2index:
                local_ctr_labels.append(-1)
                continue

            q_label = random.choice(k_label2index[q_rel_id])
            for label_k_index in k_label2index[q_rel_id]:
                if label_k_index != q_label:
                    local_ctr_score_mask[q_idx, label_k_index] = 0
            local_ctr_labels.append(q_label)
        # logger.info(f"Local ctr labels: {local_ctr_labels}")
        # logger.info(f"Local ctr score mask: {local_ctr_score_mask.tolist()}")
        local_ctr_labels = torch.tensor(local_ctr_labels, dtype=torch.long)

        results = self.previous_forward(batch)
        results.update({
            "q_sent_token_index": q_sent_token_index,
            "q_sent_token_mask": q_sent_token_mask,
            "q_sent_index": q_sent_index,
            "k_sent_token_index": k_sent_token_index,
            "k_sent_token_mask": k_sent_token_mask,
            "k_sent_index": k_sent_index,
            "local_ctr_labels": local_ctr_labels,
            "local_ctr_mask": local_ctr_score_mask,
            "local_ctr_true_label_num": (local_ctr_labels != -1).sum(),
            "local_ctr_value_num": local_ctr_score_mask.sum() / len(batch)
        })
        return results


class WikiPathDatasetCollatorWithContextAndPairCompleteKMeansLocalBatch(WikiPathDatasetCollatorWithContextAndPairCompleteKMeansLocal):
    def __init__(self, max_seq_length: int, tokenizer: str, mlm_probability: float = 0.15, max_option_num: int = 4, swap: bool = False,
                 option_dropout: float = 0.0, k_option_dropout: float = 0.0):
        super().__init__(max_seq_length, tokenizer, mlm_probability, max_option_num, swap, option_dropout, k_option_dropout)

    def __call__(self, batch):
        pair_q_examples = [b["pair_q"] for b in batch]
        pair_k_examples = [b["pair_k"] for b in batch]

        # batch, max_sent_num, max_sent_len
        q_sent_token_index, q_sent_token_mask, _ = sentence_spans_list_to_tensor(pair_q_examples)
        k_sent_token_index, k_sent_token_mask, _ = sentence_spans_list_to_tensor(pair_k_examples)

        batch_size, q_sent_num = q_sent_token_index.size()[:2]
        k_sent_num = k_sent_token_index.size(1)

        local_ctr_labels = torch.ones(batch_size, q_sent_num, dtype=torch.long).fill_(-1)
        local_ctr_score_mask = torch.zeros(batch_size, q_sent_num, k_sent_num)

        for pair_id, (q_exp, k_exp) in enumerate(zip(pair_q_examples, pair_k_examples)):
            for q_sent_id, q_sent_rel_id in enumerate(q_exp["rel_ids_input_order"]):
                local_ctr_score_mask[pair_id, q_sent_id, :len(k_exp["rel_ids_input_order"])] = 1
                for k_sent_id, k_sent_rel_id in enumerate(k_exp["rel_ids_input_order"]):
                    if q_sent_rel_id == k_sent_rel_id:
                        if local_ctr_labels[pair_id, q_sent_id] != -1:
                            local_ctr_score_mask[pair_id, q_sent_id, k_sent_id] = 0
                            continue
                        local_ctr_labels[pair_id, q_sent_id] = k_sent_id

        results = self.previous_forward(batch)
        results.update({
            "q_sent_token_index": q_sent_token_index,
            "q_sent_token_mask": q_sent_token_mask,
            "k_sent_token_index": k_sent_token_index,
            "k_sent_token_mask": k_sent_token_mask,
            "local_ctr_labels": local_ctr_labels,
            "local_ctr_mask": local_ctr_score_mask,
            "local_ctr_true_label_num": (local_ctr_labels != -1).sum(),
            "local_ctr_value_num": local_ctr_score_mask.sum() / len(batch)
        })
        return results


class WikiPathDatasetCollatorWithContextAndPairCompleteDropout(WikiPathDatasetCollator):
    def __init__(self, max_seq_length: int, tokenizer: str, mlm_probability: float = 0.15, max_option_num: int = 4, swap: bool = False,
                 q_option_dropout: float = 0.0, k_option_dropout: float = 0.0, k_context_dropout: float = 0.0, add_tagging: bool = False):
        super().__init__(max_seq_length, tokenizer, mlm_probability, max_option_num)
        self.swap = swap
        self.q_option_dropout = q_option_dropout
        self.k_option_dropout = k_option_dropout
        self.k_context_dropout = k_context_dropout
        self.add_tagging = add_tagging

    def concat_sentence_and_dropout(self, item):
        # item["example"]["context"] = " ".join(item["example"]["context"])
        # if "negative_context" in item["example"]:
        #     item["example"]["negative_context"] = [" ".join(ctx) for ctx in item["example"]["negative_context"]]
        #
        # item["pair_q"]["context"] = " ".join(item["pair_q"]["context"])

        k_sent_drop_cnt = 0
        if self.k_context_dropout > 0:
            dropped_context = []
            for _sent in item["pair_k"]["context_sentences"]:
                _r = random.random()
                if _r < self.k_context_dropout:
                    k_sent_drop_cnt += 1
                else:
                    dropped_context.append(_sent)
            item["pair_k"]["context"] = " ".join(dropped_context)

        # item["pair_k"]["context"] = " ".join(item["pair_k"]["context"])
        # if "negative_context" in item["pair_k"]:
        #     item["pair_k"]["negative_context"] = [" ".join(ctx) for ctx in item["pair_k"]["negative_context"]]

        return item, k_sent_drop_cnt

    @staticmethod
    def prepare_single_example_positive(example):
        if "negative_context" in example:
            input_a = example["context"]
            input_b = example["condition"]
        else:
            input_a = example["context"]
            input_b = example["positive"]
        return input_a, input_b

    def prepare_single_example(self, item):
        input_a = []
        input_b = []
        if "negative_context" in item["example"]:
            input_a.extend(([item["example"]["context"]] + item["example"]["negative_context"])[:self.max_option_num])
            input_b.extend([item["example"]["condition"]] * self.max_option_num)
        else:
            op = ([item["example"]["positive"]] + item["example"]["negative"])[:self.max_option_num]
            if self.swap:
                input_a.extend([item["example"]["context"]] * len(op))
                input_b.extend(op)
            else:
                input_a.extend(op)
                input_b.extend([item["example"]["context"]] * len(op))

        assert len(input_a) == len(input_b) == self.max_option_num

        pair_q_input_a, pair_q_input_b = self.prepare_single_example_positive(item["pair_q"])
        pair_k_input_a, pair_k_input_b = self.prepare_single_example_positive(item["pair_k"])

        return input_a, input_b, item["pair_q"]["orig_id"], pair_q_input_a, pair_q_input_b, \
            item["pair_k"]["orig_id"], pair_k_input_a, pair_k_input_b, \
            item["pair_k_orig_ids"]

    def __call__(self, batch):
        input_a, input_b, texts = [], [], []
        pair_q_orig_ids, pair_q_a, pair_q_b, pair_k_orig_ids, pair_k_a, pair_k_b, all_paired_orig_ids = [], [], [], [], [], [], []
        dropped_op_cnt = 0
        k_sent_drop_cnt = 0
        for b in batch:
            b, b_k_sent_drop_cnt = self.concat_sentence_and_dropout(b)
            k_sent_drop_cnt += b_k_sent_drop_cnt

            b_input_a, b_input_b, b_pair_q_orig_id, b_pair_q_a, b_pair_q_b, \
                b_pair_k_orig_id, b_pair_k_a, b_pair_k_b, b_pair_k_orig_ids = self.prepare_single_example(b)

            input_a.extend(b_input_a)
            input_b.extend(b_input_b)
            pair_q_orig_ids.append(b_pair_q_orig_id)
            pair_q_a.append(b_pair_q_a)

            _r = random.random()
            if _r < self.q_option_dropout:
                pair_q_b.append("")
                dropped_op_cnt += 1
            else:
                pair_q_b.append(b_pair_q_b)

            pair_k_orig_ids.append(b_pair_k_orig_id)
            pair_k_a.append(b_pair_k_a)

            _r = random.random()
            if _r < self.k_option_dropout:
                pair_k_b.append("")
            else:
                pair_k_b.append(b_pair_k_b)

            all_paired_orig_ids.append(b_pair_k_orig_ids)
            texts.append(b["text"])

        if self.add_tagging:
            tagging_labels = [b["example"].pop("tagging_label") for b in batch]
        else:
            tagging_labels = None

        del batch
        _tokenize_kwargs = {
            "padding": PaddingStrategy.LONGEST,
            "truncation": TruncationStrategy.LONGEST_FIRST,
            "max_length": self.max_seq_length,
            "return_tensors": "pt"
        }

        tokenizer_outputs = self.tokenizer(input_a, input_b, **_tokenize_kwargs)
        input_ids = tokenizer_outputs["input_ids"]
        attention_mask = tokenizer_outputs["attention_mask"]

        pair_q_tokenizer_outputs = self.tokenizer(pair_q_a, pair_q_b, **_tokenize_kwargs)
        pair_q_input_ids = pair_q_tokenizer_outputs["input_ids"]
        pair_q_attention_mask = pair_q_tokenizer_outputs["attention_mask"]

        pair_k_tokenizer_outputs = self.tokenizer(pair_k_a, pair_k_b, **_tokenize_kwargs)
        pair_k_input_ids = pair_k_tokenizer_outputs["input_ids"]
        pair_k_attention_mask = pair_k_tokenizer_outputs["attention_mask"]

        # prepare pair labels and mask
        pair_align_mask = []  # `1` for mask and `0` for true value.
        pair_align_labels = []  # `-1` for non-pair examples.
        assert len(pair_q_orig_ids) == len(pair_k_orig_ids) == len(all_paired_orig_ids)
        for q_id, (q_orig_id, q_paired_orig_id_list) in enumerate(zip(pair_q_orig_ids, all_paired_orig_ids)):
            # pair dot product
            pair_mask = []
            for k_id, k_orig_id in enumerate(pair_k_orig_ids):
                if q_id == k_id:
                    assert k_orig_id in q_paired_orig_id_list
                    pair_mask.append(0)
                else:
                    if k_orig_id in q_paired_orig_id_list or k_orig_id == q_orig_id:
                        pair_mask.append(1)
                    else:
                        pair_mask.append(0)
            if sum(pair_mask) + 1 == len(pair_k_orig_ids):
                pair_align_labels.append(-1)
            else:
                pair_align_labels.append(q_id)

            # TODO: Add self-product for more negative samples? Then the option dropout should be modified.

            pair_align_mask.append(pair_mask)

        pair_align_mask = torch.tensor(pair_align_mask, dtype=torch.int)
        pair_align_labels = torch.tensor(pair_align_labels, dtype=torch.long)
        num_labels = (pair_align_labels > -1).sum()

        mlm_tokenize_outputs = self.tokenizer(texts, **_tokenize_kwargs)
        mlm_input_ids = mlm_tokenize_outputs["input_ids"]
        mlm_attention_mask = mlm_tokenize_outputs["attention_mask"]

        mlm_input_ids, mlm_labels = self.mask_tokens(mlm_input_ids)

        batch_size = len(pair_q_input_ids)
        option_num = self.max_option_num

        res = {
            "input_ids": input_ids.reshape(batch_size, option_num, -1),
            "attention_mask": attention_mask.reshape(batch_size, option_num, -1),
            "labels": torch.zeros(batch_size, dtype=torch.long),
            "mlm_input_ids": mlm_input_ids,
            "mlm_attention_mask": mlm_attention_mask,
            "mlm_labels": mlm_labels,
            "pair_q_input_ids": pair_q_input_ids,
            "pair_q_attention_mask": pair_q_attention_mask,
            "pair_k_input_ids": pair_k_input_ids,
            "pair_k_attention_mask": pair_k_attention_mask,
            "pair_mask": pair_align_mask,
            "pair_labels": pair_align_labels,
            "pair_label_num": num_labels,
            "pair_value_num": (1 - pair_align_mask).sum(dim=-1)[pair_align_labels > -1].sum() / num_labels,
            "dropped_op_cnt": dropped_op_cnt,
            "k_sent_drop_cnt": k_sent_drop_cnt / batch_size
        }
        if "token_type_ids" in tokenizer_outputs:
            res["token_type_ids"] = tokenizer_outputs["token_type_ids"]

        if self.add_tagging:
            tagging_labels = torch.tensor(tagging_labels, dtype=torch.long)
            tagging_labels = tagging_labels[:, :input_ids.size(1)]
            tagging_labels[res["input_ids"][:, 0] == self.tokenizer.pad_token_id] = -1
            res["tagging_labels"] = tagging_labels

        # print("pair_value_num", res["pair_value_num"])
        return res


class WikiPathDatasetCollatorWithContextInMLMPredict(WikiPathDatasetCollatorWithContext):
    def __init__(self, max_seq_length: int, tokenizer: str, mlm_probability: float = 0.15, max_option_num: int = 4, swap: bool = False,
                 use_mask: bool = False):
        super().__init__(max_seq_length, tokenizer, mlm_probability, max_option_num)
        self.swap = swap
        self.use_mask = use_mask

    def __call__(self, batch):
        index = torch.tensor([b["index"] for b in batch], dtype=torch.long)
        inputs = super().__call__(batch)

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        if "token_type_ids" in inputs:
            token_type_ids = inputs["token_type_ids"]
        else:
            token_type_ids = None

        assert input_ids.size(1) == self.max_option_num
        input_ids = input_ids.reshape(-1, self.max_seq_length)
        attention_mask = attention_mask.reshape(-1, self.max_seq_length)

        sep_mask = input_ids == self.tokenizer.sep_token_id
        acc_sep_mask = sep_mask.cumsum(dim=1)
        sep_mask_num = acc_sep_mask.max(dim=1, keepdim=True)[0]
        mlm_mask = (acc_sep_mask.eq(sep_mask_num) * attention_mask).bool()

        if self.use_mask:
            input_ids[mlm_mask] = self.tokenizer.mask_token_id

        labels = input_ids.clone()
        labels[~mlm_mask] = -1

        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "mlm_labels": labels,
            "labels": inputs["labels"],
            "index": index
        }
        if token_type_ids is not None:
            inputs["token_type_ids"] = token_type_ids.reshape(-1, self.max_seq_length)

        return inputs


class WikiPathDatasetCollatorWithContextBinary(WikiPathDatasetCollator):
    def __init__(self, max_seq_length: int, tokenizer: str, mlm_probability: float = 0.15, max_option_num: int = 4, swap: bool = False):
        super().__init__(max_seq_length, tokenizer, mlm_probability, max_option_num)
        self.swap = swap

    def __call__(self, batch):
        # examples, texts = list(zip(*batch))
        op_examples, ctx_examples, texts = [], [], []
        for b in batch:
            example = b.pop("example")
            if "negative_context" in example:
                ctx_examples.append(example)
            else:
                op_examples.append(example)
            # examples.append(b.pop("example"))
            texts.append(b.pop("text"))
            # assert isinstance(texts[-1], str), texts[-1]
        del batch
        batch_size = len(op_examples) + len(ctx_examples)
        assert batch_size == len(texts)

        input_a = []
        input_b = []
        option_num = -1

        for e in op_examples:
            # op = ([e["positive"]] + e["negative"])[:self.max_option_num]
            op = [e["positive"]] + [random.choice(e["negative"])]
            if self.swap:
                input_a.extend([e["context"]] * len(op))
                input_b.extend(op)
            else:
                input_a.extend(op)
                input_b.extend([e["context"]] * len(op))
            if option_num == -1:
                option_num = len(op)
            else:
                assert option_num == len(op)

        for e in ctx_examples:
            positive_context = e.pop("context")
            negative_context = e.pop("negative_context")
            op = e.pop("condition")
            # input_a.extend([positive_context] + negative_context)
            # input_b.extend([op] * (len(negative_context) + 1))
            input_a.extend([positive_context] + [random.choice(negative_context)])
            input_b.extend([op] * 2)
            if option_num == -1:
                option_num = 2
            else:
                assert option_num == 1 + 1, option_num

        option_num = min(option_num, self.max_option_num)

        tokenizer_outputs = self.tokenizer(input_a, input_b, padding=PaddingStrategy.MAX_LENGTH,
                                           truncation=TruncationStrategy.LONGEST_FIRST, max_length=self.max_seq_length,
                                           return_tensors="pt")
        input_ids = tokenizer_outputs["input_ids"]
        attention_mask = tokenizer_outputs["attention_mask"]

        mlm_tokenize_outputs = self.tokenizer(texts, padding=PaddingStrategy.MAX_LENGTH,
                                              truncation=TruncationStrategy.LONGEST_FIRST, max_length=self.max_seq_length,
                                              return_tensors="pt")
        mlm_input_ids = mlm_tokenize_outputs["input_ids"]
        mlm_attention_mask = mlm_tokenize_outputs["attention_mask"]

        mlm_input_ids, mlm_labels = self.mask_tokens(mlm_input_ids)

        labels_true = torch.ones((batch_size, 1), dtype=torch.long, device=input_ids.device)
        labels_false = torch.zeros((batch_size, 1), dtype=torch.long, device=input_ids.device)
        labels = torch.cat([labels_true, labels_false], dim=1).reshape(-1)
        assert labels.size(0) == input_ids.size(0)

        res = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "mlm_input_ids": mlm_input_ids,
            "mlm_attention_mask": mlm_attention_mask,
            "mlm_labels": mlm_labels
        }
        if "token_type_ids" in tokenizer_outputs:
            res["token_type_ids"] = tokenizer_outputs["token_type_ids"]
        return res


class WikiPathDatasetCollatorSeq2Seq:
    def __init__(self, max_input_length: int, max_output_length: int, tokenizer: str, sent_sample_ratio: float = 0.4,
                 use_template: bool = True):
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast=False)
        if isinstance(self.tokenizer, T5Tokenizer):
            self.sep_token = "<extra_id_0>"
        elif isinstance(self.tokenizer, BartTokenizer):
            self.sep_token = "<s>"
        else:
            raise RuntimeError("Unsupported tokenizer {}".format(tokenizer.__class__.__name__))
        self.sent_sample_ratio = sent_sample_ratio
        self.prefix = "generate logically consistent deductions: "
        self.templates = [
            "What would happen if {}?",
            "If {} then what would happen?",
        ]
        self.use_template = use_template

        transformers.logging.set_verbosity_error()

    def __call__(self, batch):

        inputs = []
        outputs = []
        for b in batch:
            example = b["example"]
            ctx = example["context"]
            condition = example["condition"] if "condition" in example else example["positive"]
            if self.use_template:
                condition = random.choice(self.templates).format(condition)

            output_sent_num = int(len(ctx) * self.sent_sample_ratio)
            output_sent_num = max(output_sent_num, 1)

            sent_ids = list(range(len(ctx)))
            output_sent_ids = set(random.sample(sent_ids, output_sent_num))
            output_sents = [ctx[i] for i in output_sent_ids]
            input_sents = [ctx[i] for i in sent_ids if i not in output_sent_ids]

            inputs.append(' '.join([self.prefix] + input_sents + [condition]))
            outputs.append(self.sep_token.join(output_sents))

        model_inputs = self.tokenizer(inputs, return_tensors="pt",
                                      padding=PaddingStrategy.LONGEST,
                                      truncation=True, max_length=self.max_input_length)

        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(outputs, return_tensors="pt",
                                    padding=PaddingStrategy.LONGEST,
                                    truncation=True, max_length=self.max_output_length)["input_ids"]
        model_inputs["labels"] = labels

        return model_inputs


class WikiPathDatasetCollatorSeq2SeqV2:
    def __init__(self, max_input_length: int, max_output_length: int, tokenizer: str, use_template: bool = True):
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast=False)
        if isinstance(self.tokenizer, T5Tokenizer):
            self.sep_token = "<extra_id_0>"
        elif isinstance(self.tokenizer, BartTokenizer):
            self.sep_token = "<s>"
        else:
            raise RuntimeError("Unsupported tokenizer {}".format(tokenizer.__class__.__name__))
        self.prefix = "generate logically consistent deductions: "
        self.templates = [
            "What would happen if {}?",
            "If {} then what would happen?",
        ]
        self.use_template = use_template

        transformers.logging.set_verbosity_error()

    def __call__(self, batch):

        inputs = []
        outputs = []
        for b in batch:
            example = b["example"]
            ctx = example["context"]
            output = example["condition"] if "condition" in example else example["positive"]

            if self.use_template:
                question = ctx[-1]
                question = random.choice(self.templates).format(question)
                context = ctx[:-1]
                assert len(context) > 0

                inputs.append(' '.join([self.prefix] + context + [question]))
            else:
                inputs.append(' '.join([self.prefix] + ctx))
            outputs.append(output)

        model_inputs = self.tokenizer(inputs, return_tensors="pt",
                                      padding=PaddingStrategy.LONGEST,
                                      truncation=True, max_length=self.max_input_length)

        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(outputs, return_tensors="pt",
                                    padding=PaddingStrategy.LONGEST,
                                    truncation=True, max_length=self.max_output_length)["input_ids"]
        model_inputs["labels"] = labels

        return model_inputs


def transform_rel_label_to_tensor(rel_labels):
    if isinstance(rel_labels[0][0], list):
        new_rel_labels = []
        for item_rel_labels in rel_labels:
            new_rel_labels.extend(item_rel_labels)
        rel_labels = new_rel_labels

    assert isinstance(rel_labels[0][0], int)

    max_input_len = max(map(len, rel_labels))
    invalid = 0
    decoder_input_ids = torch.zeros(len(rel_labels), max_input_len, dtype=torch.long).fill_(-1)
    for b, b_decoder_inputs in enumerate(rel_labels):
        decoder_input_ids[b, :len(b_decoder_inputs)] = torch.tensor(b_decoder_inputs, dtype=torch.long)
        if b_decoder_inputs[0] == -1:
            invalid += 1

    return decoder_input_ids, invalid


class WikiPathDatasetCollatorRelSeqGenV1(WikiPathDatasetCollatorWithContext):
    def __init__(self, max_seq_length: int, tokenizer: str, mlm_probability: float = 0.15, max_option_num: int = 4, swap: bool = False,
                 gen_only: bool = False, option_dropout: float = 0.0):
        super().__init__(max_seq_length, tokenizer, mlm_probability, max_option_num, swap)
        self.gen_only = gen_only
        self.option_dropout = option_dropout

    def __call__(self, batch):
        rel_decode = []
        for b in batch:
            rel_decode.append(b["rel_labels"])

        # max_input_len = max(map(len, rel_decode))
        # invalid = 0
        # decoder_input_ids = torch.zeros(len(batch), max_input_len, dtype=torch.long).fill_(-1)
        # for b, b_decoder_inputs in enumerate(rel_decode):
        #     decoder_input_ids[b, :len(b_decoder_inputs)] = torch.tensor(b_decoder_inputs, dtype=torch.long)
        #     if b_decoder_inputs[0] == -1:
        #         invalid += 1
        decoder_input_ids, invalid = transform_rel_label_to_tensor(rel_decode)

        dropped_op_cnt = 0
        if self.option_dropout > 0:
            examples = [b["example"] for b in batch]
            inputs_a = []
            inputs_b = []
            for exp in examples:
                inputs_a.append(exp["context"])

                _r = random.random()
                if _r < self.option_dropout:
                    inputs_b.append("")
                    dropped_op_cnt += 1
                    continue

                if "negative_context" in exp:
                    op = exp["condition"]
                else:
                    op = exp["positive"]
                inputs_b.append(op)
            assert len(inputs_a) == len(inputs_b)
            dropout_res = self.tokenizer(inputs_a, inputs_b, padding=PaddingStrategy.LONGEST,
                                         truncation=TruncationStrategy.LONGEST_FIRST, max_length=self.max_seq_length,
                                         return_tensors="pt")
        else:
            dropout_res = None

        res = super().__call__(batch)
        res["rel_labels"] = decoder_input_ids
        res["invalid_path"] = invalid
        res["dropped_op_cnt"] = dropped_op_cnt
        if dropout_res is not None:
            for k, v in dropout_res.items():
                res[f"{k}_dropout"] = v
                assert v.size(0) == decoder_input_ids.size(0)

        if self.gen_only:
            res["input_ids"] = res["input_ids"][:, 0]
            res["attention_mask"] = res["attention_mask"][:, 0]

        return res


class WikiPathDatasetCollatorRelSeqGenV3(WikiPathDatasetCollatorRelSeqGenV1):
    def __init__(self, max_seq_length: int, tokenizer: str, mlm_probability: float = 0.15, max_option_num: int = 4, swap: bool = False,
                 gen_only: bool = False, option_dropout: float = 0.0, max_output_length: int = 128):
        super().__init__(max_seq_length, tokenizer, mlm_probability, max_option_num, swap, gen_only, option_dropout)
        self.max_output_length = max_output_length

    def __call__(self, batch):
        ent_mentions = []
        for b in batch:
            item_ent_mentions = b["example"]["ent_mentions"]
            item_ent_mentions = item_ent_mentions[:-1]  # Remove the direct entity pair.

            tmp = []
            for pair_id, ent_pair in enumerate(item_ent_mentions):
                if pair_id == 0:
                    tmp.append(ent_pair[0])
                tmp.append(ent_pair[1])
            ent_mentions.append(self.tokenizer.sep_token.join(tmp))

        labels = self.tokenizer(ent_mentions, return_tensors="pt",
                                padding=PaddingStrategy.LONGEST,
                                truncation=True, max_length=self.max_output_length)["input_ids"]

        res = super().__call__(batch)
        res["entity_mentions"] = labels

        return res


class WikiPathDatasetCollatorRelSeqGenV2(WikiPathDatasetCollatorWithContext):
    def __init__(self, max_seq_length: int, tokenizer: str, mlm_probability: float = 0.15, max_option_num: int = 4, swap: bool = False):
        super().__init__(max_seq_length, tokenizer, mlm_probability, max_option_num, swap)

    def __call__(self, batch):
        examples = [b["example"] for b in batch]

        rel_decode = []
        for b in batch:
            rel_decode.append(b["rel_labels"])

        max_input_len = max(map(len, rel_decode))
        invalid = 0
        decoder_input_ids = torch.zeros(len(batch), max_input_len, dtype=torch.long).fill_(-1)
        for b, b_decoder_inputs in enumerate(rel_decode):
            # if "original_orders" in batch[b]["example"] and b_decoder_inputs[0] != -1:
            #     # print("Entering Here...")
            #     assert len(batch[b]["example"]["original_orders"]) == len(b_decoder_inputs) - 2, (batch[b]["example"]["original_orders"],
            #                                                                                       b_decoder_inputs)
            #     rerank_b_decoder_inputs = [b_decoder_inputs[rank] for rank in batch[b]["example"]["original_orders"]] + \
            #                               b_decoder_inputs[len(batch[b]["example"]["original_orders"]):]
            #     b_decoder_inputs = rerank_b_decoder_inputs
            if b_decoder_inputs[0] != -1:
                if "sent_order2edge_rank" in examples[b]:
                    assert len(examples[b]["sent_order2edge_rank"]) == len(b_decoder_inputs) - 2, (
                        examples[b]["sent_order2edge_rank"], b_decoder_inputs)
                    rerank_b_decoder_inputs = [b_decoder_inputs[edge_rank] for edge_rank in examples[b]["sent_order2edge_rank"]] + \
                                              b_decoder_inputs[len(examples[b]["sent_order2edge_rank"]):]
                    b_decoder_inputs = rerank_b_decoder_inputs

                decoder_input_ids[b, :len(b_decoder_inputs)] = torch.tensor(b_decoder_inputs, dtype=torch.long)
            else:
                invalid += 1
            # if b_decoder_inputs[0] == -1:
            #     invalid += 1

        res = super().__call__(batch)
        res["rel_labels"] = decoder_input_ids
        res["invalid_path"] = invalid

        if "sentence_spans" in examples[0]:
            max_sent_num = max(map(lambda x: len(x["sentence_spans"]), examples))
            max_sent_len = 0
            for example in examples:
                max_sent_len = max(max_sent_len, max(map(lambda x: x[1] - x[0], example["sentence_spans"])))

            # padding relation labels to avoid the example with more sentences than none relations.
            decoder_input_ids = torch.cat([decoder_input_ids,
                                           torch.zeros(len(batch), max_sent_num - (max_input_len - 1), dtype=torch.long).fill_(-1)], dim=1)
            res["rel_labels"] = decoder_input_ids

            sent_token_index = torch.zeros(len(batch), max_sent_num, max_sent_len, dtype=torch.long)
            sent_token_mask = torch.zeros(len(batch), max_sent_num, max_sent_len)
            # sent_mask = torch.zeros(len(batch), max_sent_num)
            for exp_id, exp in enumerate(examples):
                for s_id, (s, e) in enumerate(exp["sentence_spans"]):
                    sent_token_index[exp_id, s_id, :(e - s)] = torch.arange(s, e, dtype=torch.long)
                    sent_token_mask[exp_id, s_id, :(e - s)] = 1
                # sent_mask[exp_id, :len(exp["sentence_spans"])] = 1
                # if rel_decode[exp_id][0] != -1:
                #     assert len(exp["sent_order2edge_rank"]) + 1 == len(rel_decode[exp_id])

            res["sent_token_index"] = sent_token_index
            res["sent_token_mask"] = sent_token_mask
            # res["sent_mask"] = sent_mask
        else:
            input_ids = res["input_ids"][:, 0]
            # print(self.tokenizer.convert_ids_to_tokens(input_ids[0]))
            sep_token_mask = input_ids == self.tokenizer.sep_token_id  # [batch, max_seq_length]
            sent_num = sep_token_mask.sum(dim=1)  # [batch]
            sep_index = torch.zeros(input_ids.size(0), sent_num.max().item() - 1, dtype=torch.long)  # For roberta only.
            for b, b_sep_token_mask in enumerate(sep_token_mask):
                cnt = 0
                for tk_id, tk_mask in enumerate(b_sep_token_mask):
                    if tk_mask == 1:
                        # overlook the continuous sep tokens and only keep the first.
                        if cnt > 0 and all(b_sep_token_mask[tmp] == 1 for tmp in range(sep_index[b, cnt - 1].item(), tk_id + 1, 1)):
                            continue
                        sep_index[b, cnt] = tk_id
                        cnt += 1
                if rel_decode[b][0] != -1:
                    assert len(rel_decode[b]) == cnt + 1, (rel_decode[b], cnt, self.tokenizer.convert_ids_to_tokens(input_ids[b]),
                                                           examples[b]["original_orders"])
                else:
                    if cnt >= max_input_len:
                        decoder_input_ids = torch.cat([decoder_input_ids,
                                                       torch.zeros(len(batch), cnt + 1 - max_input_len, dtype=torch.long).fill_(-1)], dim=1)
                        res["rel_labels"] = decoder_input_ids
                        max_input_len = cnt + 1

            res["sep_index"] = sep_index
            assert sep_index.size(1) == decoder_input_ids.size(1) - 1, (sep_index, decoder_input_ids)

        return res


class WikiPathDatasetCollatorReconstruction(WikiPathDatasetCollatorWithContext):
    def __init__(self, max_seq_length: int, tokenizer: str, mlm_probability: float = 0.15, max_option_num: int = 4, swap: bool = False):
        super().__init__(max_seq_length, tokenizer, mlm_probability, max_option_num, swap)

    def make_triplet_tensor(self, ent_mentions: List[Tuple[str, str]]):
        input_ids = []
        rel_markers = []
        for ent_tuple in ent_mentions:
            ent_token_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(ent_tuple[0]))
            input_ids.extend(ent_token_ids)

    def __call__(self, batch):
        ent_mentions = [b["example"]["ent_mentions"] for b in batch]

        return res
