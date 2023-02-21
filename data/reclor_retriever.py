import base64
import collections
import os.path

import torch
from torch.utils.data import Dataset, default_collate
import functools

from transformers import PreTrainedTokenizer, AutoTokenizer
from transformers.tokenization_utils import TruncationStrategy, PaddingStrategy
from data.data_utils import tokenizer_get_name

from general_util.logger import get_child_logger
from tqdm import tqdm
import hashlib
from multiprocessing import Pool

logger = get_child_logger(__name__)


def get_str_as_md5(parm_str):
    # 1、参数必须是utf8
    # 2、python3所有字符都是unicode形式，已经不存在unicode关键字
    # 3、python3 str 实质上就是unicode
    if isinstance(parm_str, str):
        # 如果是unicode先转utf-8
        parm_str = parm_str.encode("utf-8")
    m = hashlib.md5()
    m.update(parm_str)
    return m.hexdigest()


def load_memory(memory_path: str, tokenizer: PreTrainedTokenizer):
    all_examples, raw_texts = torch.load(memory_path, map_location="cpu")
    logger.info(f"Loading raw texts from memory.")
    memory = []
    for exp in tqdm(all_examples, total=len(all_examples)):
        token_ids = tokenizer.convert_tokens_to_ids(exp["tokens"][0])
        text = tokenizer.decode(token_ids, skip_special_tokens=True)
        memory.append(text)
    return memory


def process_single_sorting(score_list, top_k):
    idx, score_list = score_list
    return idx, sorted(score_list, key=lambda x: x[1], reverse=True)[:top_k]


def load_scores(scores_path: str, top_k: int, num_workers: int = 64):
    # predictions = torch.load(scores_path)
    # index = predictions["index"]
    # scores = predictions["predictions"]
    # logger.info(f"Loading scores and sorting")
    ranking_scores = torch.load(scores_path)

    q2k_scores = {}
    # for idx, pair_score in tqdm(zip(index, scores), total=len(index)):
    #     q_id, k_id = idx.split("-")
    #     q_id = int(q_id)
    #     k_id = int(k_id)
    #     q2k_scores[q_id].append((k_id, pair_score))
    for q_id, scores in tqdm(ranking_scores.items()):
        if isinstance(q_id, torch.Tensor):
            q_id = q_id.item()
        q2k_scores[q_id] = [int(k_id) for k_id, _ in scores][:top_k]

    # with Pool(num_workers) as p:
    #     _annotate = functools.partial(process_single_sorting, top_k=top_k)
    #     _results = list(tqdm(
    #         p.imap(_annotate, list(q2k_scores.items())),
    #         total=len(q2k_scores),
    #         desc="Sorting"
    #     ))
    #     for q_id, scores in tqdm(_results, total=len(_results)):
    #         q2k_scores[q_id] = scores

    # for q_id in tqdm(q2k_scores, total=len(q2k_scores)):
    #     q2k_scores[q_id] = sorted(q2k_scores[q_id], key=lambda x: x[1], reverse=True)[:top_k]

    return q2k_scores


class ReClorRetrieveDataset(Dataset):
    def __init__(self, file_path: str, tokenizer: PreTrainedTokenizer, read_func, memory_path: str, scores_path: str,
                 top_k: int = 2, memory_tokenizer: str = None, num_workers: int = 64):
        super().__init__()
        if memory_tokenizer is not None:
            tokenizer = AutoTokenizer.from_pretrained(memory_tokenizer)

        tokenizer_name = tokenizer_get_name(tokenizer)

        file_suffix = f"{tokenizer_name}_{read_func.__class__.__name__}_rmc"
        cached_dir = "/".join(file_path.split("/")[:-1])
        cached_file_path = f"{file_path}_{memory_path.split('/')[-1]}_{scores_path.split('/')[-1]}_{top_k}_{file_suffix}"
        cached_file_path = get_str_as_md5(cached_file_path)
        cached_file_path = os.path.join(cached_dir, cached_file_path)

        if os.path.exists(cached_file_path):
            self.all_context, self.all_question, self.all_option_list, self.all_label, self.memory, self.q2k_scores \
                = torch.load(cached_file_path)
        else:
            memory = load_memory(memory_path, tokenizer)
            q2k_scores = load_scores(scores_path, top_k, num_workers=num_workers)

            all_context, all_question, all_option_list, all_label = read_func(file_path)

            self.all_context = all_context
            self.all_question = all_question
            self.all_option_list = all_option_list
            self.all_label = all_label
            self.memory = memory
            self.q2k_scores = q2k_scores
            torch.save((self.all_context, self.all_question, self.all_option_list, self.all_label, self.memory, self.q2k_scores),
                       cached_file_path)

    def __len__(self):
        return len(self.all_context)

    def __getitem__(self, index):
        ctx = self.all_context[index]
        q = self.all_context[index]
        op_list = self.all_option_list[index]
        label = self.all_label[index]
        demons = [self.memory[i] for i in self.q2k_scores[index]]
        return {
            "context": ctx,
            "question": q,
            "options": op_list,
            "label": label,
            "demonstrations": demons,
            "index": index,
        }


class ReClorRetrieveCollator:
    def __init__(self, tokenizer: str, max_seq_length: int):
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast=False)
        self.max_seq_length = max_seq_length

    def __call__(self, batch):
        context = [b["context"] for b in batch]
        question = [b["question"] for b in batch]
        options = [b["options"] for b in batch]
        demonstrations = [b["demonstrations"] for b in batch]

        inputs_a = []
        inputs_b = []
        outputs = []
        for c, q, op_list, demons in zip(context, question, options, demonstrations):
            _input_a = self.tokenizer.sep_token.join(demons)
            for op in op_list:
                _input_b = self.tokenizer.sep_token.join([c, q, op])
                inputs_a.append(_input_a)
                inputs_b.append(_input_b)
                outputs.append(_input_b)

        batch_size = len(batch)
        num_choices = len(options[0])

        inputs = self.tokenizer(inputs_a, text_pair=inputs_b, text_target=outputs,
                                truncation=TruncationStrategy.ONLY_FIRST, padding=PaddingStrategy.LONGEST,
                                max_length=self.max_seq_length, return_tensors="pt")
        inputs["input_ids"] = inputs["input_ids"].reshape(batch_size, num_choices, -1)
        inputs["attention_mask"] = inputs["attention_mask"].reshape(batch_size, num_choices, -1)
        inputs["decoder_input_ids"] = inputs.pop("labels").reshape(batch_size, num_choices, -1)
        inputs["labels"] = torch.tensor([b["label"] for b in batch], dtype=torch.long)

        inputs["meta_data"] = {
            "index": [b["index"] for b in batch]
        }

        return inputs
