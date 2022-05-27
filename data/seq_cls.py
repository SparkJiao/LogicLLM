import json
import os

import torch
from tqdm import tqdm
from transformers import PreTrainedTokenizer
from transformers.tokenization_utils_base import PaddingStrategy
from nltk import sent_tokenize

from data.collators.dict2dict import DictTensorDataset
from data.data_utils import dfs_load_assignment, tokenizer_get_name
from general_util.logger import get_child_logger

logger = get_child_logger("SeqCls")


def deduction_classification_for_infer(file_path: str, tokenizer: PreTrainedTokenizer, max_seq_length: int):
    """
    Deductions are classified as "yes" or "no".
    For inference only.
    """
    data = json.load(open(file_path, "r"))
    all_input_a = []
    all_input_b = []

    for item in data:
        passage = item['passage']
        for q in item['questions']:
            ques = q['question']
            if "prediction" in q:
                pred_deduction = q['prediction']
                all_input_a.append(f"{passage} {ques}")
                all_input_b.append(pred_deduction)

    model_inputs = tokenizer(all_input_a, text_pair=all_input_b,
                             max_length=max_seq_length, return_tensors="pt",
                             padding=PaddingStrategy.LONGEST, truncation=True)

    return DictTensorDataset(model_inputs)


def deduction_classification_for_infer_by_depth(file_path: str, tokenizer: PreTrainedTokenizer, max_seq_length: int,
                                                pred_on_question: bool = False, save: bool = False):
    """
    Deductions are classified as "yes" or "no".
    For inference only.
    """
    tokenizer_name = tokenizer_get_name(tokenizer)

    file_suffix = f"{tokenizer_name}_{max_seq_length}_{pred_on_question}_deduction_cls_depth_v1"
    cached_file_path = f"{file_path}_{file_suffix}"
    cached_file_path = cached_file_path.replace('*', '')
    if os.path.exists(cached_file_path):
        logger.info(f"Loading cached file from {cached_file_path}.")
        data = torch.load(cached_file_path)
        return DictTensorDataset(data)

    data = json.load(open(file_path, "r"))
    all_input_a = []
    all_input_b = []
    all_ids = []

    for item in tqdm(data, desc="Loading data"):
        passage = item['passage']
        if 'assignment' not in item:
            continue

        all_assignments = []
        dfs_load_assignment(item['assignment'], all_assignments, '')
        for _assign, _assign_id in all_assignments:
            all_input_a.append(passage)
            all_input_b.append(_assign)
            all_ids.append(_assign_id)

        if pred_on_question:
            pass

    model_inputs = tokenizer(all_input_a, text_pair=all_input_b,
                             max_length=max_seq_length, return_tensors="pt",
                             padding=PaddingStrategy.LONGEST, truncation=True)
    model_inputs["index"] = all_ids

    if save:
        logger.info(f"Saving to {cached_file_path}.")
        torch.save(model_inputs, cached_file_path)

    return DictTensorDataset(model_inputs)


def deduction_cls_for_loginli_from_examples(file_path: str, tokenizer: PreTrainedTokenizer, max_seq_length: int, add_labels: bool = False,
                                            examples=None):
    if examples is None:
        from data.examples import LSATDeductionExamples
        examples = LSATDeductionExamples()

    all_input_a = []
    all_input_b = []
    labels = []
    for item in examples.data:
        passage = item["passage"]
        if item["positive_deductions"] and item["negative_deductions"]:
            for pos in item["positive_deductions"]:
                all_input_a.append(passage)
                all_input_b.append(pos)
                if add_labels:
                    labels.append(1)
            for neg in item["negative_deductions"]:
                all_input_a.append(passage)
                all_input_b.append(neg)
                if add_labels:
                    labels.append(0)
        for q in item["questions"]:
            question = q["question"]
            if q["positive_deductions"] and q["negative_deductions"]:
                for pos in q["positive_deductions"]:
                    all_input_a.append(passage + ' ' + question)
                    all_input_b.append(pos)
                    if add_labels:
                        labels.append(1)
                for neg in q["negative_deductions"]:
                    all_input_a.append(passage + ' ' + question)
                    all_input_b.append(neg)
                    if add_labels:
                        labels.append(0)

    model_inputs = tokenizer(all_input_a, text_pair=all_input_b,
                             max_length=max_seq_length, return_tensors="pt",
                             padding=PaddingStrategy.LONGEST, truncation=True)
    if add_labels:
        model_inputs["labels"] = torch.tensor(labels, dtype=torch.long)

    return DictTensorDataset(model_inputs)


def lsat_weak_supervision_cls(file_path: str, tokenizer: PreTrainedTokenizer, max_input_length: int, sample: bool = False):
    label2id = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}

    data = json.load(open(file_path, 'r'))
    inputs_a = []
    inputs_b = []
    labels = []
    for item in data:
        passage = item['passage']
        for q in item['questions']:
            ques = q['question']
            if 'if' not in ques.lower():
                # inputs.append(prefix + ' ' + passage + ' ' + ques)
                # outputs.append(q['options'][label2id[q['answer']]])
                ans_id = label2id[q['answer']]
                neg_inputs_a = []
                neg_inputs_b = []
                for op_id, option in enumerate(q['options']):
                    if op_id == ans_id:
                        inputs_a.append(passage + ' ' + ques)
                        inputs_b.append(option)
                        labels.append(1)
                    else:
                        # labels.append(0)
                        neg_inputs_a.append(passage + ' ' + ques)
                        neg_inputs_b.append(option)
                if sample:
                    inputs_a.append(neg_inputs_a[0])
                    inputs_b.append(neg_inputs_b[0])
                    labels.append(0)
                else:
                    inputs_a.extend(neg_inputs_a)
                    inputs_b.extend(neg_inputs_b)
                    labels.extend([0] * len(neg_inputs_a))

    assert len(inputs_a) == len(inputs_b) == len(labels)
    logger.info(f"Preparing {len(labels)} samples...")
    model_inputs = tokenizer(inputs_a, text_pair=inputs_b,
                             padding=PaddingStrategy.LONGEST,
                             truncation=True,
                             max_length=max_input_length,
                             return_tensors="pt")
    model_inputs["labels"] = torch.tensor(labels, dtype=torch.long)

    return DictTensorDataset(model_inputs)


def lsat_triplet_binary_cls(file_path: str, tokenizer: PreTrainedTokenizer, max_input_length: int):
    data = json.load(open(file_path, 'r'))
    inputs_a = []
    inputs_b = []
    labels = []
    for item in data:
        if "examples" not in item:
            continue
        for ex in item["examples"]:
            if isinstance(ex["neg_sent"], str):
                inputs_a.append(" ".join(ex["rest_sentences"]))
                inputs_b.append(ex["ori_sent"])
                labels.append(1)

                inputs_a.append(" ".join(ex["rest_sentences"]))
                inputs_b.append(ex["neg_sent"])
                labels.append(0)
            elif isinstance(ex["neg_sent"], list):
                for neg_sent in ex["neg_sent"]:
                    inputs_a.append(" ".join(ex["rest_sentences"]))
                    inputs_b.append(ex["ori_sent"])
                    labels.append(1)

                    inputs_a.append(" ".join(ex["rest_sentences"]))
                    inputs_b.append(neg_sent)
                    labels.append(0)
            else:
                raise ValueError("neg_sent is not str or list")

    assert len(inputs_a) == len(inputs_b) == len(labels)
    logger.info(f"Preparing {len(labels)} samples...")
    model_inputs = tokenizer(inputs_a, text_pair=inputs_b,
                             padding=PaddingStrategy.LONGEST,
                             truncation=True,
                             max_length=max_input_length,
                             return_tensors="pt")
    model_inputs["labels"] = torch.tensor(labels, dtype=torch.long)
    return DictTensorDataset(model_inputs)


def lsat_sentence_pair_binary_cls_from_examples(file_path: str, tokenizer: PreTrainedTokenizer, max_seq_length: int):
    from data.examples import LSATDeductionExamples

    all_input_a = []
    all_input_b = []
    examples = LSATDeductionExamples()
    for item in examples.data:
        passage = item["passage"]
        sentences = sent_tokenize(passage)
        if item["positive_deductions"] and item["negative_deductions"]:
            for pos in item["positive_deductions"]:
                # all_input_a.append(passage)
                # all_input_b.append(pos)
                all_input_a.extend(sentences)
                all_input_b.extend([pos] * len(sentences))
            for neg in item["negative_deductions"]:
                # all_input_a.append(passage)
                # all_input_b.append(neg)
                all_input_a.extend(sentences)
                all_input_b.extend([neg] * len(sentences))
        for q in item["questions"]:
            question = q["question"]
            if q["positive_deductions"] and q["negative_deductions"]:
                for pos in q["positive_deductions"]:
                    # all_input_a.append(passage + ' ' + question)
                    # all_input_b.append(pos)
                    all_input_a.extend([sent + ' ' + question for sent in sentences])
                    all_input_b.extend([pos] * len(sentences))
                for neg in q["negative_deductions"]:
                    # all_input_a.append(passage + ' ' + question)
                    # all_input_b.append(neg)
                    all_input_a.extend([sent + ' ' + question for sent in sentences])
                    all_input_b.extend([neg] * len(sentences))

    model_inputs = tokenizer(all_input_a, text_pair=all_input_b,
                             max_length=max_seq_length, return_tensors="pt",
                             padding=PaddingStrategy.LONGEST, truncation=True)

    return DictTensorDataset(model_inputs)


def mcqa_to_binary_cls(read_func, file_path: str, tokenizer: PreTrainedTokenizer, max_seq_length: int, expand: bool = True):
    tokenizer_name = tokenizer_get_name(tokenizer)

    file_suffix = f"{tokenizer_name}_{max_seq_length}_{read_func.__class__.__name__}{'_no_expand' if not expand else ''}_seq_cls"
    cached_file_path = f"{file_path}_{file_suffix}"
    if os.path.exists(cached_file_path):
        logger.info(f"Loading cached file from {cached_file_path}.")
        data = torch.load(cached_file_path)
        return DictTensorDataset(data)

    all_context, all_question, all_option_list, all_label = read_func(file_path)

    max_option_num = max(map(len, all_option_list))
    logger.info(f"Max option num: {max_option_num}")

    all_inputs_a = []
    all_inputs_b = []
    labels = []

    for c, q, op_ls, label in zip(all_context, all_question, all_option_list, all_label):
        c_q = c + ' ' + q
        for op_id, op in enumerate(op_ls):
            if op_id == label:
                if expand:
                    all_inputs_a.extend([c_q] * (max_option_num - 1))
                    all_inputs_b.extend([op] * (max_option_num - 1))
                    labels.extend([1] * (max_option_num - 1))
                else:
                    all_inputs_a.append(c_q)
                    all_inputs_b.append(op)
                    labels.append(1)
            else:
                all_inputs_a.append(c_q)
                all_inputs_b.append(op)
                labels.append(0)

    model_inputs = tokenizer(all_inputs_a,
                             text_pair=all_inputs_b,
                             max_length=max_seq_length,
                             padding=PaddingStrategy.LONGEST,
                             return_tensors="pt")

    model_inputs["labels"] = torch.tensor(labels, dtype=torch.long)

    logger.info(f"Saving processed tensors into {cached_file_path}.")
    torch.save(model_inputs, cached_file_path)

    dataset = DictTensorDataset(model_inputs)
    return dataset
