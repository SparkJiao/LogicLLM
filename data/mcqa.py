import copy
import json
import os.path
from collections import defaultdict
from typing import Dict, List, Tuple

import torch
from torch.utils.data import TensorDataset
from tqdm import tqdm
from transformers import PreTrainedTokenizer
from transformers.tokenization_utils_base import PaddingStrategy, TruncationStrategy, TensorType

from data.data_utils import tokenizer_get_name, get_sep_tokens, get_unused_tokens
from data.collators.lsat_span import LSATSpanDataset
from data.collators.dict2dict import DictTensorDataset
from general_util.logger import get_child_logger

logger = get_child_logger("MCQA")


def is_alphabet(char):
    res = ord('a') <= ord(char) <= ord('z') or ord('A') <= ord(char) <= ord('Z')
    return res


def replace_entity(sentence: str, entity: str, replacement: str):
    if len(entity) == 1:  # Process single character entity uniquely.
        s = sentence.find(entity)
        while s != -1:
            a = not is_alphabet(sentence[s - 1]) if s > 0 else 'jump'
            b = not is_alphabet(sentence[s + 1]) if s + 1 < len(sentence) else 'jump'
            # print(f"DEBUG: {s} [{sentence[s]}] {a} {b}")
            # if a and a != 'jump':
            #     print(f"DEBUG: {sentence[s - 1]}")
            # if b and b != 'jump':
            #     print(f"DEBUG: {sentence[s + 1]}")
            if a and b:
                # print(a, b)
                sentence = sentence[:s] + replacement + sentence[s + 1:]
                s = sentence.find(entity, s + len(replacement))
            else:
                s = sentence.find(entity, s + 1)
        return sentence

    assert replacement.strip() != ""
    return sentence.replace(entity, replacement)


def multiple_choice_get_tensor(read_func, file_path: str, tokenizer: PreTrainedTokenizer, max_seq_length: int, prefix_token_num: int = 0):
    tokenizer_name = tokenizer_get_name(tokenizer)

    file_suffix = f"{tokenizer_name}_{max_seq_length}_{read_func.__class__.__name__}" \
                  f"{'_prefix{}'.format(str(prefix_token_num)) if prefix_token_num > 0 else ''}_mc"
    cached_file_path = f"{file_path}_{file_suffix}"
    if os.path.exists(cached_file_path):
        logger.info(f"Loading cached file from {cached_file_path}.")
        tensors = torch.load(cached_file_path)
        return TensorDataset(*tensors)

    all_context, all_question, all_option_list, all_label = read_func(file_path)

    max_option_num = max(map(len, all_option_list))
    logger.info(f"Max option num: {max_option_num}")
    context = []
    q_op = []
    c_q_op_mask = []

    _prefix = ' '.join(get_unused_tokens(tokenizer, prefix_token_num)) if prefix_token_num > 0 else ''
    for c, q, op_ls, label in zip(all_context, all_question, all_option_list, all_label):
        c_q_op_mask.extend([1] * len(op_ls))

        if len(op_ls) < max_option_num:
            op_num = len(op_ls)
            op_ls.extend([' '] * (max_option_num - op_num))
            c_q_op_mask.extend([0] * (max_option_num - op_num))
        assert len(op_ls) == max_option_num

        context.extend([c] * len(op_ls))

        q_op.extend(list(map(lambda x: _prefix + q + ' '.join(get_sep_tokens(tokenizer)) + x, op_ls)))
        assert len(context) == len(q_op), (len(context), len(q_op))

    tokenizer_outputs = tokenizer(context,
                                  text_pair=q_op,
                                  max_length=max_seq_length,
                                  padding=PaddingStrategy.LONGEST,
                                  truncation=TruncationStrategy.LONGEST_FIRST,
                                  return_tensors=TensorType.PYTORCH)
    max_seq_length = min(max_seq_length, tokenizer_outputs["input_ids"].size(-1))

    data_num = len(all_context)

    input_ids = tokenizer_outputs['input_ids'].reshape(data_num, max_option_num, max_seq_length)
    attention_mask = tokenizer_outputs['attention_mask'].reshape(data_num, max_option_num, max_seq_length)
    op_mask = torch.tensor(c_q_op_mask, dtype=torch.long).reshape(data_num, max_option_num)
    labels = torch.tensor(all_label, dtype=torch.long).reshape(data_num)
    if 'token_type_ids' in tokenizer_outputs:
        token_type_ids = tokenizer_outputs['token_type_ids'].reshape(data_num, max_option_num, max_seq_length)
        inputs = (input_ids, attention_mask, token_type_ids, op_mask, labels)
    else:
        inputs = (input_ids, attention_mask, op_mask, labels)

    logger.info(f"Sequence length: {input_ids.size(2)}")

    logger.info(f"Saving processed tensors into {cached_file_path}.")
    torch.save(inputs, cached_file_path)

    dataset = TensorDataset(*inputs)
    return dataset


def multiple_choice_get_tensor_index(read_func, file_path: str, tokenizer: PreTrainedTokenizer, max_seq_length: int,
                                     prefix_token_num: int = 0):
    tokenizer_name = tokenizer_get_name(tokenizer)

    file_suffix = f"{tokenizer_name}_{max_seq_length}_{read_func.__class__.__name__}" \
                  f"{'_prefix{}'.format(str(prefix_token_num)) if prefix_token_num > 0 else ''}_mc"
    cached_file_path = f"{file_path}_{file_suffix}"
    if os.path.exists(cached_file_path):
        logger.info(f"Loading cached file from {cached_file_path}.")
        tensors = torch.load(cached_file_path)
        if len(tensors) == 4:
            input_ids, attention_mask, op_mask, labels = tensors
            data = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "op_mask": op_mask,
                "labels": labels
            }
        elif len(tensors) == 5:
            input_ids, attention_mask, token_type_ids, op_mask, labels = tensors
            data = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
                "op_mask": op_mask,
                "labels": labels
            }
        else:
            raise RuntimeError()
        meta_data = {"index": torch.arange(input_ids.size(0))}
        return DictTensorDataset(data, meta_data)

    all_context, all_question, all_option_list, all_label = read_func(file_path)

    max_option_num = max(map(len, all_option_list))
    logger.info(f"Max option num: {max_option_num}")
    context = []
    q_op = []
    c_q_op_mask = []

    _prefix = ' '.join(get_unused_tokens(tokenizer, prefix_token_num)) if prefix_token_num > 0 else ''
    for c, q, op_ls, label in zip(all_context, all_question, all_option_list, all_label):
        c_q_op_mask.extend([1] * len(op_ls))

        if len(op_ls) < max_option_num:
            op_num = len(op_ls)
            op_ls.extend([' '] * (max_option_num - op_num))
            c_q_op_mask.extend([0] * (max_option_num - op_num))
        assert len(op_ls) == max_option_num

        context.extend([c] * len(op_ls))

        q_op.extend(list(map(lambda x: _prefix + q + ' '.join(get_sep_tokens(tokenizer)) + x, op_ls)))
        assert len(context) == len(q_op), (len(context), len(q_op))

    tokenizer_outputs = tokenizer(context,
                                  text_pair=q_op,
                                  max_length=max_seq_length,
                                  padding=PaddingStrategy.LONGEST,
                                  truncation=TruncationStrategy.LONGEST_FIRST,
                                  return_tensors=TensorType.PYTORCH)
    max_seq_length = min(max_seq_length, tokenizer_outputs["input_ids"].size(-1))

    data_num = len(all_context)

    input_ids = tokenizer_outputs['input_ids'].reshape(data_num, max_option_num, max_seq_length)
    attention_mask = tokenizer_outputs['attention_mask'].reshape(data_num, max_option_num, max_seq_length)
    op_mask = torch.tensor(c_q_op_mask, dtype=torch.long).reshape(data_num, max_option_num)
    labels = torch.tensor(all_label, dtype=torch.long).reshape(data_num)
    if 'token_type_ids' in tokenizer_outputs:
        token_type_ids = tokenizer_outputs['token_type_ids'].reshape(data_num, max_option_num, max_seq_length)
        inputs = (input_ids, attention_mask, token_type_ids, op_mask, labels)
    else:
        inputs = (input_ids, attention_mask, op_mask, labels)
        token_type_ids = None

    logger.info(f"Sequence length: {input_ids.size(2)}")

    logger.info(f"Saving processed tensors into {cached_file_path}.")
    torch.save(inputs, cached_file_path)

    data = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "op_mask": op_mask,
        "labels": labels
    }
    if "token_type_ids" in tokenizer_outputs:
        data["token_type_ids"] = token_type_ids
    meta_data = {"index": torch.arange(input_ids.size(0))}
    return DictTensorDataset(data, meta_data)


def _parse_span_position(span: str, text: str) -> List[int]:
    span_ls = []
    s = text.find(span)
    while s != -1:
        a = not is_alphabet(text[s - 1]) if s > 0 else True
        b = not is_alphabet(text[s + len(span)]) if s + len(span) < len(text) else True
        if a and b:
            span_ls.append(s)
            s = text.find(span, s + len(span))
        else:
            s = text.find(span, s + 1)
    return span_ls


def _lsat_span_parse(item: Dict, tokenizer: PreTrainedTokenizer, max_seq_length: int) -> Tuple:
    """
    Data comes from ARM pre-processing.
    """
    context = item["context"]
    question = item["question"]
    answers = item["answers"]
    label = item["label"]
    participants = item["rows"]
    positions = item["columns"]
    id_string = item["id_string"]
    tags = item["tags"]

    _all_span_list_ctx = []
    for part in participants:
        _span_ls = _parse_span_position(part, context)
        _all_span_list_ctx.extend(list(map(lambda x: (x, part), _span_ls)))
    for pos in positions:
        _span_ls = _parse_span_position(pos, context)
        _all_span_list_ctx.extend(list(map(lambda x: (x, pos), _span_ls)))
    _sorted_all_span_list_ctx = sorted(_all_span_list_ctx, key=lambda x: x[0])

    tokens = []
    _all_span2token_index = defaultdict(list)
    _s = 0
    _span_offset = 1
    for _span in _sorted_all_span_list_ctx:
        tokens.extend(tokenizer.tokenize(context[_s:_span[0]]))
        _span_token_index = [len(tokens) + _span_offset, -1]

        if tokenizer.__class__.__name__ in [
            "RobertaTokenizer",
            "LongformerTokenizer",
            "BartTokenizer",
            "RobertaTokenizerFast",
            "LongformerTokenizerFast",
            "BartTokenizerFast",
        ]:
            tokens.extend(tokenizer.tokenize(' ' + _span[1]))
        else:
            tokens.extend(tokenizer.tokenize(_span[1]))

        _span_token_index[1] = len(tokens) + _span_offset
        _all_span2token_index[_span[1]].append(_span_token_index)

        _s = _span[0] + len(_span[1])
    tokens.extend(tokenizer.tokenize(context[_s:]))

    _span_offset = 1 + len(tokens) + len(get_sep_tokens(tokenizer))  # <cls> context tokens <sep> <sep> ....

    input_ids = []
    attention_mask = []
    token_type_ids = []
    span2sub_word_index = []
    for option in answers:
        q_op_text = question + ' '.join(get_sep_tokens(tokenizer)) + option

        _all_span_list_q_op = []
        for part in participants:
            _span_ls = _parse_span_position(part, q_op_text)
            _all_span_list_q_op.extend(list(map(lambda x: (x, part), _span_ls)))
        for pos in positions:
            _span_ls = _parse_span_position(pos, q_op_text)
            _all_span_list_q_op.extend(list(map(lambda x: (x, pos), _span_ls)))
        _sorted_all_span_list_q_op = sorted(_all_span_list_q_op, key=lambda x: x[0])

        _q_op_span2token_index = copy.deepcopy(_all_span2token_index)

        _op_tokens = []
        _s = 0
        for _span in _sorted_all_span_list_q_op:
            _op_tokens.extend(tokenizer.tokenize(q_op_text[_s:_span[0]]))
            _span_token_index = [len(_op_tokens) + _span_offset, -1]
            if tokenizer.__class__.__name__ in [
                "RobertaTokenizer",
                "LongformerTokenizer",
                "BartTokenizer",
                "RobertaTokenizerFast",
                "LongformerTokenizerFast",
                "BartTokenizerFast",
            ]:
                _op_tokens.extend(tokenizer.tokenize(' ' + _span[1]))
            else:
                _op_tokens.extend(tokenizer.tokenize(_span[1]))

            _span_token_index[1] = len(_op_tokens) + _span_offset
            _q_op_span2token_index[_span[1]].append(_span_token_index)

            _s = _span[0] + len(_span[1])
        _op_tokens.extend(tokenizer.tokenize(q_op_text[_s:]))

        text_a = tokenizer.convert_tokens_to_string(tokens)
        text_b = tokenizer.convert_tokens_to_string(_op_tokens)

        tokenizer_outputs = tokenizer(text_a, text_b, max_length=max_seq_length,
                                      truncation=TruncationStrategy.LONGEST_FIRST, padding=PaddingStrategy.MAX_LENGTH)
        assert isinstance(tokenizer_outputs['input_ids'], list)
        assert isinstance(tokenizer_outputs['input_ids'][0], int)

        input_ids.append(tokenizer_outputs["input_ids"])
        attention_mask.append(tokenizer_outputs["attention_mask"])
        if "token_type_ids" in tokenizer_outputs:
            token_type_ids.append(tokenizer_outputs["token_type_ids"])

        # Filter
        filtered_span2token_index = {}
        for _span, _span_index_list in _q_op_span2token_index.items():
            filtered_span2token_index[_span] = []
            for _span_index in _span_index_list:
                if _span_index[1] <= max_seq_length:
                    filtered_span2token_index[_span].append(_span_index)
        span2sub_word_index.append(filtered_span2token_index)

    participant_span2token_index = [[] for _ in range(len(answers))]
    position_span2token_index = [[] for _ in range(len(answers))]
    for op_id in range(len(answers)):
        for part in participants:
            if part in span2sub_word_index[op_id]:
                participant_span2token_index[op_id].append(span2sub_word_index[op_id][part])
        for pos in positions:
            if pos in span2sub_word_index[op_id]:
                position_span2token_index[op_id].append(span2sub_word_index[op_id][pos])

    if len(token_type_ids):
        return input_ids, attention_mask, token_type_ids, participant_span2token_index, position_span2token_index, label
    else:
        return input_ids, attention_mask, participant_span2token_index, position_span2token_index, label


def lsat_part_group_span_get_tensor(file_path: str, tokenizer: PreTrainedTokenizer, max_seq_length: int):
    tokenizer_name = tokenizer_get_name(tokenizer)

    file_suffix = f"{tokenizer_name}_{max_seq_length}_lsat_group_span_v1"
    cached_file_path = f"{file_path}_{file_suffix}"
    if os.path.exists(cached_file_path):
        logger.info(f"Loading cached file from {cached_file_path}.")
        data = torch.load(cached_file_path)
        return LSATSpanDataset(*data)

    data = json.load(open(file_path))

    input_ids = []
    attention_mask = []
    token_type_ids = []
    part_span2token_index = []
    pos_span2token_index = []
    labels = []
    for item in tqdm(data, total=len(data)):
        res = _lsat_span_parse(item, tokenizer, max_seq_length)
        if len(res) == 6:
            input_ids.append(res[0])
            attention_mask.append(res[1])
            token_type_ids.append(res[2])
            part_span2token_index.append(res[3])
            pos_span2token_index.append(res[4])
            labels.append(res[5])
        else:
            input_ids.append(res[0])
            attention_mask.append(res[1])
            part_span2token_index.append(res[2])
            pos_span2token_index.append(res[3])
            labels.append(res[4])

    max_option_num = max(map(lambda x: len(x), input_ids))
    pt_input_ids = torch.zeros(len(input_ids), max_option_num, max_seq_length, dtype=torch.long)
    pt_attention_mask = torch.zeros(len(input_ids), max_option_num, max_seq_length, dtype=torch.long)
    pt_token_type_ids = torch.zeros(len(input_ids), max_option_num, max_seq_length, dtype=torch.long)
    option_mask = torch.zeros(len(input_ids), max_option_num, dtype=torch.long)

    for i, item in enumerate(input_ids):
        op_num = len(item)
        option_mask[i, :op_num] = 1
        pt_input_ids[i, :op_num] = torch.tensor(item)
        pt_attention_mask[i, :op_num] = torch.tensor(attention_mask[i])
        if len(token_type_ids) > 0:
            pt_token_type_ids[i, :op_num] = torch.tensor(token_type_ids[i])
    labels = torch.tensor(labels, dtype=torch.long)

    logger.info(f"Saving processed tensors into {cached_file_path}.")
    torch.save((pt_input_ids, pt_attention_mask, None if len(token_type_ids) == 0 else pt_token_type_ids,
                option_mask, labels, part_span2token_index, pos_span2token_index), cached_file_path)

    dataset = LSATSpanDataset(pt_input_ids, pt_attention_mask, None if len(token_type_ids) == 0 else pt_token_type_ids,
                              option_mask, labels, part_span2token_index, pos_span2token_index)

    return dataset
