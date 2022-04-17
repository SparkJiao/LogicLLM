import os.path

import torch
from torch.utils.data import TensorDataset
from transformers import PreTrainedTokenizer
from transformers.tokenization_utils_base import PaddingStrategy

from data.collators.dict2dict import DictTensorDataset
from data.data_utils import tokenizer_get_name
from general_util.logger import get_child_logger

logger = get_child_logger("Seq2Seq")


def generation_get_tensor_test(read_func, file_path: str, tokenizer: PreTrainedTokenizer, max_input_length: int,
                               prefix: str = "generate logically consistent deductions: "):
    tokenizer_name = tokenizer_get_name(tokenizer)

    if tokenizer_name == 't5':
        sep_token = "<extra_id_0>"
    elif tokenizer_name == 'bart':
        sep_token = "<s>"
    else:
        raise RuntimeError("Unsupported tokenizer {}".format(tokenizer.__class__.__name__))

    file_suffix = f"{tokenizer_name}_{max_input_length}_{read_func.__class__.__name__}_seq2seq_test"
    cached_file_path = f"{file_path}_{file_suffix}"
    if os.path.exists(cached_file_path):
        logger.info(f"Loading cached file from {cached_file_path}.")
        tensors = torch.load(cached_file_path)
        return TensorDataset(*tensors)

    all_context, all_question, all_option_list, all_label = read_func(file_path)

    inputs = []
    for c, q in zip(all_context, all_question):
        inputs.append(prefix + c + sep_token + q)

    model_inputs = tokenizer(inputs,
                             padding=PaddingStrategy.LONGEST,
                             truncation=True,
                             max_length=max_input_length,
                             return_tensors="pt")

    return DictTensorDataset(model_inputs)
