import os.path

import torch
from torch.utils.data import TensorDataset
from transformers import PreTrainedTokenizer
from transformers.tokenization_utils_base import PaddingStrategy, TruncationStrategy, TensorType

from data.data_utils import tokenizer_get_name, get_sep_tokens
from general_util.logger import get_child_logger

logger = get_child_logger("MCQA")


def multiple_choice_get_tensor(read_func, file_path: str, tokenizer: PreTrainedTokenizer, max_seq_length: int):
    tokenizer_name = tokenizer_get_name(tokenizer)

    file_suffix = f"{tokenizer_name}_{max_seq_length}_{read_func.__class__.__name__}_mc"
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

    for c, q, op_ls, label in zip(all_context, all_question, all_option_list, all_label):
        c_q_op_mask.extend([1] * len(op_ls))

        if len(op_ls) < max_option_num:
            op_num = len(op_ls)
            op_ls.extend([' '] * (max_option_num - op_num))
            c_q_op_mask.extend([0] * (max_option_num - op_num))
        assert len(op_ls) == max_option_num

        context.extend([c] * len(op_ls))
        q_op.extend(list(map(lambda x: q + ' '.join(get_sep_tokens(tokenizer)) + x, op_ls)))
        assert len(context) == len(q_op), (len(context), len(q_op))

    tokenizer_outputs = tokenizer(context,
                                  text_pair=q_op,
                                  max_length=max_seq_length,
                                  padding=PaddingStrategy.MAX_LENGTH,
                                  truncation=TruncationStrategy.LONGEST_FIRST,
                                  return_tensors=TensorType.PYTORCH)

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

    logger.info(f"Saving processed tensors into {cached_file_path}.")
    torch.save(inputs, cached_file_path)

    dataset = TensorDataset(*inputs)
    return dataset
