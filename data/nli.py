import os.path

import torch
from torch.utils.data import TensorDataset
from transformers import PreTrainedTokenizer
from transformers.tokenization_utils_base import PaddingStrategy, TruncationStrategy, TensorType

from data.data_utils import tokenizer_get_name, get_sep_tokens
from general_util.logger import get_child_logger

logger = get_child_logger("NLI")


def nli_get_tensor(read_func, file_path: str, tokenizer: PreTrainedTokenizer, max_seq_length: int):
    tokenizer_name = tokenizer_get_name(tokenizer)

    file_suffix = f"{tokenizer_name}_{max_seq_length}_nli"
    cached_file_path = f"{file_path}_{file_suffix}"
    if os.path.exists(cached_file_path):
        logger.info(f"Loading cached file from {cached_file_path}.")
        tensors = torch.load(cached_file_path)
        return TensorDataset(*tensors)

    all_facts, all_rules, all_statements, all_labels = read_func(file_path)

    context = []
    query = []
    for fact_ls, rule_ls, statement in zip(all_facts, all_rules, all_statements):
        context.append(' '.join(fact_ls) + ''.join(get_sep_tokens(tokenizer)) + ' '.join(rule_ls))
        query.append(statement)

    tokenizer_outputs = tokenizer(context,
                                  text_pair=query,
                                  max_length=max_seq_length,
                                  padding=PaddingStrategy.MAX_LENGTH,
                                  truncation=TruncationStrategy.LONGEST_FIRST,
                                  return_tensors=TensorType.PYTORCH)

    data_num = len(all_labels)

    input_ids = tokenizer_outputs['input_ids'].reshape(data_num, max_seq_length)
    attention_mask = tokenizer_outputs['attention_mask'].reshape(data_num, max_seq_length)
    labels = torch.tensor(all_labels, dtype=torch.long).reshape(data_num)
    if 'token_type_ids' in tokenizer_outputs:
        token_type_ids = tokenizer_outputs['token_type_ids']
        inputs = (input_ids, attention_mask, token_type_ids, labels)
    else:
        inputs = (input_ids, attention_mask, labels)

    logger.info(f"Saving processed tensors into {cached_file_path}.")
    torch.save(inputs, cached_file_path)

    dataset = TensorDataset(*inputs)
    return dataset
