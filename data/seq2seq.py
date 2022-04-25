import glob
import json
import os.path
import jsonlines

import torch
from transformers import PreTrainedTokenizer, GPT2Tokenizer, GPT2TokenizerFast
from transformers.tokenization_utils_base import PaddingStrategy

from data.collators.dict2dict import DictTensorDataset
from data.data_utils import tokenizer_get_name, recursive_find_path, recursive_bfs
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

    # file_suffix = f"{tokenizer_name}_{max_input_length}_{read_func.__class__.__name__}_seq2seq_test"
    # cached_file_path = f"{file_path}_{file_suffix}"
    # if os.path.exists(cached_file_path):
    #     logger.info(f"Loading cached file from {cached_file_path}.")
    #     tensors = torch.load(cached_file_path)
    #     return TensorDataset(*tensors)

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


def lsat_annotation_get_tensor(file_path: str, tokenizer: PreTrainedTokenizer, max_input_length: int,
                               prefix: str = "generate logically consistent deductions: "):
    if os.path.exists(file_path):
        logger.info(f"Loading file from {file_path}.")
        data = json.load(open(file_path, 'r'))
        if isinstance(data, dict):
            data = [data]
    else:
        data = []
        for file in glob.glob(file_path):
            logger.info(f"Loading {file}.")
            item = json.load(open(file, 'r'))
            assert isinstance(item, dict)
            data.append(item)

    inputs = []
    outputs = []

    for item in data:
        passage = item['passage']
        if "deductions" in passage:
            context_deductions = []
            recursive_find_path(item["deductions"], context_deductions, res=[])

            for ctx_deduction_path in context_deductions:
                acc_passage = passage
                for ctx_deduction in ctx_deduction_path:
                    inputs.append(prefix + acc_passage)
                    outputs.append(ctx_deduction)
                    acc_passage = acc_passage + ' ' + ctx_deduction
                    assert isinstance(ctx_deduction, str)
            all_deduction_text = recursive_bfs(item["deductions"])
        else:
            all_deduction_text = ''

        for q in item['questions']:
            ques = q['question']
            if "deductions" in q:
                q_deductions = []
                recursive_find_path(q["deductions"], q_deductions, res=[])

                for q_deduction_path in q_deductions:
                    acc_ques = ques
                    for q_deduction in q_deduction_path:
                        inputs.append(prefix + passage + ' ' + all_deduction_text + ' ' + acc_ques)
                        outputs.append(q_deduction)
                        acc_ques = acc_ques + ' ' + q_deduction
                        assert isinstance(q_deduction, str)

    assert len(inputs) == len(outputs)
    logger.info(f"Preparing {len(inputs)} samples...")
    model_inputs = tokenizer(inputs,
                             padding=PaddingStrategy.LONGEST,
                             truncation=True,
                             max_length=max_input_length,
                             return_tensors="pt")

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(outputs, return_tensors="pt",
                           padding=PaddingStrategy.LONGEST,
                           truncation=True, max_length=max_input_length)["input_ids"]
    model_inputs["labels"] = labels

    return DictTensorDataset(model_inputs)


def lsat_weak_supervision_seq2seq_get_tensor(file_path: str, tokenizer: PreTrainedTokenizer, max_input_length: int,
                                             prefix: str = 'generate logically consistent deductions: '):
    label2id = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}

    data = json.load(open(file_path, 'r'))
    inputs = []
    outputs = []

    for item in data:
        passage = item['passage']
        for q in item['questions']:
            ques = q['question']
            if 'if' not in ques.lower():
                inputs.append(prefix + ' ' + passage + ' ' + ques)
                outputs.append(q['options'][label2id[q['answer']]])

    assert len(inputs) == len(outputs)
    logger.info(f"Preparing {len(inputs)} samples...")
    model_inputs = tokenizer(inputs,
                             padding=PaddingStrategy.LONGEST,
                             truncation=True,
                             max_length=max_input_length,
                             return_tensors="pt")
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(outputs, return_tensors="pt",
                           padding=PaddingStrategy.LONGEST,
                           truncation=True, max_length=max_input_length)["input_ids"]
    model_inputs["labels"] = labels

    return DictTensorDataset(model_inputs)


class ExampleLoadFuncV1:
    def __call__(self, data):
        examples = []
        for item in data:
            passage = item['passage']
            if "deductions" in passage:
                context_deductions = []
                recursive_find_path(item["deductions"], context_deductions, res=[])

                for ctx_deduction_path in context_deductions:
                    acc_passage = passage
                    for ctx_deduction in ctx_deduction_path:
                        examples.append(f"Input: {acc_passage} Deduction: {ctx_deduction}")
                        acc_passage = acc_passage + ' ' + ctx_deduction
                        assert isinstance(ctx_deduction, str)
                all_deduction_text = recursive_bfs(item["deductions"])
            else:
                all_deduction_text = ''

            for q in item['questions']:
                ques = q['question']
                if "deductions" in q:
                    q_deductions = []
                    recursive_find_path(q["deductions"], q_deductions, res=[])

                    for q_deduction_path in q_deductions:
                        acc_ques = ques
                        for q_deduction in q_deduction_path:
                            # inputs.append(prefix + passage + ' ' + all_deduction_text + ' ' + acc_ques)
                            # outputs.append(q_deduction)
                            examples.append(f"Input: {passage + ' ' + all_deduction_text + ' ' + acc_ques} Deduction: {q_deduction}")
                            acc_ques = acc_ques + ' ' + q_deduction
                            assert isinstance(q_deduction, str)

        return examples


class ExampleLoadFuncV2:
    def __call__(self, data):
        examples = []
        for item in data:
            passage = item['passage']
            if "deductions" in passage:
                # recursive_find_path(item["deductions"], context_deductions, res=[])
                #
                # for ctx_deduction_path in context_deductions:
                #     acc_passage = passage
                #     for ctx_deduction in ctx_deduction_path:
                #         examples.append(f"Input: {acc_passage} Deduction: {ctx_deduction}")
                #         acc_passage = acc_passage + ' ' + ctx_deduction
                #         assert isinstance(ctx_deduction, str)
                all_deduction_text = recursive_bfs(item["deductions"])

                examples.append(f"Input: {passage} Deduction: {all_deduction_text}")
            else:
                all_deduction_text = ''

            for q in item['questions']:
                ques = q['question']
                if "deductions" in q:
                    q_deductions = []
                    recursive_find_path(q["deductions"], q_deductions, res=[])

                    for q_deduction_path in q_deductions:
                        acc_ques = ques
                        for q_deduction in q_deduction_path:
                            # inputs.append(prefix + passage + ' ' + all_deduction_text + ' ' + acc_ques)
                            # outputs.append(q_deduction)
                            examples.append(f"Input: {passage + ' ' + all_deduction_text + ' ' + acc_ques} Deduction: {q_deduction}")
                            acc_ques = acc_ques + ' ' + q_deduction
                            assert isinstance(q_deduction, str)

        return examples


def lm_prompt_get_tensor(file_path: str, example_file_path: str, example_load_func, tokenizer: PreTrainedTokenizer,
                         read_func, max_input_length: int, max_example_num: int,
                         prefix: str = 'generate logically consistent deductions: '):
    # Load the examples first.
    if os.path.exists(example_file_path):
        logger.info(f"Loading file from {example_file_path}.")
        data = json.load(open(example_file_path, 'r'))
        if isinstance(data, dict):
            data = [data]
    else:
        data = []
        for file in glob.glob(example_file_path):
            logger.info(f"Loading {file}.")
            item = json.load(open(file, 'r'))
            assert isinstance(item, dict)
            data.append(item)

    prompt = prefix + ' '.join(example_load_func(data)[:max_example_num])
    tokens = tokenizer.tokenize(prompt)
    logger.info("Length of prompt: %d", len(tokens))

    all_context, all_question, _, _ = read_func(file_path)
    inputs = []
    for c, q in zip(all_context, all_question):
        inputs.append(f"{prompt} Input: {c} {q} Deduction: ")

    if isinstance(tokenizer, GPT2TokenizerFast) or isinstance(tokenizer, GPT2Tokenizer):
        tokenizer.pad_token = tokenizer.eos_token

    model_inputs = tokenizer(inputs,
                             padding=PaddingStrategy.LONGEST,
                             truncation=True,
                             max_length=max_input_length,
                             return_tensors="pt")

    return DictTensorDataset(model_inputs)


def proof_writer_stage_pure_loading(file_path: str, tokenizer: PreTrainedTokenizer, max_input_length: int, max_output_length: int,
                                    prefix: str = ''):
    tokenizer_name = tokenizer_get_name(tokenizer)

    file_suffix = f"{tokenizer_name}_{max_input_length}_{max_output_length}_proof_writer_stage_pure_{prefix.replace(' ', '-')}_v1"
    cached_file_path = f"{file_path}_{file_suffix}"
    cached_file_path = cached_file_path.replace('*', '')
    if os.path.exists(cached_file_path):
        logger.info(f"Loading cached file from {cached_file_path}.")
        data = torch.load(cached_file_path)
        return DictTensorDataset(data)

    inputs = []
    outputs = []

    if os.path.exists(file_path):
        files = [file_path]
    else:
        files = glob.glob(file_path)

    for file in files:
        logger.info(f"Loading {file}.")
        with jsonlines.open(file) as reader:
            for line in reader:
                triples = ' '.join(list(map(lambda x: x[1]['text'], line['triples'].items())))
                rules = ' '.join(list(map(lambda x: x[1]['text'], line['rules'].items())))
                context = f'{prefix} {triples} {rules}'
                for inference in line['allInferences']:  # Keep only one inference is decoded at once.
                    inputs.append(context)
                    outputs.append(inference['text'])

    model_inputs = tokenizer(inputs, padding=PaddingStrategy.LONGEST, truncation=True, max_length=max_input_length,
                             return_tensors="pt")
    with tokenizer.as_target_tokenizer():
        model_outputs = tokenizer(outputs, padding=PaddingStrategy.LONGEST, truncation=True, max_length=max_output_length,
                                  return_tensors="pt")
        model_inputs["labels"] = model_outputs["input_ids"]

    logger.info(f"Saving to {cached_file_path}.")
    torch.save(model_inputs, cached_file_path)

    return DictTensorDataset(model_inputs)
