import json
import random
from typing import List, Dict, Callable

import torch
from torch.utils.data import Dataset, default_collate
from transformers import PreTrainedTokenizer, AutoTokenizer

from data.collators.flan import convert_to_standard_inputs
from data.cot_critic import vanilla_seq2seq_convertor as vanilla_seq2seq_converter_text
from general_util.logger import get_child_logger
from general_util.tokenization_utils import expand_special_tokenizer, is_seq2seq_tokenizer

logger = get_child_logger(__name__)

_default_instruct = "Answer the following question with the given context:"

_rank2option = ["A", "B", "C", "D", "E"]

instruction_list = {
    "default": _default_instruct,
    "vicuna_style": "Below is an instruction that describes a task. Write a response that appropriately completes the request."
                    "\n\n### Instruction: ",
}


def get_instruction_by_name(name: str):
    return instruction_list[name]


def compose_flat_prompt_input(context, question, option_list) -> str:
    context = "Context:\n" + context
    question = "Question:\n" + question
    option_list = option_list
    options = "Options:\n" + "\n".join([chr(ord('A') + i) + ": " + option for i, option in enumerate(option_list)])
    return "\n\n".join([context, question, options])


def read_cot_prompts(data_file, prediction_file, sample_num: int = 5, random_sample: bool = False, self_consistency: bool = True):
    data = json.load(open(data_file, 'r'))
    predictions = json.load(open(prediction_file, 'r'))
    prompts = []
    for item, pred in zip(data, predictions):
        assert item["id_string"] == pred["id"]
        if not self_consistency or item["label"] == int(ord(pred["pred"]) - ord("A")):
            inputs = compose_flat_prompt_input(item["context"], item["question"], item["answers"])
            response = pred["response"]
            prompts.append(inputs + "\n\n" + response)
    if random_sample:
        random.shuffle(prompts)
    logger.info(f"Filtered {len(prompts)} prompts.")

    def _callable():
        return prompts[:sample_num]

    return _callable


def read_cot_prompts_logiqa_v2(data_file, prediction_file, sample_num: int = 5, random_sample: bool = False, self_consistency: bool = True,
                               instruct: str = _default_instruct):
    data = open(data_file, 'r').readlines()
    predictions = json.load(open(prediction_file, 'r'))
    prompts = []
    for line, pred in zip(data, predictions):
        item = json.loads(line)
        assert item["id"] == pred["id"]
        if not self_consistency or pred["pred"].strip() and item["answer"] == int(ord(pred["pred"].strip()) - ord("A")):
            inputs = compose_flat_prompt_input(item["text"], item["question"], item["options"])
            response = pred["response"]
            prompts.append(inputs + "\n\n" + response)
    if random_sample:
        random.shuffle(prompts)
    logger.info(f"Filtered {len(prompts)} prompts.")

    def _callable():
        return instruct + "\n\n" + "\n\n".join(prompts[:sample_num]), list(range(sample_num))

    return _callable


def read_raw_prompts_logiqa_v2(data_file, sample_num: int = 5, random_sample: bool = False):
    data = open(data_file, 'r').readlines()
    prompts = []
    for line in data:
        item = json.loads(line)
        inputs = compose_flat_prompt_input(item["text"], item["question"], item["options"])
        inputs = inputs + "\n\n" + "The answer is " + chr(ord('A') + item["answer"]) + "."
        prompts.append(inputs)
    if random_sample:
        random.shuffle(prompts)
    return prompts[:sample_num]


def read_cot_prompts_logiqa_v2_category(data_file, prediction_file, test_file, sample_num: int = 5, random_sample: bool = False,
                                        cate_overlap_num: int = 2):
    data = list(map(json.loads, open(data_file, 'r').readlines()))
    predictions = json.load(open(prediction_file, 'r'))
    test_data = list(map(json.loads, open(test_file, 'r').readlines()))
    all_prompts = []
    for item, pred in zip(data, predictions):
        assert item["id"] == pred["id"]
        if pred["pred"].strip() and item["answer"] == int(ord(pred["pred"].strip()) - ord("A")):
            item["response"] = pred["response"]
            all_prompts.append(item)

    logger.info(f"Filtered {len(all_prompts)} prompts.")
    item_prompts = []
    less = 0
    for item_id, item in enumerate(test_data):
        item_reason_types = set([r for r, f in item["type"].items() if f])
        tmp = []
        for prompt in all_prompts:
            prompt_reason_types = set([r for r, f in prompt["type"].items() if f])
            if len(item_reason_types & prompt_reason_types) >= cate_overlap_num:
                tmp.append(prompt)
        if random_sample:
            random.shuffle(tmp)
        if len(tmp) < sample_num:
            less += 1
        if len(tmp) == 0:
            if random_sample:
                tmp = random.sample(all_prompts, sample_num)
            else:
                tmp = all_prompts[:sample_num]
        tmp = [compose_flat_prompt_input(x["text"], x["question"], x["options"]) + "\n\n" + x["response"]
               for x in tmp[:sample_num]]
        item_prompts.append(tmp)

    logger.info(f"Less than {sample_num} prompts: {less}")
    return item_prompts


def _format_option_list(option_list: List[str]) -> str:
    res = ""
    for op_id, op in enumerate(option_list):
        res += f"{_rank2option[op_id]}: {op}\n"
    return res


class ReClorExemplarGenerator:
    def __init__(self, file_path, read_func, shot: int = 0, random_sampling: bool = False, instruct: str = _default_instruct):
        self.shot = shot
        self.random = random_sampling
        all_context, all_question, all_option_list, all_label = read_func(file_path)
        index = list(range(len(all_context)))

        if self.random:
            random.shuffle(index)

        prompts = []
        for i in index[:self.shot]:
            prompt = f"Context:\n{all_context[i]}\n\nQuestion:\n{all_question[i]}\n\nOptions:\n{_format_option_list(all_option_list[i])}" \
                     f"\n\nThe answer is {_rank2option[all_label[i]]}"
            prompts.append(prompt)

        self.prompt = instruct + "\n\n" + "\n\n".join(prompts)
        self.indices = index[:self.shot]

    def __call__(self):
        return self.prompt, self.indices


class ReClorExemplarGeneratorZh:
    def __init__(self, file_path, read_func, shot: int = 0, random_sampling: bool = False, instruct: str = _default_instruct):
        self.shot = shot
        self.random = random_sampling
        all_context, all_question, all_option_list, all_label = read_func(file_path)
        index = list(range(len(all_context)))

        if self.random:
            random.shuffle(index)

        prompts = []
        for i in index[:self.shot]:
            prompt = f"文章：\n{all_context[i]}\n\n问题：\n{all_question[i]}\n\n选项：\n{_format_option_list(all_option_list[i])}" \
                     f"\n\n答案是 {_rank2option[all_label[i]]}"
            prompts.append(prompt)

        self.prompt = instruct + "\n\n" + "\n\n".join(prompts)
        self.indices = index[:self.shot]

    def __call__(self):
        return self.prompt, self.indices


class ReClorGenerativeDataset(Dataset):
    def __init__(self, file_path: str, tokenizer: PreTrainedTokenizer, read_func, prompt_generator, suffix: str = "The answer is",
                 max_data_num: int = -1,
                 api_based: bool = False,
                 service_based: bool = False, service_processor: Callable = None):
        self.prompt_generator = prompt_generator
        # read_func = ReClorReader()
        all_context, all_question, all_option_list, all_label = read_func(file_path)

        self.inputs = []
        self.indices = []
        self.labels = []
        for i in range(len(all_context)):
            prompt = f"Context:\n{all_context[i]}\n\nQuestion:\n{all_question[i]}\n\nOptions:\n{_format_option_list(all_option_list[i])}" \
                     f"\n\n" + suffix
            self.inputs.append(prompt)
            self.indices.append(i)
            self.labels.append(_rank2option[all_label[i]])

        self.max_data_num = max_data_num
        self.api_based = api_based
        self.service_based = service_based
        self.service_processor = service_processor

    def __len__(self):
        if self.max_data_num:
            return min(self.max_data_num, len(self.inputs))
        return len(self.inputs)

    def service_getitem(self, index):
        prompt, prompt_indices = self.prompt_generator()
        prompt = prompt + "\n\n" + self.inputs[index]
        response = self.service_processor(prompt)
        return {
            "input": prompt,
            "response": response,
            "meta_data": {
                "index": self.indices[index],
                "label": self.labels[index],
                "text": prompt,
                "response": response,
            }
        }

    def api_getitem(self, index):
        prompt, prompt_indices = self.prompt_generator()
        prompt = prompt + "\n\n" + self.inputs[index]
        return {
            "text": prompt,
            "meta_data": {
                "index": self.indices[index],
                "label": self.labels[index],
                "text": prompt,
            }
        }

    def __getitem__(self, index):
        if self.service_based:
            return self.service_getitem(index)
        if self.api_based:
            return self.api_getitem(index)
        prompt, prompt_indices = self.prompt_generator()
        return {
            "input": prompt + "\n\n" + self.inputs[index],
            "index": self.indices[index],
            "prompt_index": ",".join(map(str, prompt_indices)),
            "label": self.labels[index],
        }


class ReClorCandidateGenerativeDataset(Dataset):
    def __init__(self, file_path: str, tokenizer: PreTrainedTokenizer, read_func, prompt_generator, suffix: str = "The answer is",
                 add_option: bool = False):
        self.prompt_generator = prompt_generator
        all_context, all_question, all_option_list, all_label = read_func(file_path)

        is_seq2seq = is_seq2seq_tokenizer(tokenizer)
        logger.info("{} is seq2seq tokenizer: {}".format(tokenizer.__class__.__name__, is_seq2seq))

        self.inputs = []
        self.indices = []
        self.labels = []
        self.outputs = []
        for i in range(len(all_context)):
            prompt = f"Context:\n{all_context[i]}\n\nQuestion:\n{all_question[i]}\n\nOptions:\n{_format_option_list(all_option_list[i])}" \
                     f"\n\n" + suffix

            if add_option:
                targets = [f"{_rank2option[i]}: {all_option_list[i]}" for i in range(len(all_option_list[i]))]
            else:
                targets = [_rank2option[i] for i in range(len(all_option_list[i]))]
            if is_seq2seq:
                self.inputs.append([prompt] * len(targets))
                self.outputs.append(targets)
            else:
                self.inputs.append([prompt + f" {tgt}" for tgt in targets])
                self.outputs.append(targets)
            # if add_option:
            #     self.inputs.append([prompt + f" {_rank2option[i]}: {all_option_list[i]}" for i in range(len(all_option_list[i]))])
            # else:
            #     self.inputs.append([prompt + f" {_rank2option[i]}" for i in range(len(all_option_list[i]))])
            self.indices.append(i)
            self.labels.append(all_label[i])

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        prompt, prompt_indices = self.prompt_generator()
        return {
            "input": [prompt + "\n\n" + x for x in self.inputs[index]],
            "output": self.outputs[index],
            "index": self.indices[index],
            "prompt_index": ",".join(map(str, prompt_indices)),
            "label": self.labels[index],
        }


class ReClorCandidateGenerativeDatasetFlat(Dataset):
    def __init__(self, file_path: str, tokenizer: PreTrainedTokenizer, read_func, prompt_generator, suffix: str = "The answer is",
                 add_option: bool = False):
        self.prompt_generator = prompt_generator
        all_context, all_question, all_option_list, all_label = read_func(file_path)

        is_seq2seq = is_seq2seq_tokenizer(tokenizer)
        logger.info("{} is seq2seq tokenizer: {}".format(tokenizer.__class__.__name__, is_seq2seq))

        self.inputs = []
        self.indices = []
        self.labels = []
        self.outputs = []
        for i in range(len(all_context)):
            prompt = f"Context:\n{all_context[i]}\n\nQuestion:\n{all_question[i]}\n\nOptions:\n{_format_option_list(all_option_list[i])}" \
                     f"\n\n" + suffix

            if add_option:
                targets = [f"{_rank2option[i]}: {all_option_list[i]}" for i in range(len(all_option_list[i]))]
            else:
                targets = [_rank2option[i] for i in range(len(all_option_list[i]))]
            if is_seq2seq:
                self.inputs.extend([prompt] * len(targets))
                self.outputs.extend(targets)
            else:
                self.inputs.extend([prompt + f" {tgt}" for tgt in targets])
                self.outputs.extend(targets)

            self.indices.extend([f"{i}_{op_id}" for op_id in range(len(all_option_list[i]))])
            self.labels.extend([all_label[i]] * len(all_option_list[i]))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        prompt, prompt_indices = self.prompt_generator()
        return {
            "input": prompt + "\n\n" + self.inputs[index],
            "output": self.outputs[index],
            "index": self.indices[index],
            "prompt_index": ",".join(map(str, prompt_indices)),
            "label": self.labels[index],
        }


class ReClorCandidateSelectionDataset(Dataset):
    def __init__(self, file_path: str, tokenizer: PreTrainedTokenizer, read_func, prompt_generator, suffix: str = "The answer is",
                 add_option: bool = False):
        self.prompt_generator = prompt_generator
        all_context, all_question, all_option_list, all_label = read_func(file_path)

        is_seq2seq = is_seq2seq_tokenizer(tokenizer)
        logger.info("{} is seq2seq tokenizer: {}".format(tokenizer.__class__.__name__, is_seq2seq))

        self.inputs = []
        self.indices = []
        self.labels = []
        self.outputs = []
        for i in range(len(all_context)):
            prompt = f"Context:\n{all_context[i]}\n\nQuestion:\n{all_question[i]}\n\nOptions:\n{_format_option_list(all_option_list[i])}" \
                     f"\n\n" + suffix

            if add_option:
                targets = [f"{_rank2option[i]}: {all_option_list[i]}" for i in range(len(all_option_list[i]))]
            else:
                targets = [_rank2option[i] for i in range(len(all_option_list[i]))]
                targets = [tokenizer.tokenize(prompt + " " + x)[-1] for x in targets]

            self.inputs.append(prompt)
            self.outputs.append(targets)

            self.indices.append(i)
            self.labels.append(all_label[i])

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        prompt, prompt_indices = self.prompt_generator()
        return {
            "input": prompt + "\n\n" + self.inputs[index],
            "output": self.outputs[index],
            "index": self.indices[index],
            "prompt_index": ",".join(map(str, prompt_indices)),
            "label": self.labels[index],
        }


class ReClorCandidateSelectionDatasetV2(Dataset):
    def __init__(self, file_path: str, tokenizer: PreTrainedTokenizer, read_func, prompt_generator, suffix: str = "The answer is"):
        self.prompt_generator = prompt_generator
        all_context, all_question, all_option_list, all_label = read_func(file_path)

        is_seq2seq = is_seq2seq_tokenizer(tokenizer)
        logger.info("{} is seq2seq tokenizer: {}".format(tokenizer.__class__.__name__, is_seq2seq))

        self.inputs = []
        self.indices = []
        self.labels = []
        self.outputs = []
        for i in range(len(all_context)):
            prompt = f"Context:\n{all_context[i]}\n\nQuestion:\n{all_question[i]}\n\nOptions:\n{_format_option_list(all_option_list[i])}" \
                     f"\n\n" + suffix

            options = [(_rank2option[i], i) for i in range(len(all_option_list[i]))]

            targets = []
            targets += [(tokenizer.tokenize(prompt + " " + x[0])[-1], x[1]) for x in options]
            targets += [(tokenizer.tokenize(prompt + x[0])[-1], x[1]) for x in options]

            self.inputs.append(prompt)
            self.outputs.append(targets)

            self.indices.append(i)
            self.labels.append(all_label[i])

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        prompt, prompt_indices = self.prompt_generator()
        return {
            "input": prompt + "\n\n" + self.inputs[index],
            "output": self.outputs[index],
            "index": self.indices[index],
            "prompt_index": ",".join(map(str, prompt_indices)),
            "label": self.labels[index],
        }


class ReClorCandidatePPLDataset(Dataset):
    def __init__(self, file_path: str, tokenizer: PreTrainedTokenizer, read_func,
                 prefix1: str = "", prefix2: str = "\n\n", suffix: str = "\n\n"):
        all_context, all_question, all_option_list, all_label = read_func(file_path)

        self.inputs = []
        self.indices = []
        self.labels = []
        for i in range(len(all_context)):
            # prompt = f"{all_context[i]}\n\n{all_question[i]}\n\n"
            prompt = f"{prefix1}{all_context[i]}{prefix2}{all_question[i]}{suffix}"
            if all_question[i][-6:] == "  _  .":
                prompt = f"{prefix1}{all_context[i]}{prefix2}{all_question[:-5]}{suffix}"

            self.inputs.append([prompt + x for x in all_option_list[i]])
            self.indices.append(i)
            self.labels.append(all_label[i])

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return {
            "input": self.inputs[index],
            "index": self.indices[index],
            "prompt_index": "0",
            "label": self.labels[index],
        }


class ReClorGenerativeDatasetZh(Dataset):
    def __init__(self, file_path: str, tokenizer: PreTrainedTokenizer, read_func, prompt_generator, suffix: str = "The answer is"):
        self.prompt_generator = prompt_generator
        # read_func = ReClorReader()
        all_context, all_question, all_option_list, all_label = read_func(file_path)

        self.inputs = []
        self.indices = []
        self.labels = []
        for i in range(len(all_context)):
            prompt = f"{all_context[i]}\n\n问题：\n{all_question[i]}\n\n选项：\n{_format_option_list(all_option_list[i])}" \
                     f"\n\n" + suffix
            self.inputs.append(prompt)
            self.indices.append(i)
            self.labels.append(_rank2option[all_label[i]])

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        prompt, prompt_indices = self.prompt_generator()
        return {
            "input": prompt + "\n\n" + self.inputs[index],
            "index": self.indices[index],
            "prompt_index": ",".join(map(str, prompt_indices)),
            "label": self.labels[index],
        }


class ReClorSeq2SeqMCQADataset(Dataset):
    def __init__(self, file_path: str, tokenizer: PreTrainedTokenizer, read_func):
        super().__init__()
        all_context, all_question, all_option_list, all_label = read_func(file_path)
        self.contexts = all_context
        self.questions = all_question
        self.option_list = all_option_list
        self.labels = all_label

    def __len__(self):
        return len(self.contexts)

    def __getitem__(self, index):
        return {
            "index": index,
            "context": self.contexts[index],
            "question": self.questions[index],
            "options": self.option_list[index],
            "label": self.labels[index],
        }


class ReClorSeq2SeqMCQACollator:
    def __init__(self, tokenizer: str, max_seq_length: int, decoder_only: bool = False):
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast=False)
        expand_special_tokenizer(self.tokenizer)
        self.max_seq_length = max_seq_length
        self.decoder_only = decoder_only

    def __call__(self, batch):
        inputs_a = []
        inputs_b = []
        batch_size = len(batch)
        labels = []
        indices = []
        for b in batch:
            op_num = len(b["options"])
            inputs_a.extend([b["context"] + b["question"]] * op_num)
            if self.decoder_only:
                inputs_b.extend(list(map(lambda x: b["context"] + b["question"] + x, b["options"])))
            else:
                inputs_b.extend(b["options"])
            labels.append(b["label"])
            indices.append(b["index"])

        op_num = len(inputs_a) // batch_size

        model_inputs = self.tokenizer(inputs_a, text_target=inputs_b,
                                      padding="longest", truncation=True, max_length=self.max_seq_length, return_tensors="pt")
        if self.decoder_only:
            input_lens = model_inputs["input_ids"].ne(self.tokenizer.pad_token_id).sum(dim=1)
            model_inputs = self.tokenizer(inputs_b, padding="longest", truncation=True, max_length=self.max_seq_length,
                                          return_tensors="pt")
            model_inputs["input_lens"] = input_lens

        model_inputs["input_ids"] = model_inputs["input_ids"].reshape(batch_size, op_num, -1)
        model_inputs["attention_mask"] = model_inputs["attention_mask"].reshape(batch_size, op_num, -1)

        if not self.decoder_only:
            model_inputs["decoder_input_ids"] = model_inputs["labels"].reshape(batch_size, op_num, -1)

        model_inputs["labels"] = torch.tensor(labels, dtype=torch.long)
        model_inputs["meta_data"] = {"index": indices}
        return model_inputs


class ReClorGenerativeCollator:
    def __init__(self, tokenizer: str, max_seq_length: int, **kwargs):
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(tokenizer, **kwargs)
        expand_special_tokenizer(self.tokenizer)
        self.max_seq_length = max_seq_length

    def __call__(self, batch):
        batch = default_collate(batch)
        inputs = batch.pop("input")
        model_inputs = self.tokenizer(inputs, padding="longest", truncation=True, max_length=self.max_seq_length, return_tensors="pt")
        if "token_type_ids" in model_inputs:
            model_inputs.pop("token_type_ids")
        model_inputs["meta_data"] = batch
        model_inputs["meta_data"]["input"] = inputs
        return model_inputs


class ReClorTrainingGenerativeCollator:
    def __init__(self, tokenizer: str, max_seq_length: int, padding: str = "longest", **kwargs):
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(tokenizer, **kwargs)
        expand_special_tokenizer(self.tokenizer)
        self.max_seq_length = max_seq_length
        self.padding = padding

    def __call__(self, batch):
        inputs = [b["input"] for b in batch]
        outputs = [b["label"] for b in batch]

        model_inputs = vanilla_seq2seq_converter_text(inputs, outputs, self.tokenizer, self.max_seq_length, self.padding, decoder_only=True)

        model_inputs["meta_data"] = {
            "input": inputs,
            "index": [b["index"] for b in batch],
            "label": [b["label"] for b in batch],
        }
        return model_inputs


class ReClorChatDataset(Dataset):
    """
    For post-processing by chat.
    The input file should be the output of `post_processors.reclor.GeneratorPredictor`.
    """

    def __init__(self, file_path: str, tokenizer: PreTrainedTokenizer, instruction: str, suffix: str):
        self.data = json.load(open(file_path, "r"))
        self.instruction = instruction
        self.suffix = suffix

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """Should note that in chat format, instruction follows original output."""
        inputs = self.data[index]["output"] + self.instruction + self.suffix
        return {
            "input": inputs,
            "index": index,
            "prompt_index": "0",
            "label": self.data[index]["label"],
        }


class ReClorChatDatasetV2(Dataset):

    def __init__(self, file_path: str, tokenizer: PreTrainedTokenizer, instruction: str, suffix: str):
        self.data = json.load(open(file_path, "r"))
        self.instruction = instruction
        self.suffix = suffix

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """Should note that in chat format, instruction follows original output."""
        inputs = self.data[index]["text"] + "\n\n" + self.data[index]["response"] + self.instruction + self.suffix
        return {
            "input": inputs,
            "index": index,
            "prompt_index": "0",
            "label": self.data[index]["label"],
        }


class ReClorRewardPairDataset(Dataset):
    def __init__(self, file_path: str, tokenizer: PreTrainedTokenizer, read_func,
                 prefix1: str = "", prefix2: str = "\n\n", suffix: str = "\n\n", eval_mode: bool = False):
        super().__init__()
        all_context, all_question, all_option_list, all_label = read_func(file_path)
        self.contexts = all_context
        self.questions = all_question
        self.option_list = all_option_list
        self.labels = all_label
        self.prefix1 = prefix1
        self.prefix2 = prefix2
        self.suffix = suffix
        self.tokenizer = tokenizer
        self.eval_mode = eval_mode

    def __len__(self):
        if self.eval_mode:
            return len(self.contexts) * len(self.option_list[0])
        return len(self.contexts)

    def __getitem__(self, index):
        if self.eval_mode:
            example_index = index // len(self.option_list[0])
            context = self.contexts[example_index]
            question = self.questions[example_index]
            option = self.option_list[example_index][index % len(self.option_list[0])]
            inputs = self.prefix1 + context + self.prefix2 + question + self.suffix + option + self.tokenizer.eos_token

            label = 1 if index % len(self.option_list[0]) == self.labels[example_index] else 0

            return {
                "input": inputs,
                "index": f"{example_index}_{index % len(self.option_list[0])}",
                "label": label,
            }

        context = self.contexts[index]
        question = self.questions[index]

        pos = self.option_list[index][self.labels[index]]
        neg = random.choice(self.option_list[index][:self.labels[index]] + self.option_list[index][self.labels[index] + 1:])

        pos_inputs = self.prefix1 + context + self.prefix2 + question + self.suffix + pos + self.tokenizer.eos_token
        neg_inputs = self.prefix1 + context + self.prefix2 + question + self.suffix + neg + self.tokenizer.eos_token
        return {
            "pos_input": pos_inputs,
            "neg_input": neg_inputs,
            "index": index,
        }


class ReClorRewardPairCollator:
    def __init__(self, tokenizer: str, max_seq_length: int, add_input_lens: bool = False, **kwargs):
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(tokenizer, **kwargs)
        expand_special_tokenizer(self.tokenizer)
        self.max_seq_length = max_seq_length
        self.add_input_lens = add_input_lens

    def __call__(self, batch):
        if "pos_input" in batch[0]:
            # TODO: Add processing for `input_lens` here, also including the corresponding dataset class.
            pos_inputs = [b["pos_input"] for b in batch]
            neg_inputs = [b["neg_input"] for b in batch]

            pos_index = torch.arange(len(batch), dtype=torch.long)
            neg_index = torch.arange(len(batch), 2 * len(batch), dtype=torch.long)

            model_inputs = self.tokenizer(pos_inputs + neg_inputs, padding="longest", truncation=True,
                                          max_length=self.max_seq_length, return_tensors="pt")

            model_inputs["pos_index"] = pos_index
            model_inputs["neg_index"] = neg_index
            model_inputs["meta_data"] = {
                "pos_index": pos_index,
                "neg_index": neg_index,
                "index": [b["index"] for b in batch],
            }
        else:
            if not self.add_input_lens or "targets" not in batch[0]:
                if "input" not in batch[0]:
                    inputs = [b["inputs"] + " " + b["targets"] for b in batch]
                else:
                    inputs = [b["input"] for b in batch]
                model_inputs = self.tokenizer(inputs, padding="longest", truncation=True,
                                              max_length=self.max_seq_length, return_tensors="pt")
            else:
                inputs = [b["inputs"] for b in batch]
                outputs = [b["targets"] for b in batch]
                model_inputs = vanilla_seq2seq_converter_text(inputs, outputs, self.tokenizer, self.max_seq_length, padding="longest",
                                                              decoder_only=True)

            model_inputs["meta_data"] = {
                "index": [b["index"] for b in batch],
                "input": inputs,
                "label": [b["label"] for b in batch],
            }

        return model_inputs


class APICollator:
    def __call__(self, batch):
        prompts = [b["input"] for b in batch]

        meta_data = {
            "index": [b["index"] for b in batch],
            "prompt": prompts,
            "label": [b["label"] for b in batch],
        }

        return {"prompts": prompts, "meta_data": meta_data}


class ReClorCoTGenerationDataset(Dataset):
    def __init__(self, file_path: str, tokenizer: PreTrainedTokenizer, read_func, cot: List[Dict],
                 prefix1: str = "", prefix2: str = "\n\n", prefix3: str = "\n\n", suffix: str = "\n\n"):
        super().__init__()
        all_context, all_question, all_option_list, all_label = read_func(file_path)
        self.contexts = all_context
        self.questions = all_question
        self.option_list = all_option_list
        self.labels = all_label
        self.cot = cot
        # assert len(cot) == len(self.contexts)
        self.prefix1 = prefix1
        self.prefix2 = prefix2
        self.prefix3 = prefix3
        self.suffix = suffix
        self.tokenizer = tokenizer

    def __len__(self):
        if isinstance(self.cot[0]["index"], str) and "Step" in self.cot[0]["index"]:
            return len(self.cot)
        if len(self.cot) != len(self.contexts):
            return len(self.cot)
        return len(self.contexts)

    def __getitem__(self, index):
        cot_index = index
        if isinstance(self.cot[0]["index"], str) and "Step" in self.cot[0]["index"]:
            index = int(self.cot[cot_index]["index"].split("Step")[0])
        elif len(self.cot) != len(self.contexts):
            index = self.cot[cot_index]["index"]

        context = self.contexts[index]
        question = self.questions[index]
        options = self.option_list[index]

        inputs = self.prefix1 + context + self.prefix2 + question + self.prefix3 + _format_option_list(options) + self.suffix
        outputs = self.cot[cot_index]["output"]

        return {
            "inputs": inputs,
            "targets": outputs,
            "cot": self.cot[cot_index]["output"],
            "index": cot_index,
            "label": self.labels[index],
        }


def vanilla_seq2seq_convertor(examples, tokenizer: PreTrainedTokenizer, max_seq_length, remove_eos_in_step_cot: bool = False,
                              decoder_only: bool = False):
    inputs = []
    outputs = []
    for exp in examples:
        inputs.append(exp["inputs"])
        if decoder_only:
            if remove_eos_in_step_cot and "answer is" not in exp["targets"]:
                outputs.append(exp["inputs"] + " " + exp["targets"])
            else:
                outputs.append(exp["inputs"] + " " + exp["targets"] + tokenizer.eos_token)
        else:
            outputs.append(exp["targets"])

    model_inputs = tokenizer(inputs, text_target=outputs, max_length=max_seq_length, padding="longest",
                             truncation=True, return_tensors="pt")
    if decoder_only:
        input_lens = model_inputs["input_ids"].ne(tokenizer.pad_token_id).sum(dim=1)
        model_inputs = tokenizer(outputs, max_length=max_seq_length, padding="max_length", truncation=True, return_tensors="pt")
        new_input_lens = model_inputs["input_ids"].ne(tokenizer.pad_token_id).sum(dim=1)
        input_lens = input_lens - input_lens.eq(new_input_lens).to(input_lens.dtype) * (input_lens // 2)
        input_lens = input_lens.to(torch.long)
        if tokenizer.padding_side == "left":
            input_lens = model_inputs["input_ids"].eq(tokenizer.pad_token_id).to(torch.long).sum(dim=1) + input_lens
        model_inputs["input_lens"] = input_lens

    return model_inputs


class ConditionalCausalLMCollator:
    def __init__(self, tokenizer: str, max_seq_length: int, decoder_only: bool = False, return_standard_inputs: bool = False,
                 remove_eos_in_step_cot: bool = False,
                 **kwargs):
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(tokenizer, **kwargs)
        expand_special_tokenizer(self.tokenizer)
        self.max_seq_length = max_seq_length
        self.decoder_only = decoder_only
        self.convert_to_standard_inputs = return_standard_inputs
        self.remove_eos_in_step_cot = remove_eos_in_step_cot

    def __call__(self, batch):
        model_inputs = vanilla_seq2seq_convertor(batch, self.tokenizer, self.max_seq_length,
                                                 self.remove_eos_in_step_cot,
                                                 self.decoder_only)

        if self.convert_to_standard_inputs:
            input_ids, attention_mask, position_ids, labels = convert_to_standard_inputs(model_inputs, self.tokenizer)

            return (
                (input_ids, attention_mask, position_ids),
                labels,
            )

        model_inputs["meta_data"] = default_collate(batch)
        return model_inputs
