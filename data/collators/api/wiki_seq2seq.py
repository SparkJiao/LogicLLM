import json
import random

import torch
from torch.utils.data import Dataset

from data.collators.api.wiki_utils import extract_ending_entity, extract_ending_entity_and_anonymization
from data.collators.wiki_seq2seq_collator import construct_seq2seq
from general_util.logger import get_child_logger

logger = get_child_logger(__name__)


class WikiDatasetUnifyInterface(Dataset):
    def __init__(self, file_path, collator=None, tokenizer=None, sample_num: int = 1000, balance: bool = False):
        logger.info(f"Loading examples from {file_path}")

        examples, raw_texts = torch.load(file_path)
        self.examples = examples
        self.example_indices = list(range(len(self.examples)))
        self.sample_num = sample_num

        if balance:
            sub_num = sample_num // 2
            normal_indices = []
            counterfactual_indices = []
            for exp_id, exp in enumerate(self.examples):
                if "h" in exp:
                    assert "t" in exp
                    counterfactual_indices.append(exp_id)
                else:
                    normal_indices.append(exp_id)

            logger.info(f"Normal data size: {len(normal_indices)}")
            logger.info(f"Counterfactual data size: {len(counterfactual_indices)}")

            self.example_indices = random.sample(normal_indices, sub_num) + random.sample(counterfactual_indices, sub_num)
        else:
            if sample_num > 0:
                self.example_indices = random.sample(self.example_indices, sample_num)

        self.collator = collator

    def __len__(self):
        return len(self.example_indices)

    def __getitem__(self, index):
        exp_id = self.example_indices[index]
        example = self.examples[exp_id]
        example = {
            "example": example,
            "index": exp_id,
        }
        if self.collator is None:
            return example

        batch = self.collator([example])
        return batch


class WikiDatasetRelExtractionInterface(Dataset):
    templates = [
        "Extract the relation between {} and {} from the following text:\n\n"
        "{}",
    ]

    def __init__(self, file_path, tokenizer=None, template_id: int = 0, sample_num: int = 1000):
        self.template_id = template_id

        logger.info(f"Loading examples from {file_path}")

        examples, raw_texts = torch.load(file_path)

        sub_num = sample_num // 2
        normal_indices = []
        counterfactual_indices = []
        for exp_id, exp in enumerate(examples):
            if "h" in exp:
                assert "t" in exp
                counterfactual_indices.append(exp_id)
            else:
                normal_indices.append(exp_id)

        logger.info(f"Normal data size: {len(normal_indices)}")
        logger.info(f"Counterfactual data size: {len(counterfactual_indices)}")

        self.example_indices = random.sample(normal_indices, sub_num) + random.sample(counterfactual_indices, sub_num)

        rel_examples = []
        for exp_id in self.example_indices:
            exp = examples[exp_id]
            flag, ent_a_span, ent_b_span = extract_ending_entity(exp)
            res = construct_seq2seq(exp, generative_mode=True)

            rel_exp_0 = self.templates[template_id].format(ent_a_span, ent_b_span, res[0][0])
            rel_exp_1 = self.templates[template_id].format(ent_a_span, ent_b_span, res[1][0])

            rel_examples.append({
                "text": rel_exp_0,
                "meta_data": {
                    "text": rel_exp_0,
                    "index": f"{exp_id}_0",
                }
            })
            rel_examples.append({
                "text": rel_exp_1,
                "meta_data": {
                    "text": rel_exp_1,
                    "index": f"{exp_id}_1",
                }
            })

        self.examples = rel_examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


class WikiRelationConsistent:
    templates = [
        "Determine whether the relation between \"{}\" and \"{}\" in the given two sentences are logically consistent.\n"
        "\n"
        "Sentence 1:\n"
        "{}\n"
        "\n"
        "Sentence 2:\n"
        "{}\n"
        "\n"
        "The output should either be Yes or No.\n"
        "Output:",

        "Determine whether the relation between \"{}\" and \"{}\" in the given two sentences are logically consistent. "
        "Ignore the content of the two entities themselves.\n"
        "\n"
        "Sentence 1:\n"
        "{}\n"
        "\n"
        "Sentence 2:\n"
        "{}\n"
        "\n"
        "The output should either be Yes or No.\n"
        "Output:",

        "Determine whether the relations between \"{}\" and \"{}\" described by the two given paragraphs respectively are logically consistent.\n"
        "\n"
        "Paragraph 1:\n"
        "{}\n"
        "\n"
        "Paragraph 2:\n"
        "{}",
    ]

    def __init__(self, template_id: int, anonymization: bool = False):
        self.template = self.templates[template_id]
        self.anonymization = anonymization

    @staticmethod
    def anonymize_call(self, batch):
        examples = [b["example"] for b in batch]
        inputs_a, inputs_b = [], []
        entity_a = []
        entity_b = []
        for exp in examples:
            exp, ent_a_span, ent_b_span = extract_ending_entity_and_anonymization(exp)
            res = construct_seq2seq(exp, generative_mode=True)
            assert len(res[0]) == len(res[1])

            inputs_a.extend(res[0])
            inputs_b.extend(res[1])

            entity_a.append(ent_a_span)
            entity_b.append(ent_b_span)

        assert len(inputs_a) == len(inputs_b) == len(entity_a) == len(entity_b)

        model_inputs = {
            "text": [self.template.format(a, b, c, d) for a, b, c, d in zip(entity_a, entity_b, inputs_a, inputs_b)],
        }

        model_inputs["meta_data"] = {
            "text": model_inputs["text"],
            "index": [b["index"] for b in batch],
        }

        return model_inputs

    def __call__(self, batch):
        if self.anonymization:
            return self.anonymize_call(self, batch)

        examples = [b["example"] for b in batch]
        inputs_a, inputs_b = [], []
        entity_a = []
        entity_b = []
        for exp in examples:
            res = construct_seq2seq(exp, generative_mode=True)
            assert len(res[0]) == len(res[1])

            flag, ent_a_span, ent_b_span = extract_ending_entity(exp)
            if flag:
                inputs_a.extend(res[0])
                inputs_b.extend(res[1])

                entity_a.append(ent_a_span)
                entity_b.append(ent_b_span)
            else:
                logger.warn("No entity span found")
                # print(json.dumps(exp, indent=2))
                inputs_a.extend([""] * len(res[0]))
                inputs_b.extend([""] * len(res[1]))

                entity_a.append("")
                entity_b.append("")

        assert len(inputs_a) == len(inputs_b) == len(entity_a) == len(entity_b)

        model_inputs = {
            "text": [self.template.format(a, b, c, d) for a, b, c, d in zip(entity_a, entity_b, inputs_a, inputs_b)],
        }

        model_inputs["meta_data"] = {
            "text": model_inputs["text"],
            "index": [b["index"] for b in batch],
        }

        return model_inputs


class RelDiscChatDataset(Dataset):
    instructions = [
        "Summarize the above thinking process and give the final answer from either Yes or No.\nAnswer:",
        "The final answer from either Yes or No is",
    ]

    def __init__(self, file_path, instruction_id: int = 0, tokenizer=None):
        logger.info(f"Loading examples from {file_path}")

        self.predictions = json.load(open(file_path, "r"))
        self.instruction = self.instructions[instruction_id]

    def __len__(self):
        return len(self.predictions)

    def __getitem__(self, index):
        pred = self.predictions[index]
        text = [
            {
                "role": "user",
                "content": pred["text"][0],
            },
            {
                "role": "assistant",
                "content": pred["response"],
            },
            {
                "role": "user",
                "content": self.instruction,
            }
        ]
        return {
            "text": text,
            "meta_data": {
                "text": text,
                "index": pred["id"],
            }
        }
