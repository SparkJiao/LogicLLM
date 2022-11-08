import glob
import json
import os

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizer
from collections import defaultdict

from general_util.logger import get_child_logger

logger = get_child_logger(__name__)


class ERICATextDataset(Dataset):
    def __init__(self, file_path: str, max_seq_length: int, tokenizer: PreTrainedTokenizer):
        if os.path.exists(file_path):
            input_files = [file_path]
        else:
            input_files = list(glob.glob(file_path))
            input_files = sorted(input_files)
        assert input_files, input_files
        self.samples = []
        for _file in input_files:
            logger.info(f"Reading from {_file}")
            self.samples.extend(json.load(open(_file)))
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer
        self.special_token_num = self.tokenizer.num_special_tokens_to_add()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.process_single_item(self.samples[index]) + (index,)

    def process_single_item(self, sample):
        entities = sample["vertexSet"]
        # relations = sample["labels"]
        sentences = sample["sents"]

        sent_len_offset = []
        words = []
        context = ""
        word_to_char_span = []
        for sent in sentences:
            sent_len_offset.append(len(words))
            words.extend(sent)

            for word in sent:
                if len(context) > 0:
                    context = context + " "
                s = len(context)
                context = context + word
                e = len(context)
                word_to_char_span.append((s, e))

            assert len(word_to_char_span) == len(words)

        all_mentions = []
        for ent in entities:
            for mention in ent:
                offset = sent_len_offset[mention["sent_id"]]
                mention["pos"] = [mention["pos"][0] + offset, mention["pos"][1] + offset]
                all_mentions.append(mention)

        sorted_mentions = sorted(all_mentions, key=lambda x: x["pos"][0])
        for i in range(len(sorted_mentions)):
            if i == 0:
                continue
            if sorted_mentions[i]["pos"][0] < sorted_mentions[i - 1]["pos"][1]:
                logger.warning(f"Bad instance checked: {sorted_mentions[i - 1]}\t{sorted_mentions[i]}")

        # Obtain the entity positions in subword sequence.
        tokens = []
        last_s = 0
        entity_subword_spans = defaultdict(list)
        for mention in sorted_mentions:
            char_s, _ = word_to_char_span[mention["pos"][0]]
            _, char_e = word_to_char_span[mention["pos"][1] - 1]
            if char_s > last_s:
                tokens.extend(self.tokenizer.tokenize(context[last_s: char_s]))
            ent_span_s = len(tokens)
            tokens.extend(self.tokenizer.tokenize(context[char_s: char_e]))
            ent_span_e = len(tokens)

            if ent_span_e < (self.max_seq_length - self.special_token_num):
                # entity_subword_spans.append((ent_span_s, ent_span_e))
                entity_subword_spans[mention["id"]].append((ent_span_s, ent_span_e))

            last_s = ent_span_e

        return self.tokenizer.convert_tokens_to_ids(tokens), entity_subword_spans, sorted_mentions


class ERICATextCollator:
    def __init__(self, tokenizer, max_seq_length: int):
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast=False)
        self.max_seq_length = max_seq_length

    def __call__(self, batch):
        batch_ids, batch_entity_spans, batch_ent_mentions, indices = list(zip(*batch))

        batch_input_ids = []
        batch_attention_mask = []
        for ids in batch_ids:
            _inputs = self.tokenizer.prepare_for_model(ids, padding="longest", truncation=True, max_length=self.max_seq_length)
            batch_input_ids.append(_inputs["input_ids"])
            batch_attention_mask.append(_inputs["attention_mask"])

        batch_max_seq_len = max(map(len, batch_input_ids))
        input_ids = torch.zeros(len(batch_ids), batch_max_seq_len, dtype=torch.long).fill_(self.tokenizer.pad_token_id)
        attention_mask = torch.zeros(len(batch_ids), batch_max_seq_len, dtype=torch.long)
        for i, (ids, mask) in enumerate(zip(batch_input_ids, batch_attention_mask)):
            input_ids[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
            attention_mask[i, :len(ids)] = torch.tensor(mask, dtype=torch.long)

        max_ent_num = max(map(len, batch_entity_spans))

        # ent_span_index = torch.zeros(len(batch_ids), max_ent_num, 2, dtype=torch.long)
        # ent_mask = torch.zeros(len(batch_ids), max_ent_num, dtype=torch.int)
        # for i, spans in enumerate(batch_entity_spans):
        #     ent_span_index[i, len(spans)] = torch.tensor(spans, dtype=torch.long)
        #     ent_mask[i, len(spans)] = 1

        ent_seq_mapping = torch.zeros(len(batch_ids), max_ent_num, batch_max_seq_len)
        for i, b_ent_spans in enumerate(batch_entity_spans):
            for j, ent_mentions in enumerate(b_ent_spans.values()):
                for ent_mention in ent_mentions:
                    ent_seq_mapping[i, j, ent_mention[0]: ent_mention[1]] = 1.0 / len(ent_mentions) / (ent_mention[1] - ent_mention[0])

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "ent_seq_mapping": ent_seq_mapping,
            "meta_data": {
                "index": indices,
                "entity_spans": batch_entity_spans,
                "entity_mentions": batch_ent_mentions,
            }
        }
