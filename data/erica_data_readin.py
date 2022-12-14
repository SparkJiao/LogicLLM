import glob
import json
import os
import pickle
import random
from collections import defaultdict
from typing import Dict, Any, List

import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from collections import Counter
from transformers import AutoTokenizer, PreTrainedTokenizer
import transformers

from general_util.logger import get_child_logger

logger = get_child_logger(__name__)


def rearrange_example_id(examples):
    id_set = {}
    for exp in examples:
        orig_id = str(exp["id"])
        if orig_id not in id_set:
            id_set[orig_id] = 0

        exp["id"] = orig_id + "_" + str(id_set[orig_id])
        id_set[orig_id] = id_set[orig_id] + 1
    return examples


class SeperatorInterface:
    h_s_sep = "<e1>"
    h_e_sep = "</e1>"
    t_s_sep = "<e2>"
    t_e_sep = "</e2>"


class ERICATextDataset(Dataset):
    def __init__(self, file_path: str, max_seq_length: int, tokenizer: PreTrainedTokenizer, mlm: bool = False):
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
        self.mlm = mlm

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
            ent_tokens = self.tokenizer.tokenize(context[char_s: char_e])
            if self.mlm:
                tokens.extend([self.tokenizer.mask_token] * len(ent_tokens))
            else:
                tokens.extend(ent_tokens)
            ent_span_e = len(tokens)

            if ent_span_e < (self.max_seq_length - self.special_token_num):
                # entity_subword_spans.append((ent_span_s, ent_span_e))
                entity_subword_spans[mention["id"]].append((ent_span_s, ent_span_e))

            last_s = ent_span_e

        return self.tokenizer.convert_tokens_to_ids(tokens), entity_subword_spans, sorted_mentions


class ERICASentenceDataset(Dataset):
    def __init__(self, file_path: str, tokenizer: PreTrainedTokenizer, cached_path: str = None, keep_original: bool = False):
        if os.path.exists(file_path):
            input_files = [file_path]
        else:
            input_files = list(glob.glob(file_path))
            input_files = sorted(input_files)
        assert input_files, input_files

        self.tokenizer = tokenizer

        self.h_s_sep = "<e1>"
        self.h_e_sep = "</e1>"
        self.t_s_sep = "<e2>"
        self.t_e_sep = "</e2>"
        self.tokenizer.add_tokens([self.h_s_sep, self.h_e_sep, self.t_s_sep, self.t_e_sep])
        self.keep_original = keep_original

        if cached_path is None or not os.path.exists(cached_path):
            samples = []
            for _file in input_files:
                logger.info(f"Reading from {_file}")
                samples.extend(json.load(open(_file)))

            self.examples = []
            for sample in tqdm(samples):
                self.examples.extend(self.process_single_para(sample))

            del samples

            if cached_path is not None:
                logger.info(f"Saving data to {cached_path}")
                torch.save(self.examples, cached_path)
        else:
            logger.info(f"Loading data from {cached_path}")
            self.examples = torch.load(cached_path)
            if isinstance(self.examples[0], tuple) and isinstance(self.examples[0][0], list):
                for exp_id, exp in enumerate(tqdm(self.examples)):
                    self.examples[exp_id] = (" ".join(exp[0]), exp[1])

    def __getitem__(self, index):
        # print(self.examples[index])
        return self.examples[index]

    def __len__(self):
        return len(self.examples)

    def annotate_entity(self, sent, mention1, mention2):
        if mention1["pos"][0] > mention2["pos"][0]:
            mention2, mention1 = mention1, mention2
        assert mention1["pos"][1] <= mention2["pos"][0]

        text = ""
        prefix = sent[:mention1["pos"][0]]
        if len(prefix):
            text = text + " ".join(prefix)

        text = text + " " + " ".join([self.h_s_sep] + sent[mention1["pos"][0]: mention1["pos"][1]] + [self.h_e_sep])

        inter = sent[mention1["pos"][1]: mention2["pos"][0]]
        if len(inter):
            text = text + " " + " ".join(inter)

        text = text + " " + " ".join([self.t_s_sep] + sent[mention2["pos"][0]: mention2["pos"][1]] + [self.t_e_sep])

        text = text + " " + " ".join(sent[mention2["pos"][1]:])
        return text

    def process_single_para(self, sample):
        entities = sample["vertexSet"]
        sentences = sample["sents"]

        results = []
        for idx, ent1 in enumerate(entities):
            for ent2 in entities[(idx + 1):]:
                if ent1[0]["id"] == ent2[0]["id"]:
                    continue
                for mention1 in ent1:
                    for mention2 in ent2:
                        if mention1["sent_id"] == mention2["sent_id"]:
                            aug = self.annotate_entity(sentences[mention1["sent_id"]], mention1, mention2)
                            if self.keep_original:
                                results.append((sentences[mention1["sent_id"]], aug))
                            else:
                                results.append(aug)
        return results


class ERICASentenceFilterDataset(Dataset):
    def __init__(self, file_path: str, tokenizer: PreTrainedTokenizer, cached_path: str = None):
        if os.path.exists(file_path):
            input_files = [file_path]
        else:
            input_files = list(glob.glob(file_path))
            input_files = sorted(input_files)
        assert input_files, input_files

        self.tokenizer = tokenizer

        self.h_s_sep = "<e1>"
        self.h_e_sep = "</e1>"
        self.t_s_sep = "<e2>"
        self.t_e_sep = "</e2>"
        self.tokenizer.add_tokens([self.h_s_sep, self.h_e_sep, self.t_s_sep, self.t_e_sep])

        if cached_path is None or not os.path.exists(cached_path):
            samples = []
            for _file in input_files:
                logger.info(f"Reading from {_file}")
                samples.extend(json.load(open(_file)))

            self.examples = []
            for sample in tqdm(samples):
                self.examples.extend(self.process_single_para(sample))

            del samples

            if cached_path is not None:
                logger.info(f"Saving data to {cached_path}")
                torch.save(self.examples, cached_path)
        else:
            logger.info(f"Loading data from {cached_path}")
            self.examples = torch.load(cached_path)

    def __getitem__(self, index):
        return self.examples[index]

    def __len__(self):
        return len(self.examples)

    def annotate_entity(self, sent, mention1, mention2):
        if mention1["pos"][0] > mention2["pos"][0]:
            mention2, mention1 = mention1, mention2
        assert mention1["pos"][1] <= mention2["pos"][0]

        text = ""
        prefix = sent[:mention1["pos"][0]]
        if len(prefix):
            text = text + " ".join(prefix)

        text = text + " " + " ".join([self.h_s_sep] + sent[mention1["pos"][0]: mention1["pos"][1]] + [self.h_e_sep])

        inter = sent[mention1["pos"][1]: mention2["pos"][0]]
        if len(inter):
            text = text + " " + " ".join(inter)

        text = text + " " + " ".join([self.t_s_sep] + sent[mention2["pos"][0]: mention2["pos"][1]] + [self.t_e_sep])

        text = text + " " + " ".join(sent[mention2["pos"][1]:])
        return text

    def process_single_para(self, sample):
        entities = sample["vertexSet"]
        sentences = sample["sents"]

        sent_ent_num = Counter()
        for ent in entities:
            for mention in ent:
                sent_ent_num[mention["sent_id"]] += 1

        results = []
        for idx, ent1 in enumerate(entities):
            for ent2 in entities[(idx + 1):]:
                if ent1[0]["id"] == ent2[0]["id"]:
                    continue
                for mention1 in ent1:
                    for mention2 in ent2:
                        if mention1["sent_id"] == mention2["sent_id"]:
                            if sent_ent_num[mention1["sent_id"]] <= 2:
                                continue
                            aug = self.annotate_entity(sentences[mention1["sent_id"]], mention1, mention2)
                            results.append((" ".join(sentences[mention1["sent_id"]]), aug))

        return results


def annotate_entity(cls, sent, mention1, mention2):
    if mention1["pos"][0] > mention2["pos"][0]:
        mention2, mention1 = mention1, mention2
    assert mention1["pos"][1] <= mention2["pos"][0]

    text = ""
    prefix = sent[:mention1["pos"][0]]
    if len(prefix):
        text = text + " ".join(prefix)

    text = text + " " + " ".join([cls.h_s_sep] + sent[mention1["pos"][0]: mention1["pos"][1]] + [cls.h_e_sep])

    inter = sent[mention1["pos"][1]: mention2["pos"][0]]
    if len(inter):
        text = text + " " + " ".join(inter)

    text = text + " " + " ".join([cls.t_s_sep] + sent[mention2["pos"][0]: mention2["pos"][1]] + [cls.t_e_sep])

    text = text + " " + " ".join(sent[mention2["pos"][1]:])
    return text


class WikiPathSentenceDataset(Dataset, SeperatorInterface):
    def __init__(self, file_path: str, tokenizer: PreTrainedTokenizer):
        data = pickle.load(open(file_path, "rb"))["examples"]
        self.sentences = []
        self.indices = []

        for item in tqdm(data, total=len(data)):
            selected_sentences = item["selected_sentences"]
            pos = item["pos"]

            for exp_id, exp in selected_sentences.items():
                self.sentences.append(annotate_entity(self, exp["sent"], exp["ent"][exp["h"]][0], exp["ent"][exp["t"]][0]))
                self.indices.append(f"{item['id']}-path-{exp_id}")

            for exp_id, exp in enumerate(pos):
                self.sentences.append(annotate_entity(self, exp["sent"], exp["ent"][exp["h"]][0], exp["ent"][exp["t"]][0]))
                self.indices.append(f"{item['id']}-pos-{exp_id}")

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        return self.sentences[index], self.indices[index]


class WikiPathSentenceConditionDataset(Dataset):
    def __init__(self, file_path: str, tokenizer: PreTrainedTokenizer, max_seq_length: int, cache_path: str = None):
        if cache_path is not None and os.path.exists(cache_path):
            self.sentences, self.indices = torch.load(cache_path)
        else:
            data = pickle.load(open(file_path, "rb"))["examples"]

            sentences: List[Dict[str, Any]] = []
            indices = []
            for item in tqdm(data, total=len(data)):
                for s_id, s in item["selected_sentences"].items():
                    sent = " ".join(s["sent"])
                    if len(tokenizer.tokenize(sent)) + 2 <= max_seq_length:
                        sentences.append(s)
                        indices.append(f"{item['id']}-path-{s_id}")

                for pos_id, pos in enumerate(item["pos"]):
                    sent = " ".join(pos["sent"])
                    if len(tokenizer.tokenize(sent)) + 2 <= max_seq_length:
                        sentences.append(pos)
                        indices.append(f"{item['id']}-pos-{pos_id}")

            if cache_path is not None:
                torch.save((sentences, indices), cache_path)

            self.sentences = sentences
            self.indices = indices

        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        sent = self.sentences[index]
        h = random.choice(sent["ent"][sent["h"]])
        t = random.choice(sent["ent"][sent["t"]])
        flag = True
        if h["pos"][0] > t["pos"][0]:
            h, t = t, h
            flag = False
        assert h["pos"][1] <= t["pos"][0]

        tokens = [self.tokenizer.cls_token]
        h_span = []
        t_span = []

        tokens.extend(self.tokenizer.tokenize(" ".join(sent["sent"][:h["pos"][0]])))
        h_span.append(len(tokens))

        h_mention = " ".join(sent["sent"][h["pos"][0]: h["pos"][1]])
        tokens.extend(self.tokenizer.tokenize(" " + h_mention))
        h_span.append(len(tokens) - 1)

        tokens.extend(self.tokenizer.tokenize(" " + " ".join(sent["sent"][h["pos"][1]: t["pos"][0]])))
        t_span.append(len(tokens))

        t_mention = " ".join(sent["sent"][t["pos"][0]: t["pos"][1]])
        tokens.extend(self.tokenizer.tokenize(" " + t_mention))
        t_span.append(len(tokens) - 1)

        tokens.extend(self.tokenizer.tokenize(" " + " ".join(sent["sent"][t["pos"][1]:])))
        tokens.append(self.tokenizer.sep_token)

        if not flag:
            h_span, t_span = t_span, h_span
            h_mention, t_mention = t_mention, h_mention

        text = " ".join(sent["sent"])

        return {
            "tokens": tokens,
            "text": text,
            "h_span": h_span,
            "t_span": t_span,
            "h": h_mention,
            "t": t_mention,
            "index": self.indices[index]
        }


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


class ERICASentenceCollator:
    def __init__(self, tokenizer, max_seq_length: int, partial_optim: bool = False, no_original: bool = False):
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.max_seq_length = max_seq_length

        self.h_s_sep = "<e1>"
        self.h_e_sep = "</e1>"
        self.t_s_sep = "<e2>"
        self.t_e_sep = "</e2>"
        self.tokenizer.add_tokens([self.h_s_sep, self.h_e_sep, self.t_s_sep, self.t_e_sep])
        self.partial_optim = partial_optim
        self.no_original = no_original

    def __call__(self, batch):
        if isinstance(batch[0], tuple) and isinstance(batch[0][1], str):
            if not self.no_original:
                orig_batch = [b[0] for b in batch]
            else:
                orig_batch = None
            batch = [b[1] for b in batch]
        else:
            orig_batch = None

        if transformers.__version__[:4] == "4.24":
            model_inputs = self.tokenizer(batch, text_target=batch,
                                          padding="longest", truncation=True, max_length=self.max_seq_length, return_tensors="pt")
        else:
            model_inputs = self.tokenizer(batch, padding="longest", truncation=True, max_length=self.max_seq_length, return_tensors="pt")
            with self.tokenizer.as_target_tokenizer():
                labels = self.tokenizer(batch, padding="longest", truncation=True, max_length=self.max_seq_length, return_tensors="pt")
                model_inputs["labels"] = labels["input_ids"]

        labels = model_inputs["labels"]  # [batch, seq_length]
        special_tokens = torch.tensor(self.tokenizer.convert_tokens_to_ids([
            self.h_s_sep, self.h_e_sep, self.t_s_sep, self.t_e_sep]), dtype=torch.long)[None, None, :]  # [1, 1, 4]
        tgt_mask = labels.unsqueeze(-1) == special_tokens
        assert tgt_mask.size() == (labels.size(0), labels.size(1), 4)
        ent_mark_mask = tgt_mask.sum(dim=-1).bool()
        model_inputs["ent_mark_mask"] = ent_mark_mask

        if self.partial_optim:
            tgt_mask = tgt_mask.sum(dim=-1).cumsum(dim=1)  # [0, 0, (1, 1, 2), 2, (3, 3, 4), 4, 4, 4]
            partial_mask = (tgt_mask == 1) | (tgt_mask == 3)
            for b_id, b_tgt_mask in enumerate(tgt_mask):
                b_tgt_mask = b_tgt_mask.tolist()
                if 2 not in b_tgt_mask or 1 not in b_tgt_mask:
                    partial_mask[b_id, :] = 1
                    continue
                else:
                    idx2 = b_tgt_mask.index(2)
                    partial_mask[b_id, idx2] = 1

                if 3 not in b_tgt_mask or 4 not in b_tgt_mask:
                    partial_mask[b_id, :] = 0
                else:
                    idx4 = b_tgt_mask.index(4)
                    partial_mask[b_id, idx4] = 1
            partial_mask = ~partial_mask
            model_inputs["partial_mask"] = partial_mask

        if orig_batch is not None:
            orig_outputs = self.tokenizer(orig_batch, padding="longest", truncation=True, max_length=self.max_seq_length,
                                          return_tensors="pt")
            model_inputs["extra_input_ids"] = orig_outputs["input_ids"]
            model_inputs["extra_attention_mask"] = orig_outputs["attention_mask"]

        return model_inputs


class WikiPathInferenceCollator(SeperatorInterface):
    def __init__(self, tokenizer, max_seq_length: int):
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.max_seq_length = max_seq_length

        self.tokenizer.add_tokens([self.h_s_sep, self.h_e_sep, self.t_s_sep, self.t_e_sep])

    def __call__(self, batch):
        sentences, indices = list(zip(*batch))

        model_inputs = self.tokenizer(batch, padding="longest", truncation=True, max_length=self.max_seq_length, return_tensors="pt")
        model_inputs["meta_data"] = {"indices": indices}

        return model_inputs


class WikiPathSentenceConditionCollator:
    def __init__(self, enc_tokenizer: str, dec_tokenizer: str, max_seq_length: int):
        self.enc_tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(enc_tokenizer)
        self.dec_tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(dec_tokenizer)
        self.max_seq_length = max_seq_length

    def __call__(self, batch):
        seq_len = max(map(lambda x: len(x["tokens"]), batch))

        input_ids = torch.zeros(len(batch), seq_len, dtype=torch.int)
        attention_mask = torch.zeros(len(batch), seq_len, dtype=torch.int)
        h_span = []
        t_span = []
        indices = []
        decoder_inputs = []
        decoder_outputs = []
        for b_id, b in enumerate(batch):
            input_ids[b_id, :len(b["tokens"])] = torch.tensor(self.enc_tokenizer.convert_tokens_to_ids(b["tokens"]), dtype=torch.int)
            attention_mask[b_id, :len(b["tokens"])] = 1

            h_span.append(b["h_span"])
            t_span.append(b["t_span"])
            indices.append(b["index"])

            decoder_inputs.append(b["h"] + self.dec_tokenizer.sep_token + b["t"])
            decoder_outputs.append(b["text"])

        h_span = torch.tensor(h_span, dtype=torch.long)
        t_span = torch.tensor(t_span, dtype=torch.long)

        decoder_inputs = self.dec_tokenizer(decoder_inputs, text_target=decoder_outputs,
                                            padding="longest", truncation=True, max_length=self.max_seq_length, return_tensors="pt")

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "decoder_input_ids": decoder_inputs["input_ids"],
            "decoder_input_attention_mask": decoder_inputs["attention_mask"],
            "decoder_output_ids": decoder_inputs["labels"],
            "h_span": h_span,
            "t_span": t_span,
            "meta_data": {
                "indices": indices,
            }
        }
