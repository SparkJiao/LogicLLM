import json
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizer
from transformers.tokenization_utils import PaddingStrategy, TruncationStrategy
from data.data_utils import tokenizer_get_name, get_sep_tokens, get_unused_tokens
from torch.utils.data import default_collate


def read_reclor_raw_data(file: str):
    data = json.load(open(file))

    all_context = []
    all_question = []
    all_answer = []
    all_id = []
    for sample in data:
        all_context.append(sample["context"])
        all_question.append(sample["question"])
        all_id.append(sample["id_string"])
        if "label" in sample:
            all_answer.append(sample["answers"][sample["label"]])

    return all_context, all_question, all_answer, all_id


def read_reclor_raw_data_w_options(file_path):
    data = json.load(open(file_path, 'r'))

    all_context = []
    all_question = []
    all_option_list = []
    all_label = []
    all_id = []
    for sample in data:
        all_context.append(sample["context"])
        all_question.append(sample["question"])
        if "label" not in sample:
            all_label.append(-1)
        else:
            all_label.append(sample["label"])
        all_option_list.append(sample["answers"])
        all_id.append(sample["id_string"])

    return all_context, all_question, all_option_list, all_label, all_id


class ReClorIndexDataset(Dataset):
    def __init__(self, file_path: str, tokenizer: PreTrainedTokenizer, corpus_file_path: str, remove_golden: bool = True):
        super().__init__()

        context, question, _, id_string = read_reclor_raw_data(file_path)
        cp_context, cp_question, cp_answer, cp_id = read_reclor_raw_data(corpus_file_path)

        self.query = {id_string: f"{ctx} {q}" for ctx, q, id_string in zip(context, question, id_string)}
        self.corpus = {id_string: f"{ctx} {q} {ans}"
                       for ctx, q, ans, id_string in zip(cp_context, cp_question, cp_answer, cp_id)}

        self.data_index = []
        for id_string1 in self.query.keys():
            for id_string2 in self.corpus.keys():
                if remove_golden and id_string1 == id_string2:
                    continue
                self.data_index.append((id_string1, id_string2))

    def __len__(self):
        return len(self.data_index)

    def __getitem__(self, index):
        que_id_string, ctx_id_string = self.data_index[index]

        item = {
            "query": self.query[que_id_string],
            "document": self.corpus[ctx_id_string],
            "meta_data": {
                "query": self.query[que_id_string],
                "document": self.corpus[ctx_id_string],
                "que_id": que_id_string,
                "ctx_id": ctx_id_string,
            }
        }
        return item


class ReClorIndexCollator:
    def __init__(self, tokenizer: str, max_seq_length: int):
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast=False)
        self.max_seq_length = max_seq_length

    def __call__(self, batch):
        query = [b.pop("query") for b in batch]
        document = [b.pop("document") for b in batch]
        meta_data = [b.pop("meta_data") for b in batch]

        model_inputs = self.tokenizer(query,
                                      document,
                                      padding=PaddingStrategy.LONGEST,
                                      truncation=True,
                                      max_length=self.max_seq_length,
                                      return_tensors="pt")

        model_inputs["input_ids"] = model_inputs["input_ids"].unsqueeze(1)
        model_inputs["attention_mask"] = model_inputs["attention_mask"].unsqueeze(1)
        if "token_type_ids" in model_inputs:
            model_inputs["token_type_ids"] = model_inputs["token_type_ids"].unsqueeze(1)

        model_inputs["meta_data"] = meta_data
        return model_inputs


class ReClorPairRepresentationDataset(Dataset):
    def __init__(self, file_path: str, tokenizer: PreTrainedTokenizer, max_seq_length: int, use_answer: bool = False):
        super().__init__()

        context, question, option_list, labels, id_list = read_reclor_raw_data_w_options(file_path)

        inputs_a = []
        inputs_b = []
        indices = []

        if use_answer:
            for c, q, op_ls, label, sample_id in zip(context, question, option_list, labels, id_list):
                inputs_a.append(c)
                inputs_b.append(q + ' '.join(get_sep_tokens(tokenizer)) + op_ls[label])
                indices.append(f"{sample_id}")
        else:
            for c, q, op_ls, sample_id in zip(context, question, option_list, id_list):
                for op_id, op in enumerate(op_ls):
                    inputs_a.append(c)
                    inputs_b.append(q + ' '.join(get_sep_tokens(tokenizer)) + op)
                    indices.append(f"{sample_id}_op{op_id}")

        model_inputs = tokenizer(inputs_a,
                                 text_pair=inputs_b,
                                 max_length=max_seq_length,
                                 padding=PaddingStrategy.LONGEST,
                                 truncation=TruncationStrategy.LONGEST_FIRST,
                                 return_tensors="pt")

        self.model_inputs = model_inputs
        self.indices = indices

    def __len__(self):
        assert len(self.model_inputs["input_ids"]) == len(self.indices)
        return len(self.indices)

    def __getitem__(self, index):
        res = {k: v[index] for k, v in self.model_inputs.items()}
        res["meta_data"] = {
            "index": self.indices[index]
        }
        return res
