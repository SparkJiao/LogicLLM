import json
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizer
from transformers.tokenization_utils import PaddingStrategy, TruncationStrategy


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


class ReClorIndexDataset(Dataset):
    def __init__(self, file_path: str, corpus_file_path: str, remove_golden: bool = True):
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
                                      padding=PaddingStrategy.MAX_LENGTH,
                                      truncation=True,
                                      max_length=self.max_seq_length)

        model_inputs["input_ids"] = model_inputs["input_ids"].unsqueeze(1)
        model_inputs["attention_mask"] = model_inputs["attention_mask"].unsqueeze(1)
        if "token_type_ids" in model_inputs:
            model_inputs["token_type_ids"] = model_inputs["token_type_ids"].unsqueeze(1)

        model_inputs["meta_data"] = meta_data
        return model_inputs


