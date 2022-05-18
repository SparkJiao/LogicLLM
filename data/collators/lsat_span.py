import copy
import random
from typing import List, Dict, Union

import torch
from torch import Tensor
from torch.utils.data import Dataset


class LSATSpanDataset(Dataset):
    def __init__(self,
                 input_ids: Tensor,
                 attention_mask: Tensor,
                 token_type_ids: Union[Tensor, None],
                 option_mask: Tensor,
                 labels: Tensor,
                 part_index_list: List[List[Dict]],
                 pos_index_list: List[List[Dict]]):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.option_mask = option_mask
        self.labels = labels
        self.part_index_list = part_index_list
        self.pos_index_list = pos_index_list

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        if self.token_type_ids is not None:
            return (self.input_ids[idx],
                    self.attention_mask[idx],
                    self.token_type_ids[idx],
                    self.option_mask[idx],
                    self.labels[idx],
                    self.part_index_list[idx],
                    self.pos_index_list[idx])
        else:
            return (self.input_ids[idx],
                    self.attention_mask[idx],
                    self.option_mask[idx],
                    self.labels[idx],
                    self.part_index_list[idx],
                    self.pos_index_list[idx])


class LSATSpanCollator:
    def __init__(self, sample_num: int = 0):
        self.sample_num = sample_num
        if self.sample_num <= 0:
            raise NotImplementedError("sample_num must be greater than 0")

    def __call__(self, batch):
        if len(batch[0]) == 7:
            input_ids, attention_mask, token_type_ids, option_mask, labels, part_index_list, pos_index_list = list(zip(*batch))
        elif len(batch[0]) == 6:
            input_ids, attention_mask, option_mask, labels, part_index_list, pos_index_list = list(zip(*batch))
            token_type_ids = None
        else:
            raise ValueError("Invalid batch size")

        input_ids = torch.stack(input_ids, dim=0)
        attention_mask = torch.stack(attention_mask, dim=0)
        option_mask = torch.stack(option_mask, dim=0)
        labels = torch.stack(labels, dim=0)

        option_num = input_ids.size(1)

        part_index, part_token_mask, part_occur_mask, part_mask, part_decoder_input_ids = self.process_span_list(part_index_list,
                                                                                                                 option_num)
        pos_index, pos_token_mask, pos_occur_mask, pos_mask, _ = self.process_span_list(pos_index_list, option_num)

        res = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "op_mask": option_mask,
            "labels": labels,
            "part_index": part_index,
            "part_token_mask": part_token_mask,
            "part_occur_mask": part_occur_mask,
            "part_mask": part_mask,
            "pos_index": pos_index,
            "pos_token_mask": pos_token_mask,
            "pos_occur_mask": pos_occur_mask,
            "pos_mask": pos_mask,
            "part_decoder_input_ids": part_decoder_input_ids
        }
        if token_type_ids is not None:
            res["token_type_ids"] = torch.stack(token_type_ids, dim=0)
        return res

    def process_span_list(self, span_index_list: List, option_num: int):
        """
        ::param::
            span_index_list: batch_size * option_num * span_num * span_occur_num * 2
        """
        max_span_num = 0
        max_span_occur_num = 0
        max_span_len = 0
        for b in range(len(span_index_list)):
            for o in range(len(span_index_list[b])):
                span_num = len(span_index_list[b][o])
                max_span_num = max(max_span_num, span_num)
                for s in range(len(span_index_list[b][o])):
                    occur_num = len(span_index_list[b][o][s])
                    max_span_occur_num = max(max_span_occur_num, occur_num)
                    for oc in range(len(span_index_list[b][o][s])):
                        span_len = span_index_list[b][o][s][oc][1] - span_index_list[b][o][s][oc][0]
                        max_span_len = max(max_span_len, span_len)

        span_index = torch.zeros(len(span_index_list), option_num, max_span_num, max_span_occur_num, max_span_len, dtype=torch.long)
        span_token_mask = torch.zeros(len(span_index_list), option_num, max_span_num, max_span_occur_num, max_span_len, dtype=torch.long)
        span_occur_mask = torch.zeros(len(span_index_list), option_num, max_span_num, max_span_occur_num, dtype=torch.long)
        span_mask = torch.zeros(len(span_index_list), option_num, max_span_num, dtype=torch.long)
        for b in range(len(span_index_list)):
            for o in range(len(span_index_list[b])):
                span_mask[b][o][:len(span_index_list[b][o])] = torch.ones(len(span_index_list[b][o]), dtype=torch.long)
                for s in range(len(span_index_list[b][o])):
                    span_occur_mask[b][o][s][:len(span_index_list[b][o][s])] = torch.ones(len(span_index_list[b][o][s]), dtype=torch.long)
                    for oc in range(len(span_index_list[b][o][s])):
                        span_len = span_index_list[b][o][s][oc][1] - span_index_list[b][o][s][oc][0]
                        span_index[b][o][s][oc][:span_len] = torch.arange(span_index_list[b][o][s][oc][0], span_index_list[b][o][s][oc][1])
                        span_token_mask[b][o][s][oc][:span_len] = torch.ones(span_len)

        decoder_input_ids = torch.zeros(len(span_index_list), option_num, self.sample_num, max_span_num, dtype=torch.long)
        if self.sample_num > 0:
            for b in range(len(span_index_list)):
                for o in range(len(span_index_list[b])):
                    span_ids = list(range(len(span_index_list[b][o])))
                    for r in range(self.sample_num):
                        random.shuffle(span_ids)
                        decoder_input_ids[b][o][r][:len(span_ids)] = torch.tensor(copy.deepcopy(span_ids))
        else:
            raise NotImplementedError

        return span_index, span_token_mask, span_occur_mask, span_mask, decoder_input_ids
