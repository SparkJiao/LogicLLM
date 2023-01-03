from typing import Dict, Union, List, Any

import torch
from torch import Tensor
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from transformers.tokenization_utils import BatchEncoding


class DictTensorDataset(Dataset):
    def __init__(self, data: Union[Dict[str, Tensor], BatchEncoding],
                 meta_data: Union[List[Dict[str, Any]], Dict[str, List[Any]]] = None):
        self.data = data
        self.meta_data = meta_data
        self.keys = list(self.data.keys())
        for v in self.data.values():
            if meta_data is not None:
                if isinstance(meta_data, list):
                    assert len(v) == len(meta_data)
                elif isinstance(meta_data, dict):
                    for meta_v in meta_data.values():
                        assert len(v) == len(meta_v)
                else:
                    raise RuntimeError()
            else:
                assert len(v) == self.data[self.keys[0]].size(0)

    def __len__(self):
        return self.data[self.keys[0]].size(0)

    def __getitem__(self, idx):
        res = {k: v[idx] for k, v in self.data.items()}
        if self.meta_data is not None:
            if isinstance(self.meta_data, list):
                res["meta_data"] = self.meta_data[idx]
            elif isinstance(self.meta_data, dict):
                res["meta_data"] = {k: v[idx] for k, v in self.meta_data.items()}
        if "index" not in res or "index" not in res["meta_data"]:
            res["index"] = torch.LongTensor([idx])
        return res


class MetaCollator:
    def __call__(self, batch):
        if "meta_data" not in batch[0]:
            return default_collate(batch)

        meta_data = [b.pop("meta_data") for b in batch]
        batch = default_collate(batch)
        batch["meta_data"] = meta_data
        return batch
