"""
Write your own collators under the directory.
"""
import torch
from torch.utils.data import Dataset, default_collate
from tqdm import tqdm
from multiprocessing import Pool

import base64
import functools
from general_util.logger import get_child_logger
from .tuple2dict import Tuple2DictCollator, NLITuple2DictCollator
from .wiki import WikiPathDatasetCollator, WikiPathDatasetCollatorOnlyMLM, WikiPathDatasetCollatorWithContext

logger = get_child_logger(__name__)


# def process_single_q_id(item, num_q: int):
#     i, q_id = item
#     mapping = {}
#     for j, k_id in enumerate(_k_ids):
#         mapping[i * num_q + j] = (i, j, q_id, k_id)
#     return mapping

#
# def process_single_k_id(item):
#     j, k_id = item
#     mapping = {}
#     for i, q_id in enumerate(_q_ids):
#         mapping[i * len(_q_ids) + j] = (i, j, q_id, k_id)
#     return mapping
#
#
# def _init_(q_ids):
#     global _q_ids
#     _q_ids = q_ids


class DotProductDataset(Dataset):
    def __init__(self, file_path: str, tokenizer, k_index_path: str, num_workers: int = 64, ):
        # q_memory = torch.load(q_index_path)
        q_memory = torch.load(file_path)
        k_memory = torch.load(k_index_path)
        self.q_index = q_memory["hidden_states"]
        self.k_index = k_memory["hidden_states"]
        self.q_ids = q_memory["indices"]
        self.k_ids = k_memory["indices"]

        logger.info(f"Q memory size: {self.q_index.size()}")
        logger.info(f"K memory size: {self.k_index.size()}")

        assert len(self.q_ids) == self.q_index.size(0), len(self.q_ids)
        assert len(self.k_ids) == self.k_index.size(0), len(self.k_ids)

        # self.mapping = {}
        # with Pool(num_workers, _init_, initargs=(self.q_ids,)) as p:
        #     # _annotate = functools.partial(process_single_k_id)
        #     _results = list(tqdm(
        #         p.imap(process_single_k_id, list(enumerate(self.k_ids)), chunksize=32),
        #         total=len(self.k_ids),
        #         desc="Reading examples"
        #     ))
        #     for res in tqdm(_results, total=len(_results)):
        #         self.mapping.update(res)
        # for i, q_id in tqdm(enumerate(self.q_ids), total=len(self.q_ids)):
        #     for j, k_id in enumerate(self.k_ids):
        #         self.mapping[i * len(self.q_ids) + j] = (i, j, q_id, k_id)

    def __len__(self):
        return len(self.q_ids) * len(self.k_ids)

    def __getitem__(self, index):
        # j = index % len(self.k_index)
        # i = index // len(self.q_index)

        i = index // len(self.k_ids)
        j = index % len(self.k_ids)
        q_id = self.q_ids[i]
        k_id = self.k_ids[j]
        # i, j, q_id, k_id = self.mapping[index]
        return {
            "q": self.q_index[i],
            "k": self.k_index[j],
            "meta_data": {
                "index": f"{q_id}-{k_id}"
            }
        }
