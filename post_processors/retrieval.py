import collections
from typing import List, Union, Tuple, Dict, Any

import torch

from general_util.logger import get_child_logger

logger = get_child_logger(__name__)


class RetrievalResultsBase:
    def __init__(self, top: Union[List[int], Tuple[int]] = (1, 3, 5, 10)):
        self.scores = []
        self.answer_ids: List[List[int]] = []
        self.top = top

    def __call__(self, meta_data: List[Dict[str, Any]], batch_model_outputs: Dict[str, Any]):
        self.scores.append(batch_model_outputs["scores"])
        self.answer_ids.extend([meta["answer_id"] for meta in meta_data])
        del batch_model_outputs, meta_data

    def get_results(self):
        scores = torch.cat(self.scores, dim=0)
        _, sorted_indices = torch.sort(scores, descending=True, dim=-1)
        logger.info(sorted_indices.size())
        sorted_indices = sorted_indices.tolist()

        recall = collections.defaultdict(list)
        for answer_id, predictions in zip(self.answer_ids, sorted_indices):
            for k in self.top:
                recall_k = len(set(answer_id) & set(predictions[:k])) * 1.0 / len(answer_id)
                recall["recall@{}".format(k)].append(recall_k)

        res = {}
        for k in recall:
            res[k] = sum(recall[k]) * 1.0 / len(recall[k]) if len(recall[k]) > 0 else 0.0

        return res, sorted_indices
