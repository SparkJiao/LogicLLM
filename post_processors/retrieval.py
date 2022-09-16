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


class MERItRetrieval:
    def __init__(self):
        self.scores_list = collections.defaultdict(list)

    def __call__(self, meta_data: List[Dict[str, Any]], batch_model_outputs: Dict[str, Any]):
        batch_size = len(meta_data)

        logits = batch_model_outputs["logits"].reshape(batch_size).detach().cpu().float().tolist()
        for meta, logit in zip(meta_data, logits):
            self.scores_list[meta["que_id"]].append((logit, meta["ctx_id"]))

    def get_results(self):
        sorted_index = {
            k: sorted(v, key=lambda x: x[0], reverse=True) for k, v in self.scores_list.items()
        }
        return {}, sorted_index
