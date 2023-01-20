import os
from typing import Dict, List, Any

import numpy as np
from torch import distributed as dist

from post_processors.dist_mixin import DistGatherMixin
import json

class NumpySaver(DistGatherMixin):
    def __init__(self):
        self.predictions = []
        self.index = []

    def __call__(self, meta_data: List[Dict[str, Any]], batch_model_outputs: Dict[str, Any], ddp: bool = False):

        logits = batch_model_outputs["logits"].detach().float()
        _, pred = logits.max(dim=-1)
        pred = pred.tolist()

        index = None
        if ddp:
            assert meta_data
            if isinstance(meta_data, list):
                index = [meta['index'].item() for meta in meta_data]
            elif isinstance(meta_data, dict):
                index = meta_data["index"].tolist()
            else:
                raise RuntimeError()
            obj = [pred, index]
            gather_res = self.gather_object(obj)
            if dist.get_rank() == 0:
                pred = []
                index = []
                for item in gather_res:
                    pred.extend(item[0])
                    index.extend(item[1])

        if index is not None:
            self.index.extend(index)
        self.predictions.extend(pred)

    def get_results(self, output_dir: str):
        # output_file = os.path.join(output_dir, "eval_predictions.npy")
        if dist.is_initialized():
            output_file = os.path.join(output_dir, f"eval_predictions_rank{dist.get_rank()}.npy")
        else:
            output_file = os.path.join(output_dir, "eval_predictions.npy")

        if len(self.index):
            assert len(self.index) == len(self.predictions)
            predictions = {idx: pred for idx, pred in zip(self.index, self.predictions)}
            predictions = sorted(predictions.items(), key=lambda x: x[0])
            predictions = [pred[1] for pred in predictions]
            np.save(output_file, np.array(predictions))
        else:
            np.save(output_file, np.array(self.predictions))

        return {}, self.predictions


class TaggingSaver(DistGatherMixin):
    def __init__(self):
        self.logits = []
        self.index = []

    def __call__(self, meta_data: List[Dict[str, Any]], batch_model_outputs: Dict[str, Any], ddp: bool = False):
        tagging_logits = batch_model_outputs["tagging_logits"].detach().float().tolist()

        index = None
        if ddp:
            assert meta_data
            if isinstance(meta_data, list):
                index = [meta['index'].item() for meta in meta_data]
            elif isinstance(meta_data, dict):
                index = meta_data["index"].tolist()
            else:
                raise RuntimeError()
            obj = [tagging_logits, index]
            gather_res = self.gather_object(obj)
            if dist.get_rank() == 0:
                tagging_logits = []
                index = []
                for item in gather_res:
                    tagging_logits.extend(tagging_logits)
                    index.extend(item[1])

        if index is not None:
            self.index.extend(index)
        self.logits.extend(tagging_logits)

    def get_results(self, output_dir: str):
        if dist.is_initialized():
            output_file = os.path.join(output_dir, f"tagging_logits_rank{dist.get_rank()}.json")
        else:
            output_file = os.path.join(output_dir, "tagging_logits.json")

        if len(self.index):
            assert len(self.index) == len(self.logits)
            predictions = {idx: pred for idx, pred in zip(self.index, self.logits)}
            predictions = sorted(predictions.items(), key=lambda x: x[0])
            predictions = [pred[1] for pred in predictions]
            json.dump(predictions, open(output_file, "w"))
        else:
            json.dump(self.logits, open(output_file, "w"))

        return {}, self.logits

