import collections
import json
import os
import re
from typing import Dict, List, Any, Union, Callable

import numpy as np
import torch
from torch import distributed as dist

from post_processors.dist_mixin import DistGatherMixin


class CategoryMetricSaver(DistGatherMixin):
    def __init__(self, save_copy: bool = False):
        self.predictions = []
        self.index = []
        self.save_copy = save_copy

    def __call__(self, meta_data: Dict[str, Any], batch_model_outputs: Dict[str, Any], ddp: bool = False):
        index = meta_data["index"].float().tolist()
        labels = meta_data["label"].tolist()
        inputs = meta_data["input"]
        category = meta_data["category"]

        logits = batch_model_outputs["logits"].detach().float()
        if logits.dim() == 1:
            logits = logits.reshape(len(labels), -1)
        _, pred = logits.max(dim=-1)
        pred = pred.tolist()

        if ddp:
            obj = [pred, index, labels, inputs, category]
            gather_res = self.gather_object(obj)
            if dist.get_rank() == 0:
                pred = []
                index = []
                labels = []
                inputs = []
                category = []
                for item in gather_res:
                    pred.extend(item[0])
                    index.extend(item[1])
                    labels.extend(item[2])
                    inputs.extend(item[3])
                    category.extend(item[4])

        self.predictions.extend([{
            "index": idx,
            "pred": p,
            "label": t,
            "input": src,
            "category": c
        } for idx, p, t, src, c in zip(index, pred, labels, inputs, category)])

    def get_results(self, output_dir: str):
        if dist.is_initialized():
            output_file = os.path.join(output_dir, f"eval_predictions_rank{dist.get_rank()}.json")
        else:
            output_file = os.path.join(output_dir, "eval_predictions.json")

        self.predictions = sorted(self.predictions, key=lambda x: x["index"])

        correct = 0
        existing_ids = set()
        npy_outputs = []
        category_correct = collections.defaultdict(list)
        for pred in self.predictions:
            if pred["index"] in existing_ids:
                continue
            existing_ids.add(pred["index"])
            if pred["pred"] == pred["label"]:
                correct += 1
            npy_outputs.append(pred["pred"])
            category_correct[pred["category"]].append(int(pred["pred"] == pred["label"]))

        metrics = {"overall_acc": round(correct / len(existing_ids), 3)}
        for k, v in category_correct.items():
            metrics[f"{k}_acc"] = round(sum(v) / len(v), 3)

        if not dist.is_initialized() or dist.get_rank() == 0:
            np.save(os.path.join(output_dir, "decode_results.npy"), np.array(npy_outputs))
            json.dump(metrics, open(os.path.join(output_dir, "metrics.json"), "w"))
            json.dump(self.predictions, open(output_file, "w"))
        assert len(npy_outputs) == len(existing_ids), (len(npy_outputs), len(self.predictions), len(existing_ids))
        return metrics, self.predictions
