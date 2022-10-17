import os
from typing import Dict, List, Any

import numpy as np
from torch import distributed as dist

from post_processors.dist_mixin import DistGatherMixin


class NumpySaver(DistGatherMixin):
    def __init__(self):
        self.predictions = []

    def __call__(self, meta_data: List[Dict[str, Any]], batch_model_outputs: Dict[str, Any], ddp: bool = False):

        logits = batch_model_outputs["logits"].detach().float()
        _, pred = logits.max(dim=-1)
        pred = pred.tolist()

        if ddp:
            obj = pred
            gather_res = self.gather_object(obj)
            if dist.get_rank() == 0:
                tmp = []
                for item in gather_res:
                    tmp.extend(item)
                pred = tmp
            # TODO: We need `index` in `meta_data` to re-order the predictions across different process.

        self.predictions.extend(pred)

    def get_results(self, output_dir: str):
        output_file = os.path.join(output_dir, "eval_predictions.npy")
        np.save(output_file, np.array(self.predictions))

        return {}, self.predictions
