from torch.utils.tensorboard import SummaryWriter
from typing import List, Union, Dict, Tuple


class SummaryWriterHelper:
    def __init__(self,
                 writer: SummaryWriter,
                 batch_index_or_keys: Dict[str, Union[int, str]] = None,
                 outputs_index_or_keys: Dict[str, Union[int, str]] = None):
        """
        :param writer:
        :param batch_index_or_keys: use key to support dict and index (int) to support tuple.
        :param outputs_index_or_keys: use key to support dict and index (int) to support tuple.
        """
        self.writer = writer
        self.batch_index_or_keys = batch_index_or_keys
        self.outputs_index_or_keys = outputs_index_or_keys

    def __call__(self, step: int, last_batch: Union[Dict, Tuple] = None, last_outputs: Union[Dict, Tuple] = None):
        if last_batch is not None and self.batch_index_or_keys is not None:
            for name, k in self.batch_index_or_keys:
                self.writer.add_scalar(name, last_batch[k], global_step=step)
        if last_outputs is not None and self.outputs_index_or_keys is not None:
            for name, k in self.outputs_index_or_keys:
                self.writer.add_scalar(name, last_outputs[k], global_step=step)
