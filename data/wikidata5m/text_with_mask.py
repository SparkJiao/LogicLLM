import json
from torch.utils.data import Dataset
from typing import List, Union
from transformers import PreTrainedTokenizer
from transformers.tokenization_utils import TruncationStrategy, PaddingStrategy
from torch import Tensor


class LogicCircleDataset(Dataset):
    def __init__(self, logical_circle: str, id2ent: str, id2rel: str, triplet2sent: str):
        super(self).__init__()

        self.logical_circle = json.load(open(logical_circle, 'r'))
        self.id2ent = json.load(open(id2ent, 'r'))
        self.id2rel = json.load(open(id2rel, 'r'))
        self.triplet2sent = json.load(open(triplet2sent, 'r'))

    def __iter__(self):
        pass

    def __len__(self):
        pass


class MLMBatchCollatorMixin:
    @staticmethod
    def _padding_to_sequence(input_ids: Union[Tensor, List[int]], sequence: List[int],
                             tokenizer: PreTrainedTokenizer, padding: int = 0):
        assert len(input_ids.size()) == 1
        special_token_mask = tokenizer.get_special_tokens_mask(input_ids, already_has_special_tokens=True)
        padded_sequence = []
        cnt = 0
        for mask in special_token_mask:
            if mask == 1:
                padded_sequence.append(sequence[cnt])
                cnt += 1
            else:
                padded_sequence.append(padding)
        assert len(padded_sequence) == len(input_ids)
        return padded_sequence


    @staticmethod
    def mask_text(text: List[str], token2word_index: List[List[int]], indicate_mask: List[List[int]],
                  tokenizer: PreTrainedTokenizer):
        model_inputs = tokenizer(text,
                                 truncation=True,
                                 padding=PaddingStrategy.LONGEST,
                                 return_tensors="pt")

        padded_token2word_index = []
        padded_indicate_mask = []
        for input_ids, item_tk2wd_index,


