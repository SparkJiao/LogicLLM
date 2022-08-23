import torch
from transformers.generation_logits_process import LogitsProcessor

from modules.trie import Trie


class TrieConstrainedLogitsProcessor(LogitsProcessor):
    def __init__(self, trie: Trie):
        self.trie = trie

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        sequence_ls = input_ids.tolist()
        scores_mask = scores.new_zeros(scores.size()).fill_(-10000.0)
        for seq_id, seq in enumerate(sequence_ls):
            output = self.trie.get(seq)
            scores_mask[seq_id, output] = 0.0
        return scores + scores_mask
