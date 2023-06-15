import os
from abc import ABC
from dataclasses import dataclass
from typing import Optional, List, Tuple, Union

import bitsandbytes as bnb
import hydra.utils
import omegaconf
import torch
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel,
    prepare_model_for_kbit_training,
)
from peft.tuners.lora import LoraLayer
from torch import nn
from transformers.models.llama.modeling_llama import (
    LlamaModel,
    LlamaForCausalLM,
    LlamaPreTrainedModel,
    LlamaConfig,
    SequenceClassifierOutputWithPast,
    LlamaDecoderLayer,
)

from general_util.logger import get_child_logger
from general_util.mixin import LogMixin
from modules.layers import fold_tensor, get_accuracy

logger = get_child_logger(__name__)

# LORA_TARGET_MODULES = [
#     "q_proj",
#     "v_proj",
# ]

PAD_TOKEN_ID = 32000




class EmbeddingPipeLayer(nn.Module):
    def __init__(self, model: LlamaForCausalLM):
        super().__init__()
        self.word_embeddings: nn.Embedding = model.model.embed_tokens
        self.weight = self.word_embeddings.weight

    def forward(self, ipt):
        input_ids, labels = ipt
        hidden_states = self.word_embeddings(input_ids)

        device = input_ids.device
        position_ids = torch.arange(
            past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
        )
        position_ids = position_ids.unsqueeze(0).view(-1, seq_length)

        mask_token = 150001
        seqs = input_ids.tolist()
        mask_positions = [seq.index(mask_token) for seq in seqs]
        attention_mask = get_masks(input_ids, device=input_ids.device)
        position_ids = get_position_ids(input_ids, device=input_ids.device, mask_positions=mask_positions)
        hidden_states = hidden_states.transpose(0, 1).contiguous()
        return hidden_states, position_ids, attention_mask, labels

