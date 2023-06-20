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
from torch.nn import CrossEntropyLoss
import random
import numpy as np
from transformers.models.llama.modeling_llama import (
    LlamaModel,
    LlamaForCausalLM,
    LlamaPreTrainedModel,
    LlamaConfig,
    SequenceClassifierOutputWithPast,
    LlamaDecoderLayer,
    BaseModelOutputWithPast,
)

from general_util.logger import get_child_logger
from general_util.mixin import LogMixin
from modules.layers import fold_tensor, get_accuracy
from deepspeed.pipe import PipelineModule, TiedLayerSpec, LayerSpec

logger = get_child_logger(__name__)

# LORA_TARGET_MODULES = [
#     "q_proj",
#     "v_proj",
# ]

PAD_TOKEN_ID = 32000


# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(
        input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.tensor(torch.finfo(dtype).min, device=device), device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


# Copied from transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask
def _prepare_decoder_attention_mask(attention_mask, input_shape, inputs_embeds, past_key_values_length):
    # create causal mask
    # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
    combined_attention_mask = None
    if input_shape[-1] > 1:
        combined_attention_mask = _make_causal_mask(
            input_shape,
            inputs_embeds.dtype,
            device=inputs_embeds.device,
            past_key_values_length=past_key_values_length,
        )

    if attention_mask is not None:
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
            inputs_embeds.device
        )
        combined_attention_mask = (
            expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
        )

    return combined_attention_mask


class EmbeddingPipeLayer(torch.nn.Module):
    def __init__(self, model: LlamaModel):
        super().__init__()
        self.embed_tokens = model.embed_tokens
        self.padding_idx = model.padding_idx
        self.weight = self.embed_tokens.weight

    def forward(self, ipt):
        input_ids, attention_mask, labels = ipt

        batch_size, seq_length = input_ids.shape
        seq_length_with_past = seq_length
        past_key_values_length = 0

        # device = input_ids.device if input_ids is not None else inputs_embeds.device
        device = input_ids.device
        position_ids = torch.arange(
            past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
        )
        position_ids = position_ids.unsqueeze(0).view(-1, seq_length)

        inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds

        return hidden_states, attention_mask, position_ids, labels


class LlamaPipeLayer(torch.nn.Module):
    def __init__(self, model: LlamaModel, layer_idx):
        super().__init__()
        self.layer = model.layers[layer_idx]
        self.layer_idx = torch.tensor(layer_idx)

    def forward(self, ipt):
        hidden_states, attention_mask, position_ids, labels = ipt
        layer_outputs = self.layer(hidden_states, attention_mask, position_ids)
        hidden_states = layer_outputs[0]
        return hidden_states, attention_mask, position_ids, labels


class FLNPipeLayer(torch.nn.Module):
    def __init__(self, model: LlamaModel):
        super().__init__()
        self.norm = model.norm

    def forward(self, ipt):
        hidden_states, attention_mask, position_ids, labels = ipt
        hidden_states = self.norm(hidden_states)
        return hidden_states, labels


class LMPipeLayer(torch.nn.Module):
    def __init__(self, model: LlamaModel):
        super().__init__()
        self.lm_head = model.lm_head

    def forward(self, ipt):
        hidden_states, labels = ipt
        logits = self.lm_head(hidden_states)
        return logits, labels


class LossPipeLayer(torch.nn.Module):
    def __init__(self, model: LlamaModel):
        super().__init__()

    def forward(self, ipt):
        logits, labels = ipt
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))
        return loss


def get_model(model):
    layers = [EmbeddingPipeLayer(model=model),
              *[LlamaPipeLayer(model=model, layer_idx=idx) for idx in
                range(model.config.num_layers)],
              FLNPipeLayer(model=model),
              TiedLayerSpec("embed_tokens", LMPipeLayer, model=model),
              LossPipeLayer(model=model)]
    return layers
