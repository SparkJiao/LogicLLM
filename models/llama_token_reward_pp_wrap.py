import os

import deepspeed
import torch
from torch import nn
from deepspeed.pipe import TiedLayerSpec, LayerSpec
from torch.nn import CrossEntropyLoss
from transformers.models.llama.modeling_llama import (
    LlamaForCausalLM,
    LlamaConfig,
    LlamaDecoderLayer,
    LlamaRMSNorm,
)
from models.llama import llama_fast_attention_wrap

from general_util.logger import get_child_logger

logger = get_child_logger(__name__)


class EmbeddingPipe(torch.nn.Embedding):
    def forward(self, args):
        input_ids, attention_mask, position_ids, condition_mask = args
        inputs_embeds = super().forward(input_ids)
        return inputs_embeds, attention_mask, position_ids, condition_mask


class ParallelTransformerLayerPipe(LlamaDecoderLayer):
    def __init__(self, config: LlamaConfig,
                 activation_checkpointing: bool = False,
                 enable_flash_attention: bool = False,
                 flash_attention_vanilla_torch: bool = False,
                 flash_attention_var_len: bool = False,
                 ):
        super().__init__(config)
        self.activation_checkpointing = activation_checkpointing

        if enable_flash_attention:
            llama_fast_attention_wrap(self.self_attn, vanilla_torch=flash_attention_vanilla_torch, var_len=flash_attention_var_len)

    def forward(self, args):
        if self.activation_checkpointing:
            return self._ckpt_forward(args)

        hidden_states, attention_mask, position_ids, condition_mask = args
        outputs = LlamaDecoderLayer.forward(self,
                                            hidden_states,
                                            attention_mask,
                                            position_ids,
                                            )
        return outputs[0], attention_mask, position_ids

    def _ckpt_forward(self, args):
        hidden_states, attention_mask, position_ids, condition_mask = args

        def create_custom_forward(module):
            def custom_forward(*inputs):
                return LlamaDecoderLayer.forward(module, *inputs)

            return custom_forward

        # deepspeed checkpoint auto use outputs[0] if len(outputs) == 1
        outputs = deepspeed.checkpointing.checkpoint(
            create_custom_forward(self),
            hidden_states,
            attention_mask,
            position_ids,
            None,
        )

        return outputs, attention_mask, position_ids, condition_mask


class LayerNormPipe(LlamaRMSNorm):
    def forward(self, args):
        hidden_states, _, _, condition_mask = args
        last_hidden_states = super().forward(hidden_states)
        return last_hidden_states, condition_mask


class LMAndTokenClsDoubleHeadPipe(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.rw_head = nn.Linear(config.hidden_size, 1, bias=False)

    def forward(self, args):
        hidden_states, condition_mask = args

        lm_logits = self.lm_head(hidden_states)
        rw_logits = self.rw_head(hidden_states).squeeze(-1)
        rw_logits = (rw_logits * condition_mask).sum(dim=1) / condition_mask.sum(dim=1)
        assert rw_logits.shape == (hidden_states.shape[0],)
        return lm_logits, rw_logits
    
    def load_state_dict(self, state_dict, strict: bool = True):  # Hack here to avoid weights mismatch.
        if "weight" in state_dict and "lm_head.weight" not in state_dict:
            tmp = state_dict.pop("weight")
            state_dict["lm_head.weight"] = tmp
        
        return super().load_state_dict(state_dict, strict=False)
            


class LMAndRewardLoss:
    def __init__(self, ignore_index: int = -100, lm_ratio: float = 0.5):
        self.ignore_index = ignore_index
        self.lm_ratio = lm_ratio

    def __call__(self, outputs, labels):
        lm_logits, rw_logits = outputs

        batch_size = lm_logits.shape[0] // 2

        lm_logits = lm_logits[:batch_size]
        shifted_lm_logits = lm_logits[..., :-1, :].contiguous()
        
        labels = labels[:batch_size]
        shifted_labels = labels[..., 1:].contiguous()

        pos_rw_logits = rw_logits[:batch_size]
        neg_rw_logits = rw_logits[batch_size:]

        loss_fct = CrossEntropyLoss(ignore_index=self.ignore_index)
        lm_loss = loss_fct(shifted_lm_logits.reshape(-1, shifted_lm_logits.size(-1)), shifted_labels.reshape(-1))

        rw_loss = -torch.nn.LogSigmoid()(pos_rw_logits - neg_rw_logits).mean()
        loss = self.lm_ratio * lm_loss + (1 - self.lm_ratio) * rw_loss
        return loss


def get_layers_from_config(model_config,
                           activation_checkpointing: bool = False,
                           enable_flash_attention: bool = False,
                           flash_attention_vanilla_torch: bool = False,
                           flash_attention_var_len: bool = False,
                           ):
    """
    `tie_word_embeddings` in LLaMA is set to `false`.
    """
    if enable_flash_attention:
        logger.info("⚡⚡⚡ enable llama flash attention.")

    layers = [
        LayerSpec(EmbeddingPipe, model_config.vocab_size, model_config.hidden_size),
        *[LayerSpec(ParallelTransformerLayerPipe, model_config, activation_checkpointing, enable_flash_attention,
                    flash_attention_vanilla_torch, flash_attention_var_len)
          for _ in range(model_config.num_hidden_layers)],
        LayerSpec(LayerNormPipe, model_config.hidden_size, model_config.rms_norm_eps),
        LayerSpec(LMAndTokenClsDoubleHeadPipe, model_config),
    ]
    return layers
