from abc import ABC
from typing import Optional, Tuple, Union

import torch
from torch import nn
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXModel, GPTNeoXPreTrainedModel, GPTNeoXConfig, GPTNeoXForCausalLM, \
    CausalLMOutputWithPast

from general_util.logger import get_child_logger
from general_util.mixin import LogMixin
from modules.layers import get_accuracy

logger = get_child_logger(__name__)

LORA_TARGET_MODULES = [
    "q_proj",
    "v_proj",
]


class GPTNeoXForConditionalGeneration(GPTNeoXPreTrainedModel, LogMixin, ABC):
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]

    def __init__(self, config, gradient_checkpointing=False):
        super().__init__(config)

        self.gpt_neox = GPTNeoXModel(config)
        self.gpt_neox.gradient_checkpointing = gradient_checkpointing

        self.embed_out = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.config.pad_token_id = self.config.eos_token_id

        # Initialize weights and apply final processing
        self.post_init()

        metrics = ["loss", "acc"]
        self.init_metric(*metrics)

    def get_output_embeddings(self):
        return self.embed_out

    def set_output_embeddings(self, new_embeddings):
        self.embed_out = new_embeddings

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            input_lens: Optional[torch.LongTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`. The two additional tensors are
            only required when the model is used as a decoder in a Sequence to Sequence model.

            Contains pre-computed hidden-states (key and values in the self-attention blocks that can be used (see
            `past_key_values` input) to speed up sequential decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the left-to-right language modeling loss (next word prediction). Indices should be in
            `[-100, 0, ..., config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are
            ignored (masked), the loss is only computed for the tokens with labels n `[0, ..., config.vocab_size]`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, GPTNeoXForCausalLM, GPTNeoXConfig
        >>> import torch

        >>> tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        >>> config = GPTNeoXConfig.from_pretrained("EleutherAI/gpt-neox-20b")
        >>> config.is_decoder = True
        >>> model = GPTNeoXForCausalLM.from_pretrained("EleutherAI/gpt-neox-20b", config=config)

        >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
        >>> outputs = model(**inputs)

        >>> prediction_logits = outputs.logits
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.gpt_neox(
            input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        lm_logits = self.embed_out(hidden_states)

        shifted_logits = lm_logits[..., :-1, :].contiguous()

        label_mask = input_ids.ne(self.config.pad_token_id)
        if input_lens is not None:
            lens_mask = torch.arange(input_ids.size(1), device=input_ids.device)[None, :] >= input_lens[:, None]
            label_mask = label_mask & lens_mask

        lm_labels = input_ids.masked_fill(~label_mask, -1).contiguous()
        shifted_lm_labels = lm_labels[:, 1:].contiguous()

        loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
        loss = loss_fct(shifted_logits.view(-1, shifted_logits.size(-1)), shifted_lm_labels.view(-1))

        if not self.training:
            acc, true_label_num = get_accuracy(shifted_logits, shifted_lm_labels)
            self.eval_metrics.update("acc", val=acc, n=true_label_num)
            self.eval_metrics.update("loss", val=loss.item(), n=true_label_num)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=shifted_logits,
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **model_kwargs):
        input_shape = input_ids.shape

        # if model is used as a decoder in encoder-decoder model, the decoder attention mask is created on the fly
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_shape)

        # cut decoder_input_ids if past is used
        if past_key_values and past_key_values[0] is not None:
            input_ids = input_ids[:, -1:]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
        }

    def _reorder_cache(self, past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past
