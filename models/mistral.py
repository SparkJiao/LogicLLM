from abc import ABC
from abc import ABC
from typing import Optional, List, Tuple, Union

import torch
from torch import nn
from transformers.models.llama.modeling_llama import SequenceClassifierOutputWithPast
from transformers.models.mistral.modeling_mistral import (
    MistralPreTrainedModel,
    MistralConfig,
    MistralModel,
)

from general_util.logger import get_child_logger
from general_util.mixin import LogMixin
from modules.layers import get_accuracy

logger = get_child_logger(__name__)


class MistralForSequenceClassification(MistralPreTrainedModel, ABC, LogMixin):
    def __init__(self, config: MistralConfig, gradient_checkpointing: bool = False, _flash_attn_2_enabled: bool = False):
        config._flash_attn_2_enabled = _flash_attn_2_enabled
        super().__init__(config)

        self.num_labels = config.num_labels
        self.model = MistralModel(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        self.gradient_checkpointing = gradient_checkpointing
        if self.gradient_checkpointing:
            self.config.use_cache = False
            self.gradient_checkpointing_enable()
        logger.info(f"gradient_checkpointing: {self.gradient_checkpointing}")
        logger.info(self.model.layers[0].self_attn.__class__.__name__)

        # Initialize weights and apply final processing
        self.post_init()

        metrics = ["loss", "acc"]
        self.init_metric(*metrics)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = (torch.eq(input_ids, self.config.pad_token_id).long().argmax(-1) - 1).to(
                    logits.device
                )
            else:
                sequence_lengths = -1

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))

            if not self.training:
                acc, num = get_accuracy(pooled_logits, labels)
                self.eval_metrics.update("acc", acc, num)
                self.eval_metrics.update("loss", loss.item(), n=labels.size(0))

        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
