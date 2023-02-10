from abc import ABC
from typing import Tuple, Optional, Union

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.models.gpt2.modeling_gpt2 import GPT2PreTrainedModel, GPT2Model, CausalLMOutputWithCrossAttentions, \
    get_device_map, assert_device_map, GPT2LMHeadModel
from transformers import AutoModelForCausalLM
from general_util.logger import get_child_logger


logger = get_child_logger(__name__)


class GPT2CausalLMInferenceInterface(GPT2LMHeadModel, ABC):
    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            token_type_ids: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_labels[shift_labels == self.config.eos_token_id] = -100
            shift_labels = shift_labels.contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(reduction="none")
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).reshape(shift_labels.size())
            non_pad_mask = (shift_labels != self.config.eos_token_id).to(loss.dtype)
            loss = (loss * non_pad_mask).sum(dim=-1) / non_pad_mask.sum(dim=-1)  # [b]
            assert loss.size() == (shift_labels.size(0),)

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )


class GPT2ForConditionalGeneration(GPT2LMHeadModel, ABC):
    def __init__(self, config, z_add_to_output: bool = False, z_add_to_head: bool = False):
        super().__init__(config)
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.z_add_to_output = z_add_to_output
        self.z_add_to_head = z_add_to_head

        # Model parallel
        self.model_parallel = False
        self.device_map = None

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            token_type_ids: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            target_attention_mask: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            z_hidden_states: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        # if inputs_embeds is not None:
        #     logger.warning("`inputs_embeds` will not be used.")
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        batch_size = input_ids.size(0) if input_ids is not None else inputs_embeds.size(0)

        if input_ids is None:
            input_len = inputs_embeds.size(1)
        else:
            input_len = input_ids.size(1)

        if target_attention_mask is None:
            label_mask = labels.ne(self.config.eos_token_id).to(dtype=attention_mask.dtype)
        else:
            label_mask = target_attention_mask

        if input_ids is None:
            label_embeds = self.transformer.wte(labels)
            inputs_embeds = torch.cat([inputs_embeds, label_embeds], dim=1)
        else:
            input_ids = torch.cat([input_ids, labels], dim=1)
            inputs_embeds = self.transformer.wte(input_ids)

        attention_mask = torch.cat([attention_mask, label_mask], dim=1)

        if z_hidden_states is not None:
            # inputs_embeds = self.transformer.wte(input_ids)
            inputs_embeds = torch.cat([z_hidden_states, inputs_embeds], dim=1)
            attention_mask = torch.cat([attention_mask.new_ones(batch_size, z_hidden_states.size(1)),  # FIXME: Fixed at 2023/02/08
                                        attention_mask], dim=1)
            assert attention_mask.size(1) == inputs_embeds.size(1), (attention_mask.size(), inputs_embeds.size())

            input_len = input_len + z_hidden_states.size(1)

            if self.z_add_to_output:
                inputs_embeds = inputs_embeds + z_hidden_states

            # input_ids = None

        transformer_outputs = self.transformer(
            input_ids=None,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        hidden_states = hidden_states[:, (input_len - 1):]

        if z_hidden_states is not None and self.z_add_to_head:
            hidden_states = hidden_states + z_hidden_states

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            # shift_labels = labels[..., 1:].contiguous() # Removed.
            shift_labels = labels.masked_fill(~label_mask.bool(), -100).contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )
