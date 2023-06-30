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
from .modeling_rw import RWModel, RWPreTrainedModel, RWForCausalLM, CausalLMOutputWithCrossAttentions
from .configuration_RW import RWConfig

from general_util.logger import get_child_logger
from general_util.mixin import LogMixin
from modules.layers import fold_tensor, get_accuracy

logger = get_child_logger(__name__)

LORA_TARGET_MODULES = [
    "q_proj",
    "v_proj",
]

PAD_TOKEN_ID = 32000


def find_all_linear_names(model, bits: int):
    cls = bnb.nn.Linear4bit if bits == 4 else (bnb.nn.Linear8bitLt if bits == 8 else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def return_single_device_map():
    return {"": "cuda:" + str(int(os.environ.get("LOCAL_RANK") or 0))}


def return_cpu_device_map():
    return {"": "cpu"}


def return_single_device_map_emb():
    return {"embed_tokens": "cuda:" + str(int(os.environ.get("LOCAL_RANK") or 0)), "": "cpu"}


@dataclass
class MultipleChoicePreTrainModelOutput(CausalLMOutputWithCrossAttentions):
    mlm_loss: torch.FloatTensor = None
    mlm_acc: torch.FloatTensor = None
    cls_loss: torch.FloatTensor = None
    cls_acc: torch.FloatTensor = None
    pair_loss: torch.FloatTensor = None
    tagging_loss: torch.FloatTensor = None
    path_gen_loss: torch.FloatTensor = None
    path_gen_acc: torch.FloatTensor = None
    ent_gen_loss: torch.FloatTensor = None
    ent_gen_acc: torch.FloatTensor = None
    rel_ctr_loss: torch.FloatTensor = None
    local_ctr_loss: torch.FloatTensor = None
    local_ctr_acc: torch.FloatTensor = None
    original_logits: torch.FloatTensor = None


class RWPreTrainedModelPeftMixin(RWPreTrainedModel, ABC):
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], *model_args, **kwargs):
        # if "pad_token_id" in kwargs:
        #     # Hack here to avoid embedding weight size mismatch during loading pre-trained weights.
        #     vocab_size = kwargs.pop("vocab_size")
        #     pad_token_id = kwargs.pop("pad_token_id")
        # else:
        #     vocab_size = None
        #     pad_token_id = None

        use_peft = kwargs.pop("use_peft", False)
        lora_config = kwargs.pop("lora_config", None)
        load_in_8bit = kwargs.pop("load_in_8bit", False)
        load_in_4bit = kwargs.pop("load_in_4bit", False)

        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

        # if vocab_size is not None and pad_token_id is not None:
        #     # assert vocab_size == model.config.vocab_size + 1, "Currently, only hack here to add pad token id is supported. "
        #     model.resize_token_embeddings(vocab_size)
        #     model.config.pad_token_id = pad_token_id

        if use_peft:
            if lora_config is None:
                lora_config = LoraConfig(task_type=TaskType.SEQ_CLS, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)

            # logger.info(*model_args)
            # logger.info(kwargs)
            logger.info(lora_config)

            # if vocab_size is not None and pad_token_id is not None:
            #     assert vocab_size == model.config.vocab_size + 1, "Currently, only hack here to add pad token id is supported. "
            #     model.resize_token_embeddings(vocab_size)
            #     model.config.pad_token_id = pad_token_id

            logger.info(f"LORA Config: {lora_config}")
            logger.info(lora_config.target_modules.__class__)
            if isinstance(lora_config.target_modules, omegaconf.listconfig.ListConfig):
                lora_config.target_modules = list(lora_config.target_modules)
            elif isinstance(lora_config.target_modules, omegaconf.DictConfig):
                lora_config.target_modules = hydra.utils.instantiate(lora_config.target_modules, model=model)
            else:
                raise ValueError(f"Unsupported type of target modules: {lora_config.target_modules.__class__}")

            logger.info(lora_config.target_modules.__class__)
            logger.info(lora_config.target_modules)
            gradient_checkpointing = model.transformer.gradient_checkpointing
            if load_in_8bit or load_in_4bit:
                # model = prepare_model_for_int8_training(model, use_gradient_checkpointing=gradient_checkpointing)
                model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=gradient_checkpointing)
            model = get_peft_model(model, lora_config)

            compute_dtype = kwargs["torch_dtype"]
            for name, module in model.named_modules():
                if isinstance(module, LoraLayer):
                    if compute_dtype == torch.bfloat16:
                        module = module.to(torch.bfloat16)
                if isinstance(module, torch.nn.LayerNorm):
                    module = module.to(torch.float32)
                if 'lm_head' in name or 'word_embeddings' in name:
                    if hasattr(module, 'weight'):
                        if compute_dtype and module.weight.dtype == torch.float32:
                            module = module.to(torch.bfloat16)

            model.print_trainable_parameters()

        logger.info(f"Config pad token id after loading pre-trained weights: {model.config.pad_token_id}")

        return model

    @classmethod
    def from_pretrained_peft_eval(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], *model_args, **kwargs):
        base_model_name_or_path = kwargs.pop("base_model_name_or_path", pretrained_model_name_or_path)

        model = super().from_pretrained(base_model_name_or_path, *model_args, **kwargs)
        model = PeftModel.from_pretrained(model, pretrained_model_name_or_path)
        return model


class RWForMultipleChoiceCLS(RWPreTrainedModelPeftMixin, LogMixin, ABC):
    _keys_to_ignore_on_load_missing = [r"lm_head.weight"]

    def __init__(self, config: RWConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.transformer = RWModel(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)
        # self.config.pad_token_id = self.config.eos_token_id
        # self.config.pad_token_id = self.config.vocab_size
        # self.config.pad_token_id = PAD_TOKEN_ID
        logger.info(f"Config pad token id: {self.config.pad_token_id}")

        # Initialize weights and apply final processing
        self.post_init()

        self.init_metric("loss", "acc")

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: torch.Tensor):
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(
            self,
            input_ids: torch.LongTensor,
            past: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            **kwargs,
    ) -> dict:
        # only last token for input_ids if past is not None
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)

            # the cache may be in the stardard format (e.g. in contrastive search), convert to our's format if needed
            if past[0][0].shape[0] == input_ids.shape[0]:
                past = self._convert_to_rw_cache(past)

        return {
            "input_ids": input_ids,
            "past_key_values": past,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
        }

    def get_cls_head(self):
        return self.score

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            **kwargs,
    ) -> Union[Tuple, MultipleChoicePreTrainModelOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        num_choices = input_ids.shape[1]
        input_ids = fold_tensor(input_ids)
        attention_mask = fold_tensor(attention_mask)

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = torch.zeros(batch_size, 1).fill_(-1).to(hidden_states.device)
        else:
            sequence_lengths = (torch.ne(input_ids, self.config.pad_token_id).sum(-1, keepdim=True) - 1).to(hidden_states.device)
        length_index = sequence_lengths.unsqueeze(-1).expand(-1, -1, hidden_states.size(-1)).contiguous()

        sentence_representation = torch.gather(hidden_states, 1, length_index).squeeze(1)
        reshaped_logits = self.score(sentence_representation).view(-1, num_choices)

        loss = 0.
        cls_loss = 0.
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            cls_loss = loss_fct(reshaped_logits, labels)
            loss = loss + cls_loss

            if not self.training:
                acc, true_label_num = get_accuracy(reshaped_logits, labels)
                self.eval_metrics.update("acc", val=acc, n=true_label_num)
                self.eval_metrics.update("loss", val=loss.item(), n=true_label_num)

        return MultipleChoicePreTrainModelOutput(
            loss=loss,
            cls_loss=cls_loss,
            logits=reshaped_logits,
        )

    def _reorder_cache(
            self, past: Tuple[Tuple[torch.Tensor, torch.Tensor], ...], beam_idx: torch.LongTensor
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.

        Output shares the same memory storage as `past`.
        """
        standardized_past = self._convert_to_standard_cache(past, batch_size=len(beam_idx))

        # Get a copy of `beam_idx` on all the devices where we need those indices.
        device_to_beam_idx = {
            past_state.device: beam_idx.to(past_state.device) for layer_past in past for past_state in layer_past
        }
        reordered_past = tuple(
            (
                layer_past[0].index_select(0, device_to_beam_idx[layer_past[0].device]),
                layer_past[1].index_select(0, device_to_beam_idx[layer_past[0].device]),
            )
            for layer_past in standardized_past
        )
        return self._convert_to_rw_cache(reordered_past)


class RWForMultipleChoiceCausalLM(RWPreTrainedModelPeftMixin, LogMixin, ABC):
    _keys_to_ignore_on_load_missing = [r"lm_head.weight"]

    def __init__(self, config: RWConfig, add_lm_loss: bool = False):
        super().__init__(config)
        self.transformer = RWModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # self.config.pad_token_id = self.config.eos_token_id
        # self.config.pad_token_id = self.config.vocab_size
        # self.config.pad_token_id = PAD_TOKEN_ID
        logger.info(f"Config pad token id: {self.config.pad_token_id}")

        # Initialize weights and apply final processing
        self.post_init()

        self.add_lm_loss = add_lm_loss

        metrics = ["loss", "acc", "cls_loss"]
        if add_lm_loss:
            metrics.extend(["mlm_loss"])
        self.init_metric(*metrics)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: torch.Tensor):
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(
            self,
            input_ids: torch.LongTensor,
            past: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            **kwargs,
    ) -> dict:
        # only last token for input_ids if past is not None
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)

            # the cache may be in the stardard format (e.g. in contrastive search), convert to our's format if needed
            if past[0][0].shape[0] == input_ids.shape[0]:
                past = self._convert_to_rw_cache(past)

        return {
            "input_ids": input_ids,
            "past_key_values": past,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
        }

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            input_lens: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, MultipleChoicePreTrainModelOutput]:
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
                Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
                shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of
                shape `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`. The two additional
                tensors are only required when the model is used as a decoder in a Sequence to Sequence model.

                Contains pre-computed hidden-states (key and values in the self-attention blocks and in the
                cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

                If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those
                that don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of
                all `decoder_input_ids` of shape `(batch_size, sequence_length)`.
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.

        Returns:

        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        num_choices = input_ids.shape[1]
        batch_size = input_ids.shape[0]

        input_ids = fold_tensor(input_ids)
        attention_mask = fold_tensor(attention_mask)

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        shifted_logits = logits[..., :-1, :].contiguous()

        label_mask = input_ids.ne(self.config.pad_token_id)
        # logger.info(label_mask[0])
        # if input_lens is not None:
        # keep only logits after the end of the condition part in each item of the batch
        # [batch_size * num_choices, input_lens]
        if input_lens is not None:
            lens_mask = torch.arange(input_ids.size(1), device=label_mask.device)[None, :] >= input_lens[:, None]
            # logger.info(lens_mask[0])
            label_mask = label_mask & lens_mask
        # logger.info(label_mask[0])
        lm_labels = input_ids.masked_fill(~label_mask, -100).contiguous()
        shifted_lm_labels = lm_labels[..., 1:].contiguous()

        lm_ppl = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")(shifted_logits.view(-1, logits.size(-1)),
                                                                          shifted_lm_labels.view(-1))
        lm_ppl = lm_ppl.reshape(batch_size, num_choices, input_ids.size(1) - 1)
        # [batch_size, num_choices]
        true_seq_len = label_mask.to(dtype=lm_ppl.dtype).reshape(batch_size, num_choices, -1).sum(-1).detach()
        # logger.info(true_seq_len.sum().item())
        no_seq_mask = true_seq_len.eq(0)
        lm_ppl = lm_ppl.sum(-1) / (true_seq_len + no_seq_mask.to(dtype=lm_ppl.dtype))
        assert lm_ppl.size() == (batch_size, num_choices)

        reshaped_logits = -lm_ppl
        reshaped_logits = reshaped_logits.masked_fill(no_seq_mask, -10000.0)

        loss = 0.
        cls_loss = 0.
        lm_loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            cls_label_mask = torch.gather(no_seq_mask, 1, labels[:, None]).squeeze(1)
            masked_labels = labels.masked_fill(cls_label_mask, -1)
            cls_loss = loss_fct(reshaped_logits, masked_labels)
            if masked_labels.ne(-1).sum().item() > 0:
                loss = loss + cls_loss
            else:
                cls_loss = 0.

            if self.add_lm_loss:
                # lm_loss = lm_ppl[:, 0].mean()
                masked_lm_ppl = lm_ppl.masked_fill(no_seq_mask, 0.0)
                lm_loss = torch.gather(masked_lm_ppl, 1, labels[:, None]).squeeze(1).mean()
                loss = loss + lm_loss

            if not self.training:
                acc, true_label_num = get_accuracy(reshaped_logits, masked_labels)
                self.eval_metrics.update("acc", val=acc, n=true_label_num)
                self.eval_metrics.update("loss", val=loss.item(), n=true_label_num)
                self.eval_metrics.update("cls_loss", val=cls_loss, n=true_label_num)

                if self.add_lm_loss:
                    self.eval_metrics.update("mlm_loss", val=lm_loss, n=1)

        return MultipleChoicePreTrainModelOutput(
            loss=loss,
            logits=reshaped_logits,
            cls_loss=cls_loss,
            mlm_loss=lm_loss,
        )

    def _reorder_cache(
            self, past: Tuple[Tuple[torch.Tensor, torch.Tensor], ...], beam_idx: torch.LongTensor
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.

        Output shares the same memory storage as `past`.
        """
        standardized_past = self._convert_to_standard_cache(past, batch_size=len(beam_idx))

        # Get a copy of `beam_idx` on all the devices where we need those indices.
        device_to_beam_idx = {
            past_state.device: beam_idx.to(past_state.device) for layer_past in past for past_state in layer_past
        }
        reordered_past = tuple(
            (
                layer_past[0].index_select(0, device_to_beam_idx[layer_past[0].device]),
                layer_past[1].index_select(0, device_to_beam_idx[layer_past[0].device]),
            )
            for layer_past in standardized_past
        )
        return self._convert_to_rw_cache(reordered_past)


class RWForConditionalGeneration(RWPreTrainedModelPeftMixin, LogMixin, ABC):
    _keys_to_ignore_on_load_missing = [r"lm_head.weight"]

    def __init__(self, config: RWConfig, gradient_checkpointing=False):
        super().__init__(config)
        self.transformer = RWModel(config)
        # set gradient checkpointing
        # self.model.gradient_checkpointing = gradient_checkpointing
        if gradient_checkpointing:
            self.config.use_cache = False
            self.gradient_checkpointing_enable()
        logger.info(f"gradient_checkpointing: {gradient_checkpointing}")

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # logger.info(f"Config pad token id: {self.config.pad_token_id}")

        # Initialize weights and apply final processing
        self.post_init()

        metrics = ["loss", "acc"]
        self.init_metric(*metrics)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: torch.Tensor):
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(
            self,
            input_ids: torch.LongTensor,
            past: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            **kwargs,
    ) -> dict:
        # only last token for input_ids if past is not None
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)

            # the cache may be in the stardard format (e.g. in contrastive search), convert to our's format if needed
            if past[0][0].shape[0] == input_ids.shape[0]:
                past = self._convert_to_rw_cache(past)

        return {
            "input_ids": input_ids,
            "past_key_values": past,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
        }

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            input_lens: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            **kwargs,
    ) -> Union[Tuple, MultipleChoicePreTrainModelOutput]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size = input_ids.shape[0]

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        shifted_logits = logits[..., :-1, :].contiguous()

        label_mask = input_ids.ne(self.config.pad_token_id)
        # logger.info(label_mask[0])
        # if input_lens is not None:
        # keep only logits after the end of the condition part in each item of the batch
        # [batch_size * num_choices, input_lens]
        if input_lens is not None:
            lens_mask = torch.arange(input_ids.size(1), device=label_mask.device)[None, :] >= input_lens[:, None]
            # logger.info(lens_mask[0])
            label_mask = label_mask & lens_mask
        # logger.info(label_mask[0])
        lm_labels = input_ids.masked_fill(~label_mask, -1).contiguous()
        shifted_lm_labels = lm_labels[..., 1:].contiguous()

        # loss = 0.
        # if labels is not None:
        loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
        lm_loss = loss_fct(shifted_logits.view(-1, logits.size(-1)), shifted_lm_labels.view(-1))
        loss = lm_loss
        if torch.isnan(lm_loss):
            print(lm_labels)
            print(input_lens)
            print("========================")

        if not self.training:
            acc, true_label_num = get_accuracy(shifted_logits, shifted_lm_labels)
            self.eval_metrics.update("acc", val=acc, n=true_label_num)
            self.eval_metrics.update("loss", val=loss.item(), n=true_label_num)

            score_loss_fct = nn.CrossEntropyLoss(ignore_index=-1, reduction="none")
            score_loss = score_loss_fct(shifted_logits.view(-1, logits.size(-1)), shifted_lm_labels.view(-1))
            score_loss = score_loss.reshape(batch_size, -1)
            score_loss = score_loss.sum(dim=-1) / label_mask.sum(dim=-1).float()
            return MultipleChoicePreTrainModelOutput(
                loss=loss,
                logits=-score_loss,
            )
        return MultipleChoicePreTrainModelOutput(
            loss=loss,
            logits=shifted_logits,
        )

    def _reorder_cache(
            self, past: Tuple[Tuple[torch.Tensor, torch.Tensor], ...], beam_idx: torch.LongTensor
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.

        Output shares the same memory storage as `past`.
        """
        standardized_past = self._convert_to_standard_cache(past, batch_size=len(beam_idx))

        # Get a copy of `beam_idx` on all the devices where we need those indices.
        device_to_beam_idx = {
            past_state.device: beam_idx.to(past_state.device) for layer_past in past for past_state in layer_past
        }
        reordered_past = tuple(
            (
                layer_past[0].index_select(0, device_to_beam_idx[layer_past[0].device]),
                layer_past[1].index_select(0, device_to_beam_idx[layer_past[0].device]),
            )
            for layer_past in standardized_past
        )
        return self._convert_to_rw_cache(reordered_past)


class RWForConditionalGenerationFlan(RWForConditionalGeneration, LogMixin, ABC):
    def __init__(self, config: RWConfig, gradient_checkpointing=False, merit_ratio: float = 0.5):
        super().__init__(config, gradient_checkpointing)
        self.merit_ratio = merit_ratio

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            input_lens: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            flan_input_ids: Optional[torch.LongTensor] = None,
            flan_attention_mask: Optional[torch.FloatTensor] = None,
            flan_token_type_ids: Optional[torch.LongTensor] = None,
            flan_input_lens: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, MultipleChoicePreTrainModelOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs1 = super().forward(input_ids=input_ids,
                                   attention_mask=attention_mask,
                                   input_lens=input_lens,
                                   return_dict=return_dict)
        outputs2 = super().forward(input_ids=flan_input_ids,
                                   attention_mask=flan_attention_mask,
                                   input_lens=flan_input_lens,
                                   return_dict=return_dict)
        # if torch.isnan(outputs1.loss):
        #     print("Normal inputs NAN loss")
        # if torch.isnan(outputs2.loss):
        #     print("Flan inputs NAN loss")

        # loss = (outputs1.loss + outputs2.loss) / 2
        loss = self.merit_ratio * outputs1.loss + (1 - self.merit_ratio) * outputs2.loss

        return MultipleChoicePreTrainModelOutput(
            loss=loss,
            mlm_loss=outputs1.loss,
            logits=outputs1.logits,
        )


def mask_according_lens(input_ids, input_lens, pad_token_id):
    label_mask = input_ids.ne(pad_token_id)
    # keep only logits after the end of the condition part in each item of the batch
    # [batch_size * num_choices, input_lens]
    lens_mask = torch.arange(input_ids.size(1), device=label_mask.device)[None, :] >= input_lens[:, None]
    label_mask = label_mask & lens_mask
    return label_mask


def token_wise_ctr_forward(
        model: RWModel,
        linear_layer: nn.Module,
        input_ids: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        input_lens: Optional[torch.Tensor] = None,
        pad_token_id: int = 0,
):
    batch_size, num_choice = input_ids.size()[:2]
    input_ids = fold_tensor(input_ids)
    attention_mask = fold_tensor(attention_mask)

    outputs = model(input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True)

    hidden_states = outputs[0]
    label_mask = mask_according_lens(input_ids, input_lens, pad_token_id)
    # [batch_size * num_choices, seq_len]
    token_logits = linear_layer(hidden_states).squeeze(-1)
    logits = token_logits.masked_fill(~label_mask, 0).sum(dim=1) / label_mask.sum(dim=1)
    logits = logits.view(batch_size, num_choice)
    return logits

# class LlamaCtrAndLMPretrain(LlamaForConditionalGeneration, ABC):
#     def __init__(self, config: LlamaConfig, gradient_checkpointing=False):
#         super().__init__(config, gradient_checkpointing)
#
#         self.linear = nn.Linear(config.hidden_size, 1)
#         self.init_metric("loss", "acc", "cls_loss")
#
#     def forward(
#             self,
#             input_ids: torch.LongTensor = None,
#             attention_mask: Optional[torch.Tensor] = None,
#             token_type_ids: Optional[torch.Tensor] = None,
#             input_lens: Optional[torch.Tensor] = None,
#             past_key_values: Optional[List[torch.FloatTensor]] = None,
#             inputs_embeds: Optional[torch.FloatTensor] = None,
#             labels: Optional[torch.LongTensor] = None,
#             flan_input_ids: Optional[torch.LongTensor] = None,
#             flan_attention_mask: Optional[torch.FloatTensor] = None,
#             flan_token_type_ids: Optional[torch.LongTensor] = None,
#             flan_input_lens: Optional[torch.LongTensor] = None,
#             use_cache: Optional[bool] = None,
#             output_attentions: Optional[bool] = None,
#             output_hidden_states: Optional[bool] = None,
#             return_dict: Optional[bool] = None,
#     ) -> Union[Tuple, MultipleChoicePreTrainModelOutput]:
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict
#
#         ctr_logits = token_wise_ctr_forward(self.model, self.linear, input_ids, attention_mask, input_lens, self.config.pad_token_id)
#         ctr_loss = nn.CrossEntropyLoss()(ctr_logits, labels)
#
#         lm_outputs = super().forward(input_ids=flan_input_ids,
#                                      attention_mask=flan_attention_mask,
#                                      input_lens=flan_input_lens,
#                                      return_dict=return_dict)
#         lm_loss = lm_outputs.loss
#         loss = ctr_loss + lm_loss
#
#         ctr_acc = get_accuracy(ctr_logits, labels)
#         return MultipleChoicePreTrainModelOutput(
#             loss=loss,
#             logits=ctr_logits,
#             mlm_loss=lm_loss,
#             cls_loss=ctr_loss,
#             cls_acc=ctr_acc,
#         )