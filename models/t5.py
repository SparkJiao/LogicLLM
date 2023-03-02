from abc import ABC
from typing import Optional, Tuple, Union, Dict

import torch
from nltk import word_tokenize
from nltk.translate.bleu_score import sentence_bleu
from torch import Tensor
from torch.nn import CrossEntropyLoss
from transformers.generation_logits_process import LogitsProcessor, LogitsProcessorList
from transformers.models.bart.modeling_bart import BartClassificationHead
from transformers.models.t5.modeling_t5 import (
    T5Config,
    T5Stack,
    Seq2SeqLMOutput, T5ForConditionalGeneration, BaseModelOutput,
    T5DenseActDense, T5DenseGatedActDense, T5LayerNorm
)
from transformers.models.t5.tokenization_t5 import T5Tokenizer
from dataclasses import dataclass

from general_util.logger import get_child_logger
from general_util.mixin import LogMixin
from modules import layers
from modules.layers import fold_tensor

logger = get_child_logger("T5")


@dataclass
class MultipleChoicePreTrainModelOutput(Seq2SeqLMOutput):
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


class T5LayerFF(torch.nn.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        if config.is_gated_act:
            self.DenseReluDense = T5DenseGatedActDense(config)
        else:
            self.DenseReluDense = T5DenseActDense(config)

        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = torch.nn.Dropout(config.dropout_rate)

    def _forward(self, hidden_states):
        forwarded_states = self.layer_norm(hidden_states)
        forwarded_states = self.DenseReluDense(forwarded_states)
        hidden_states = hidden_states + self.dropout(forwarded_states)
        return hidden_states

    def forward(self, hidden_states):
        # many t5/mt5 models are trained in bfloat16 and don't do well under mixed precision (fp16).
        # It appears that it's enough to disable autocast for this FF layer to avoid inf/nan
        # problems for the whole model
        # if torch.is_autocast_enabled():
        with torch.cuda.amp.autocast(enabled=False):
            hidden_states = hidden_states.to(torch.float32)
            return self._forward(hidden_states)
        # else:
        #     return self._forward(hidden_states)


class T5ForSeq2Seq(T5ForConditionalGeneration, LogMixin, ABC):

    def __init__(self, config: T5Config, tokenizer: str, fp16_compatible: bool = False,
                 logits_processor: LogitsProcessor = None):
        super().__init__(config)
        self.config = config

        # if fp16_compatible:  # This doesn't work on RTX 6000.
        #     for block_id, block in enumerate(self.encoder.block):
        #         assert isinstance(block.layer[-1], transformers.models.t5.modeling_t5.T5LayerFF)
        #         self.encoder.block[block_id].layer[-1] = T5LayerFF(config)
        #
        #     for block_id, block in enumerate(self.decoder.block):
        #         assert isinstance(block.layer[-1], transformers.models.t5.modeling_t5.T5LayerFF)
        #         self.decoder.block[block_id].layer[-1] = T5LayerFF(config)
        # if fp16_compatible:
        #     self.lm_scale_modifier = torch.nn.Parameter(torch.ones(config.d_model))
        #     self.fix_t5_fp16()
        # else:
        #     self.lm_scale_modifier = 1

        self.init_metric("loss", "acc", "bleu")

        self.logits_processor = logits_processor
        if self.logits_processor is not None:
            self.logits_processor = LogitsProcessorList([self.logits_processor])

        self.tokenizer = T5Tokenizer.from_pretrained(tokenizer, use_fast=False)

        self.generating = False

    # def fix_t5_fp16(self):
    #     emb_scaling = 1 / 32.0
    #     att_v_scaling = 1 / 4.0
    #     att_o_scaling = 1 / 8.0
    #     ff_wi_scaling = 1 / 4.0
    #     ff_wo_scaling = 1 / 1.0
    #     ff_ln_scaling = 1 / 2.0
    #
    #     assert att_v_scaling * att_o_scaling == emb_scaling
    #     assert ff_wi_scaling * ff_wi_scaling * ff_wo_scaling * ff_ln_scaling == emb_scaling
    #
    #     with torch.no_grad():
    #         self.shared.weight *= emb_scaling
    #         for unit in self.encoder.block:
    #             unit.layer[0].SelfAttention.v.weight *= att_v_scaling
    #             unit.layer[0].SelfAttention.o.weight *= att_o_scaling
    #             unit.layer[1].DenseReluDense.wi_0.weight *= ff_wi_scaling
    #             unit.layer[1].DenseReluDense.wi_1.weight *= ff_wi_scaling
    #             unit.layer[1].DenseReluDense.wo.weight *= ff_wo_scaling
    #             unit.layer[1].layer_norm.weight *= ff_ln_scaling
    #         for unit in self.decoder.block:
    #             unit.layer[0].SelfAttention.v.weight *= att_v_scaling
    #             unit.layer[0].SelfAttention.o.weight *= att_o_scaling
    #             unit.layer[1].EncDecAttention.v.weight *= att_v_scaling
    #             unit.layer[1].EncDecAttention.o.weight *= att_o_scaling
    #             unit.layer[2].DenseReluDense.wi_0.weight *= ff_wi_scaling
    #             unit.layer[2].DenseReluDense.wi_1.weight *= ff_wi_scaling
    #             unit.layer[2].DenseReluDense.wo.weight *= ff_wo_scaling
    #             unit.layer[2].layer_norm.weight *= ff_ln_scaling
    #         self.lm_scale_modifier /= emb_scaling

    def generate(self, *model_args, **model_kwargs):
        self.generating = True
        model_kwargs.pop("labels", None)
        if self.logits_processor is not None:
            res = super().generate(*model_args, **model_kwargs, logits_processor=self.logits_processor)
        else:
            res = super().generate(*model_args, **model_kwargs)
        self.generating = False
        return res

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            decoder_input_ids: Optional[torch.LongTensor] = None,
            decoder_attention_mask: Optional[torch.BoolTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            decoder_head_mask: Optional[torch.FloatTensor] = None,
            cross_attn_head_mask: Optional[torch.Tensor] = None,
            encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            **kwargs
    ) -> Union[Dict[str, Tensor], T5Stack]:
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Labels for computing the cross entropy classification loss.
            Indices should be in ``[0, ..., config.vocab_size - 1]``.

        Returns:

        Examples::

        >>> from transformers import T5Tokenizer, T5ForConditionalGeneration
        >>> import torch

        >>> tokenizer = T5Tokenizer.from_pretrained('t5-small')
        >>> model = T5ForConditionalGeneration.from_pretrained('t5-small')

        >>> input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", return_tensors="pt")).unsqueeze(0)  # Batch size 1
        >>> outputs = model.generate(input_ids)

        >>> print("Generated: {}".format(tokenizer.decode(outputs[0], skip_special_tokens=True)))

        Generated: Hello, my dog is cute
        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        # if head_mask is not None and decoder_head_mask is None:
        #     if self.config.num_layers == self.config.num_decoder_layers:
        #         warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
        #         decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim ** -0.5)

        # lm_logits = self.lm_head(sequence_output * self.lm_scale_modifier)
        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            label_padding_mask = labels == self.config.pad_token_id
            labels[label_padding_mask] = -1
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom):
            #  Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

            if not self.training:
                # Generate sentences for BLEU evaluation
                max_output_length = labels.size(1)
                # Greedy decoding.
                eval_gen_sentences = self.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=max_output_length,
                                                   num_beams=1, do_sample=False)
                eval_gen_sentences = self.tokenizer.batch_decode(eval_gen_sentences, skip_special_tokens=True)
                target = self.tokenizer.batch_decode(labels.masked_fill(label_padding_mask, self.config.pad_token_id),
                                                     skip_special_tokens=True)
                # bleu = sum(
                #     [sentence_bleu([tgt.split()], gen_sentence.split()) for tgt, gen_sentence in zip(target, eval_gen_sentences)]
                # ) / labels.size(0)
                bleu = sum(
                    [sentence_bleu([word_tokenize(tgt)], word_tokenize(gen_sentence))
                     for tgt, gen_sentence in zip(target, eval_gen_sentences)]
                ) / labels.size(0)  # FIXED: the value should be the averaged one.
                self.eval_metrics.update("bleu", bleu, n=labels.size(0))

                acc, true_label_num = layers.get_accuracy(lm_logits, labels)
                self.eval_metrics.update("acc", acc, n=true_label_num)
                self.eval_metrics.update("loss", loss.item(), n=true_label_num)

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


"""
To solve NCCL error:
https://www.cnblogs.com/marsggbo/p/16556963.html
https://github.com/pytorch/pytorch/issues/54822
https://github.com/Lightning-AI/lightning/issues/4420

"""


class T5ForMultipleChoiceAndSeq2Seq(T5ForConditionalGeneration, LogMixin, ABC):
    def __init__(self, config: T5Config, ):
        super().__init__(config)
        self.num_labels = getattr(config, "num_labels", 1)
        self.classification_head = BartClassificationHead(
            config.d_model,
            config.d_model,
            self.num_labels,
            config.dropout_rate,
        )

        # Initialize weights and apply final processing
        self.post_init()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

        metrics = ["loss", "acc", "mlm_loss", "mlm_acc", "cls_loss"]
        self.init_metric(*metrics)

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            decoder_input_ids: Optional[torch.LongTensor] = None,
            decoder_attention_mask: Optional[torch.BoolTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            decoder_head_mask: Optional[torch.FloatTensor] = None,
            cross_attn_head_mask: Optional[torch.Tensor] = None,
            encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            mlm_input_ids: Tensor = None,
            mlm_attention_mask: Tensor = None,
            mlm_labels: Tensor = None,
            **kwargs,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to `-100` are ignored (masked), the loss is only computed for
            labels in `[0, ..., config.vocab_size]`

        Returns:
        ```"""
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        num_choices = input_ids.shape[1]

        input_ids = fold_tensor(input_ids)
        attention_mask = fold_tensor(attention_mask)
        decoder_input_ids = fold_tensor(decoder_input_ids)

        if decoder_input_ids is not None:
            eos_mask = decoder_input_ids.eq(self.config.eos_token_id)
        else:
            eos_mask = input_ids.eq(self.config.eos_token_id)

        if decoder_input_ids is not None:
            decoder_input_ids = self._shift_right(decoder_input_ids)
        else:
            decoder_input_ids = self._shift_right(input_ids)

        outputs = super().forward(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  decoder_input_ids=decoder_input_ids,
                                  return_dict=return_dict,
                                  output_hidden_states=True)

        assert len(outputs.decoder_hidden_states)
        sequence_output = outputs.decoder_hidden_states[-1]
        lm_logits = outputs.logits

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.classification_head = self.classification_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.classification_head.dense.weight.device)

        sentence_representation = sequence_output[eos_mask, :].view(sequence_output.size(0), -1, sequence_output.size(-1))[:, -1, :]
        logits = self.classification_head(sentence_representation)
        reshaped_logits = logits.view(-1, num_choices)

        loss = 0.
        mlm_loss = 0.
        cls_loss = 0.
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            cls_loss = loss_fct(reshaped_logits, labels)
            loss = loss + cls_loss

            if mlm_labels is not None:
                mlm_outputs = super().forward(
                    input_ids=mlm_input_ids,
                    attention_mask=mlm_attention_mask,
                    labels=mlm_labels,
                    return_dict=return_dict
                )

                mlm_scores = mlm_outputs.logits
                mlm_loss = self.mlm_alpha * mlm_outputs.loss
            else:
                mlm_scores = None
                mlm_loss = None

            if not self.training:
                acc, true_label_num = layers.get_accuracy(reshaped_logits, labels)
                self.eval_metrics.update("acc", val=acc, n=true_label_num)
                self.eval_metrics.update("loss", val=loss.item(), n=true_label_num)
                self.eval_metrics.update("cls_loss", val=cls_loss.item(), n=true_label_num)

                if mlm_labels is not None:
                    acc, true_label_num = layers.get_accuracy(mlm_scores, mlm_labels)
                    self.eval_metrics.update("mlm_acc", val=acc, n=true_label_num)
                    self.eval_metrics.update("mlm_loss", val=mlm_loss.item(), n=true_label_num)

        return MultipleChoicePreTrainModelOutput(
            loss=loss,
            logits=reshaped_logits,
            mlm_loss=mlm_loss,
            cls_loss=cls_loss,
        )
