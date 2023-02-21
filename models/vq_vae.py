from abc import ABC
from typing import Optional, Union, Tuple, List
import os
import copy

import torch
from torch import nn
from transformers.models.bart.modeling_bart import BartForConditionalGeneration, BartConfig, Seq2SeqLMOutput, shift_tokens_right
from transformers.models.roberta.modeling_roberta import RobertaModel, RobertaPreTrainedModel, RobertaConfig, \
    RobertaLMHead
from transformers import PreTrainedTokenizer, AutoTokenizer

from general_util.logger import get_child_logger
from general_util.mixin import LogMixin
from models.bart import DVAESeq2SeqLMOutput
from modules import layers

logger = get_child_logger("VQ-VAE")


class RobertaVQVAE(RobertaPreTrainedModel, ABC, LogMixin):
    """
    Prior model `p(r|x, e_i, e_j)`
    Prior model encoder `p(h_r|x, e_i, e_j)`
    Prior model decoder (quantizer): `p(r|h_r)`
    Generator `p(x|r, e_i, e_j)`
    """

    # _keys_to_ignore_on_save = [r"lm_head.decoder.weight", r"lm_head.decoder.bias"]
    # _keys_to_ignore_on_load_missing = [r"position_ids", r"lm_head.decoder.weight", r"lm_head.decoder.bias"]
    # _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config: RobertaConfig, seq2seq_path: str, quantizer: nn.Module, embedding_dim: int, input_type: str = "add",
                 z_to_decoder: bool = False, z_add_top: bool = False):
        super().__init__(config)

        if config.is_decoder:
            logger.warning(
                "If you want to use `RobertaForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.lm_head = RobertaLMHead(config)

        self.seq2seq_config: BartConfig = BartConfig.from_pretrained(seq2seq_path)

        self.dense1 = nn.Linear(config.hidden_size * 2, embedding_dim)
        self.dense2 = nn.Linear(embedding_dim, self.seq2seq_config.d_model)

        # The LM head weights require special treatment only when they are tied with the word embeddings
        self.update_keys_to_ignore(config, ["lm_head.decoder.weight"])

        # Initialize weights and apply final processing
        # self.post_init()
        self.roberta.post_init()
        self._init_weights(self.lm_head)
        self._init_weights(self.dense1)
        self._init_weights(self.dense2)

        self.z_to_decoder = z_to_decoder
        self.z_add_top = z_add_top
        if z_to_decoder:
            self.seq2seq: BartWithLatent = BartWithLatent.from_pretrained(seq2seq_path, z_add_top=z_add_top)
        else:
            self.seq2seq: BartForConditionalGeneration = BartForConditionalGeneration.from_pretrained(seq2seq_path)

        # Initialize after `post_init()`
        self.quantizer = quantizer

        self.init_metric("acc", "loss", "mlm_loss", "commit_loss", "vq_loss")

        self.input_type = input_type

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            token_type_ids: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            decoder_input_ids: Optional[torch.LongTensor] = None,
            decoder_input_attention_mask: Optional[torch.LongTensor] = None,
            decoder_output_ids: Optional[torch.LongTensor] = None,
            h_span: Optional[torch.LongTensor] = None,
            t_span: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple[torch.Tensor], DVAESeq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        kwargs (`Dict[str, any]`, optional, defaults to *{}*):
            Used to hide legacy arguments that have been deprecated.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]

        h_span_hidden = torch.gather(sequence_output, dim=1,
                                     index=h_span.unsqueeze(-1).expand(-1, -1, sequence_output.size(-1))).mean(dim=1)
        t_span_hidden = torch.gather(sequence_output, dim=1,
                                     index=t_span.unsqueeze(-1).expand(-1, -1, sequence_output.size(-1))).mean(dim=1)
        rel_hidden = self.dense1(torch.cat([h_span_hidden, t_span_hidden], dim=-1))
        res = self.quantizer(rel_hidden)
        if len(res) == 4:
            quantized_rel_emb, vq_loss, l2_diff, emb_idx = res
        elif len(res) == 3:
            quantized_rel_emb, l2_diff, emb_idx = res
            vq_loss = None
        else:
            raise RuntimeError()

        seq2seq_input_emb = self.seq2seq.model.encoder.embed_tokens(decoder_input_ids) * self.seq2seq.model.encoder.embed_scale
        z = quantized_rel_emb = self.dense2(quantized_rel_emb)
        labels = decoder_output_ids.masked_fill(decoder_output_ids == self.seq2seq.config.pad_token_id, -100)

        if self.input_type == "add":
            seq2seq_input_emb = seq2seq_input_emb + quantized_rel_emb[:, None, :]
        elif self.input_type[:6] == "concat":
            _code_num = int(self.input_type.split("$")[1])
            quantized_rel_emb = quantized_rel_emb.unsqueeze(1).expand(-1, _code_num, -1)
            seq2seq_input_emb = torch.cat([quantized_rel_emb, seq2seq_input_emb], dim=1)
            decoder_input_attention_mask = torch.cat([torch.ones(quantized_rel_emb.size()[:-1],
                                                                 device=quantized_rel_emb.device, dtype=decoder_input_attention_mask.dtype),
                                                      decoder_input_attention_mask], dim=1)
        else:
            raise NotImplementedError

        if self.z_to_decoder:
            seq2seq_outputs = self.seq2seq(
                attention_mask=decoder_input_attention_mask,
                inputs_embeds=seq2seq_input_emb,
                labels=labels,
                return_dict=return_dict,
                z=z
            )
        else:
            seq2seq_outputs = self.seq2seq(
                attention_mask=decoder_input_attention_mask,
                inputs_embeds=seq2seq_input_emb,
                labels=labels,
                return_dict=return_dict,
            )

        reconstruction_loss = seq2seq_outputs.loss
        seq2seq_logits = seq2seq_outputs.logits

        loss = reconstruction_loss + l2_diff
        if vq_loss is not None:
            loss = loss + vq_loss

        if not self.training:
            acc, true_label_num = layers.get_accuracy(seq2seq_logits, labels, pad_id=-100)
            self.eval_metrics.update("acc", val=acc, n=true_label_num)
            self.eval_metrics.update("loss", val=loss.item(), n=sequence_output.size(0))
            self.eval_metrics.update("commit_loss", val=l2_diff.item(), n=sequence_output.size(0))
            self.eval_metrics.update("mlm_loss", val=reconstruction_loss.item(), n=true_label_num)
            if vq_loss is not None:
                self.eval_metrics.update("vq_loss", val=vq_loss.item(), n=sequence_output.size(0))

        if not return_dict:
            # output = (prediction_scores,) + outputs[2:]
            # return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
            return loss, reconstruction_loss, seq2seq_logits, vq_loss, l2_diff, emb_idx

        return DVAESeq2SeqLMOutput(
            loss=loss,
            logits=seq2seq_logits,
            vq_loss=vq_loss,
            commitment_loss=l2_diff,
            mlm_loss=reconstruction_loss,
            z_encoder_mean=torch.mean(rel_hidden.detach()),
            z_decoder_mean=torch.mean(quantized_rel_emb.detach()),
            code_indices=emb_idx,
        )


class RobertaVQVAEMultiple(RobertaPreTrainedModel, ABC, LogMixin):
    """
    Prior model `p(r|x, e_i, e_j)`
    Prior model encoder `p(h_r|x, e_i, e_j)`
    Prior model decoder (quantizer): `p(r|h_r)`
    Generator `p(x|r, e_i, e_j)`
    """

    # _keys_to_ignore_on_save = [r"lm_head.decoder.weight", r"lm_head.decoder.bias"]
    # _keys_to_ignore_on_load_missing = [r"position_ids", r"lm_head.decoder.weight", r"lm_head.decoder.bias"]
    # _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config: RobertaConfig, seq2seq_path: str, quantizer: nn.Module, embedding_dim: int, input_type: str = "add",
                 freeze_seq2seq_encoder: bool = False, z_to_decoder: bool = False, seq2seq: BartForConditionalGeneration = None):
        super().__init__(config)

        self.freeze_seq2seq_encoder = freeze_seq2seq_encoder

        if config.is_decoder:
            logger.warning(
                "If you want to use `RobertaForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.lm_head = RobertaLMHead(config)

        self.seq2seq_config: BartConfig = BartConfig.from_pretrained(seq2seq_path)
        self.seq2seq_tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(seq2seq_path)

        self.code_embedding_dim = embedding_dim
        self.dense1 = nn.Linear(config.hidden_size * 2, embedding_dim)
        self.dense2 = nn.Linear(embedding_dim, self.seq2seq_config.d_model)

        # The LM head weights require special treatment only when they are tied with the word embeddings
        self.update_keys_to_ignore(config, ["lm_head.decoder.weight"])

        # Initialize weights and apply final processing
        # self.post_init()
        self.roberta.post_init()
        self._init_weights(self.lm_head)
        self._init_weights(self.dense1)
        self._init_weights(self.dense2)

        self.z_to_decoder = z_to_decoder
        # self.z_add_top = z_add_top
        self.seq2seq = seq2seq
        # if z_to_decoder:
        #     self.seq2seq: BartWithLatent = BartWithLatent.from_pretrained(seq2seq_path,
        #                                                                   decoder_layers=seq2seq_decoder_layer,
        #                                                                   z_add_top=z_add_top)
        # else:
        #     self.seq2seq: BartForConditionalGeneration = BartForConditionalGeneration.from_pretrained(seq2seq_path,
        #                                                                                               decoder_layers=seq2seq_decoder_layer,
        #                                                                                               z_add_top=z_add_top)

        if self.freeze_seq2seq_encoder:
            layers.freeze_module(self.seq2seq.model.encoder)
            for p in self.seq2seq.model.shared.parameters():
                p.requires_grad = True

        # Initialize after `post_init()`
        self.quantizer = quantizer

        self.init_metric("acc", "loss", "mlm_loss", "commit_loss", "vq_loss")

        self.input_type = input_type

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    def get_non_sharded_modules(self):
        if self.freeze_seq2seq_encoder:
            return [
                self.quantizer,
                self.dense1,
                self.dense2,
                self.seq2seq.model.encoder,
            ]
        return [
            self.quantizer,
            self.dense1,
            self.dense2,
        ]

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], *model_args, **kwargs):
        kai_full_state = kwargs.pop("kai_full_state", False)
        if kai_full_state:
            return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

        seq2seq = kwargs.pop("seq2seq")
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

        model.seq2seq = seq2seq
        return model

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            token_type_ids: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            decoder_input_ids: Optional[torch.LongTensor] = None,
            decoder_input_attention_mask: Optional[torch.LongTensor] = None,
            decoder_output_ids: Optional[torch.LongTensor] = None,
            entity_spans: Optional[torch.FloatTensor] = None,
            rel_ent_index: Optional[torch.LongTensor] = None,
            rel_emb_index: Optional[torch.LongTensor] = None,
            **kwargs
    ) -> Union[Tuple[torch.Tensor], DVAESeq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        kwargs (`Dict[str, any]`, optional, defaults to *{}*):
            Used to hide legacy arguments that have been deprecated.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]

        entity_hidden = torch.einsum("bsh,bns->bnh", sequence_output, entity_spans)

        batch_size, code_num, _ = rel_ent_index.size()
        rel_ent_index_expand = rel_ent_index.reshape(batch_size, code_num * 2, 1).expand(-1, -1, sequence_output.size(-1))
        rel_hidden = torch.gather(entity_hidden, index=rel_ent_index_expand, dim=1)
        rel_hidden = self.dense1(rel_hidden.reshape(batch_size, code_num, -1)).reshape(batch_size * code_num, -1)

        res = self.quantizer(rel_hidden)
        if len(res) == 4:
            quantized_rel_emb, vq_loss, l2_diff, emb_idx = res
        elif len(res) == 3:
            quantized_rel_emb, l2_diff, emb_idx = res
            vq_loss = None
        else:
            raise RuntimeError()

        quantized_rel_emb = quantized_rel_emb.reshape(batch_size, code_num, self.code_embedding_dim)
        quantized_rel_emb = self.dense2(quantized_rel_emb)

        z = quantized_rel_emb.mean(dim=1)

        rel_emb_index_expand = rel_emb_index.unsqueeze(-1).expand(-1, -1, quantized_rel_emb.size(-1))
        quantized_rel_emb_ex = torch.gather(quantized_rel_emb, index=rel_emb_index_expand, dim=1)  # [batch, decoder_inputs_len, h]

        seq2seq_input_emb = self.seq2seq.model.encoder.embed_tokens(decoder_input_ids) * self.seq2seq.model.encoder.embed_scale
        seq2seq_input_emb = torch.where((decoder_input_ids == self.seq2seq_tokenizer.mask_token_id).unsqueeze(-1).expand(
            -1, -1, quantized_rel_emb.size(-1)),
            quantized_rel_emb_ex, seq2seq_input_emb)
        # rel_mask = decoder_input_ids == self.seq2seq_tokenizer.mask_token_id
        # seq2seq_input_emb[rel_mask, :] = quantized_rel_emb_ex[rel_mask, :]

        labels = decoder_output_ids.masked_fill(decoder_output_ids == self.seq2seq.config.pad_token_id, -100)

        if self.input_type == "add":
            seq2seq_input_emb = seq2seq_input_emb + quantized_rel_emb_ex
        # elif self.input_type[:6] == "concat":
        #     _code_num = int(self.input_type.split("$")[1])
        #     quantized_rel_emb = quantized_rel_emb.unsqueeze(1).expand(-1, _code_num, -1)
        #     seq2seq_input_emb = torch.cat([quantized_rel_emb, seq2seq_input_emb], dim=1)
        #     decoder_input_attention_mask = torch.cat([torch.ones(quantized_rel_emb.size()[:-1],
        #                                                        device=quantized_rel_emb.device, dtype=decoder_input_attention_mask.dtype),
        #                                               decoder_input_attention_mask], dim=1)
        elif self.input_type[:6] == "concat":
            pass
        else:
            raise NotImplementedError

        if self.z_to_decoder:
            seq2seq_outputs = self.seq2seq(
                attention_mask=decoder_input_attention_mask,
                inputs_embeds=seq2seq_input_emb,
                labels=labels,
                return_dict=return_dict,
                z=z
            )
        else:
            seq2seq_outputs = self.seq2seq(
                attention_mask=decoder_input_attention_mask,
                inputs_embeds=seq2seq_input_emb,
                labels=labels,
                return_dict=return_dict,
            )

        reconstruction_loss = seq2seq_outputs.loss
        seq2seq_logits = seq2seq_outputs.logits

        loss = reconstruction_loss + l2_diff
        if vq_loss is not None:
            loss = loss + vq_loss

        if not self.training:
            acc, true_label_num = layers.get_accuracy(seq2seq_logits, labels, pad_id=-100)
            self.eval_metrics.update("acc", val=acc, n=true_label_num)
            self.eval_metrics.update("loss", val=loss.item(), n=sequence_output.size(0))
            self.eval_metrics.update("commit_loss", val=l2_diff.item(), n=sequence_output.size(0))
            self.eval_metrics.update("mlm_loss", val=reconstruction_loss.item(), n=true_label_num)
            if vq_loss is not None:
                self.eval_metrics.update("vq_loss", val=vq_loss.item(), n=sequence_output.size(0))

        if not return_dict:
            # output = (prediction_scores,) + outputs[2:]
            # return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
            return loss, reconstruction_loss, seq2seq_logits, vq_loss, l2_diff, emb_idx

        return DVAESeq2SeqLMOutput(
            loss=loss,
            logits=seq2seq_logits,
            vq_loss=vq_loss,
            commitment_loss=l2_diff,
            mlm_loss=reconstruction_loss,
            z_encoder_mean=torch.mean(rel_hidden.detach()),
            z_decoder_mean=torch.mean(quantized_rel_emb.detach()),
            code_indices=emb_idx,
        )


class BartWithLatent(BartForConditionalGeneration, ABC):
    def __init__(self, config: BartConfig, z_add_top: bool = False):
        super().__init__(config)

        logger.warning("Latent augmented bart initialized.")
        self.z_add_top = z_add_top
        logger.info(f"Latent variable added to output layer: {self.z_add_top}")

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            decoder_input_ids: Optional[torch.LongTensor] = None,
            decoder_attention_mask: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            decoder_head_mask: Optional[torch.Tensor] = None,
            cross_attn_head_mask: Optional[torch.Tensor] = None,
            encoder_outputs: Optional[List[torch.FloatTensor]] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            z: Optional[torch.FloatTensor] = None,
    ) -> Union[Tuple, Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if decoder_inputs_embeds is not None:
            logger.warning("The `decoder_input_embeds` here maybe override. Please carefully check if it really plays an effect.")

        if labels is not None:
            if use_cache:
                logger.warning("The `use_cache` argument is changed to `False` since `labels` is provided.")
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        if z is not None:
            decoder_input_shape = decoder_input_ids.size()
            decoder_input_ids = decoder_input_ids.view(-1, decoder_input_shape[-1])
            decoder_inputs_embeds = self.model.decoder.embed_tokens(decoder_input_ids) * self.model.decoder.embed_scale
            decoder_inputs_embeds = decoder_inputs_embeds + z[:, None, :]
            decoder_input_ids = None

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        if z is not None and self.z_add_top:
            lm_logits = self.lm_head(outputs[0] + z[:, None, :]) + self.final_logits_bias
        else:
            lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias

        masked_lm_loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )
