from abc import ABC
from typing import Optional, Union, Tuple

import torch
from torch import nn
from transformers.models.bart.modeling_bart import BartForConditionalGeneration
from transformers.models.roberta.modeling_roberta import RobertaModel, RobertaPreTrainedModel, RobertaConfig, \
    RobertaLMHead

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

    def __init__(self, config: RobertaConfig, seq2seq_path: str, quantizer: nn.Module, embedding_dim: int):
        super().__init__(config)

        if config.is_decoder:
            logger.warning(
                "If you want to use `RobertaForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.lm_head = RobertaLMHead(config)

        self.seq2seq: BartForConditionalGeneration = BartForConditionalGeneration.from_pretrained(seq2seq_path)

        self.dense1 = nn.Linear(config.hidden_size * 2, embedding_dim)
        self.dense2 = nn.Linear(embedding_dim, self.seq2seq.config.d_model)

        # The LM head weights require special treatment only when they are tied with the word embeddings
        self.update_keys_to_ignore(config, ["lm_head.decoder.weight"])

        # Initialize weights and apply final processing
        self.post_init()

        # Initialize after `post_init()`
        self.quantizer = quantizer

        self.init_metric("acc", "loss", "mlm_loss", "commit_loss")

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
        quantized_rel_emb, l2_diff, emb_idx = self.quantizer(rel_hidden)

        seq2seq_input_emb = self.seq2seq.model.encoder.embed_tokens(decoder_input_ids) * self.seq2seq.model.encoder.embed_scale
        seq2seq_input_emb = seq2seq_input_emb + self.dense2(quantized_rel_emb).unsqueeze(1).expand(-1, decoder_input_ids.size(1), -1)
        labels = decoder_output_ids.masked_fill(decoder_output_ids == self.seq2seq.config.pad_token_id, -100)
        seq2seq_outputs = self.seq2seq(
            attention_mask=decoder_input_attention_mask,
            inputs_embeds=seq2seq_input_emb,
            labels=labels,
            return_dict=return_dict,
        )

        reconstruction_loss = seq2seq_outputs.loss
        seq2seq_logits = seq2seq_outputs.logits

        loss = reconstruction_loss + l2_diff

        if not self.training:
            acc, true_label_num = layers.get_accuracy(seq2seq_logits, labels, pad_id=-100)
            self.eval_metrics.update("acc", val=acc, n=true_label_num)
            self.eval_metrics.update("loss", val=loss.item(), n=sequence_output.size(0))
            self.eval_metrics.update("commit_loss", val=l2_diff.item(), n=sequence_output.size(0))
            self.eval_metrics.update("mlm_loss", val=reconstruction_loss.item(), n=true_label_num)

        if not return_dict:
            # output = (prediction_scores,) + outputs[2:]
            # return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
            return loss, reconstruction_loss, seq2seq_logits, l2_diff, emb_idx

        return DVAESeq2SeqLMOutput(
            loss=loss,
            logits=seq2seq_logits,
            commitment_loss=l2_diff,
            mlm_loss=reconstruction_loss,
            z_encoder_mean=torch.mean(rel_hidden.detach()),
            z_decoder_mean=torch.mean(quantized_rel_emb.detach()),
            code_indices=emb_idx,
        )
