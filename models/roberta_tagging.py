import copy
import pickle
from abc import ABC
from dataclasses import dataclass
from typing import Union, Dict, Optional
import os

import torch
import transformers
from torch import nn, Tensor
from torch.nn import CrossEntropyLoss
from transformers.models.roberta.modeling_roberta import RobertaConfig
from transformers import AutoTokenizer

from general_util.logger import get_child_logger
from models.roberta import RobertaForMultipleChoiceForPreTrain, RelDecoderHead, MultipleChoicePreTrainModelOutput, attentive_pooling, \
    RobertaModel, RobertaLMHead
from modules import layers
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel, GPT2Config, GPT2Model
from models.gpt2 import GPT2ForConditionalGeneration

logger = get_child_logger("RoBERTa.Tagger")


@dataclass
class TaggingOutputClass(MultipleChoicePreTrainModelOutput):
    tagging_loss: torch.FloatTensor = None
    tagging_acc: torch.FloatTensor = None
    tagging_precision: torch.FloatTensor = None
    tagging_recall: torch.FloatTensor = None
    tagging_logits: torch.FloatTensor = None
    s_path_gen_loss: torch.FloatTensor = None
    rmlm_loss: torch.FloatTensor = None
    me_loss: torch.FloatTensor = None


class RobertaForMultipleChoicePreTrainTaggerV1(RobertaForMultipleChoiceForPreTrain, ABC):
    def __init__(self, config: RobertaConfig,
                 rel_vocab: str = None,
                 rel_vocab_size: int = None,
                 mlp_hidden_size: int = 768,
                 mlm_alpha: float = 1.0,
                 mlm_disabled: bool = False,
                 fs_checkpoint: bool = False,
                 fs_checkpoint_offload_to_cpu: bool = False,
                 fs_checkpoint_start_layer_id: int = 0,
                 rel_emb_weights: Union[str, torch.Tensor] = None,
                 rel_gen_coff: float = 1.0,
                 tagging_coff: float = 1.0,
                 token_prediction_layer: int = -1):
        super().__init__(config, mlp_hidden_size, mlm_alpha, mlm_disabled,
                         fs_checkpoint, fs_checkpoint_offload_to_cpu, fs_checkpoint_start_layer_id)

        if rel_vocab is not None:
            self.rel_vocab = pickle.load(open(rel_vocab, "rb"))
            self.rel_vocab_size = len(set(list(self.rel_vocab.values())))
            self.eos_token_id = self.rel_vocab_size
            self.pad_token_id = self.rel_vocab_size + 1
            self.unk_token_id = -2
            for token, token_id in self.rel_vocab.items():
                if token == "<unk>":
                    self.unk_token_id = token_id
                    break
                assert token_id < self.rel_vocab_size
        elif rel_vocab_size is not None:
            self.rel_vocab_size = rel_vocab_size
            self.eos_token_id = self.rel_vocab_size
            self.pad_token_id = self.rel_vocab_size + 1
            self.unk_token_id = -2
        else:
            raise RuntimeError()

        self.tagger = nn.Linear(config.hidden_size, 2)
        self._init_weights(self.tagger)

        rel_decoding_config = copy.deepcopy(config)
        rel_decoding_config.vocab_size = self.rel_vocab_size
        print(self.rel_vocab_size)
        if rel_emb_weights is not None:
            if isinstance(rel_emb_weights, str):
                # item_hidden_states, cluster_ids_x, cluster_centers
                _, _, rel_emb_weights = torch.load(rel_emb_weights, map_location="cpu")
            elif isinstance(rel_emb_weights, torch.Tensor):
                rel_emb_weights = rel_emb_weights.transpose(0, 1)
            else:
                raise RuntimeError(type(rel_emb_weights))
            rel_emb_weights = rel_emb_weights.to(dtype=self.lm_head.decoder.weight.data.dtype)
            print(rel_emb_weights.size())
            self.rel_decoder = RelDecoderHead(config.hidden_size * 2, rel_emb_weights.size(1), self.rel_vocab_size)

            self._init_weights(self.rel_decoder)

            self.rel_decoder.decoder.weight.data.copy_(rel_emb_weights)
            assert self.rel_decoder.decoder.weight.requires_grad
        else:
            self.rel_decoder = RelDecoderHead(config.hidden_size * 2, config.hidden_size, self.rel_vocab_size)
            self._init_weights(self.rel_decoder)

        self.rel_gen_coff = rel_gen_coff
        self.tagging_coff = tagging_coff
        self.token_prediction_layer = token_prediction_layer
        logger.info(f"{self.rel_gen_coff}\t{self.tagging_coff}\t{self.token_prediction_layer}")

        self.init_metric("loss", "acc", "mlm_loss", "mlm_acc", "cls_loss", "path_gen_acc", "path_gen_loss",
                         "tagging_loss", "tagging_acc", "tagging_p", "tagging_r")

    def forward(
            self,
            input_ids: Tensor,
            attention_mask: Tensor = None,
            token_type_ids: Tensor = None,
            labels: Tensor = None,
            h_span_marks: Tensor = None,
            t_span_marks: Tensor = None,
            entity_pair_mask: Tensor = None,
            tagging_labels: Tensor = None,
            mlm_input_ids: Tensor = None,
            mlm_attention_mask: Tensor = None,
            rel_labels: Tensor = None,
            mlm_labels: Tensor = None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            **kwargs
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the multiple choice classification loss. Indices should be in ``[0, ...,
            num_choices-1]`` where :obj:`num_choices` is the size of the second dimension of the input tensors. (See
            :obj:`input_ids` above)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if self.token_prediction_layer != -1:
            output_hidden_states = True
        batch_size, num_choices, seq_len = input_ids.size()

        input_ids = self.fold_tensor(input_ids)
        attention_mask = self.fold_tensor(attention_mask)
        token_type_ids = self.fold_tensor(token_type_ids)

        outputs = self.roberta(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        pooled_output = outputs[0][:, 0]

        logits = self.cls(self.dropout(self.pooler(pooled_output)))
        reshaped_logits = logits.view(-1, num_choices)

        if self.token_prediction_layer == -1:
            seq_hidden = self.dropout(outputs[0].reshape(batch_size, num_choices, seq_len, -1)[:, 0])
        else:
            seq_hidden = outputs["hidden_states"][self.token_prediction_layer].reshape(batch_size, num_choices, seq_len, -1)
            seq_hidden = self.dropout(seq_hidden[:, 0])

        tagging_logits = self.tagger(seq_hidden)

        h_hidden = torch.einsum("bst,bth->bsh", h_span_marks, seq_hidden)
        t_hidden = torch.einsum("bst,bth->bsh", t_span_marks, seq_hidden)
        rel_inputs = torch.cat([h_hidden, t_hidden], dim=-1)
        sent_rel_logits = self.rel_decoder(rel_inputs)

        loss = 0.
        mlm_loss = 0.
        cls_loss = 0.
        path_gen_loss = 0.
        tagging_loss = 0.
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            cls_loss = loss_fct(reshaped_logits, labels)
            loss = loss + cls_loss

            if mlm_labels is not None and (self.mlm_disabled is False or self.training is False):
                if mlm_attention_mask is None:
                    mlm_attention_mask = attention_mask.reshape(reshaped_logits.size(0), num_choices, -1)[:, 0]

                mlm_outputs = self.roberta(
                    mlm_input_ids,
                    attention_mask=mlm_attention_mask,
                    return_dict=return_dict
                )

                mlm_scores = self.lm_head(mlm_outputs[0])
                mlm_loss = self.mlm_alpha * loss_fct(mlm_scores.reshape(-1, self.vocab_size), mlm_labels.reshape(-1))
                if not self.mlm_disabled:
                    loss = loss + mlm_loss
            else:
                mlm_scores = None
                mlm_loss = None

            rel_labels[rel_labels == self.unk_token_id] = -1
            rel_labels = rel_labels[:, :-1]  # remove <bos> token
            rel_labels[rel_labels == self.eos_token_id] = -1
            rel_labels[entity_pair_mask] = -1

            path_gen_loss = self.rel_gen_coff * loss_fct(sent_rel_logits.reshape(-1, self.rel_vocab_size), rel_labels.reshape(-1))
            _, true_label_num = layers.get_accuracy(sent_rel_logits, rel_labels)
            if true_label_num:
                loss = loss + path_gen_loss
            else:
                path_gen_loss = 0.

            tagging_loss = self.tagging_coff * loss_fct(tagging_logits.reshape(-1, 2), tagging_labels.view(-1))
            loss = loss + tagging_loss

            if not self.training:
                acc, true_label_num = layers.get_accuracy(reshaped_logits, labels)
                self.eval_metrics.update("acc", val=acc, n=true_label_num)
                self.eval_metrics.update("loss", val=loss.item(), n=true_label_num)
                self.eval_metrics.update("cls_loss", val=cls_loss.item(), n=true_label_num)

                acc, true_label_num = layers.get_accuracy(sent_rel_logits, rel_labels)
                if true_label_num:
                    self.eval_metrics.update("path_gen_acc", val=acc, n=true_label_num)
                    self.eval_metrics.update("path_gen_loss", val=path_gen_loss, n=true_label_num)

                acc, true_label_num = layers.get_accuracy(tagging_logits, tagging_labels)
                self.eval_metrics.update("tagging_loss", val=tagging_loss, n=true_label_num)
                self.eval_metrics.update("tagging_acc", val=acc, n=true_label_num)

                precision, recall, batch_size = layers.get_precision_recall(tagging_logits, tagging_labels)
                self.eval_metrics.update("tagging_p", val=precision, n=batch_size)
                self.eval_metrics.update("tagging_r", val=recall, n=batch_size)

                if mlm_labels is not None:
                    acc, true_label_num = layers.get_accuracy(mlm_scores, mlm_labels)
                    self.eval_metrics.update("mlm_acc", val=acc, n=true_label_num)
                    self.eval_metrics.update("mlm_loss", val=mlm_loss.item(), n=true_label_num)

        if not return_dict:
            output = (reshaped_logits,) + outputs[2:] + (mlm_loss, cls_loss,)
            return ((loss,) + output) if loss is not None else output

        return TaggingOutputClass(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            mlm_loss=mlm_loss,
            cls_loss=cls_loss,
            path_gen_loss=path_gen_loss,
            tagging_loss=tagging_loss,
            tagging_logits=tagging_logits,
        )


class RobertaForMultipleChoicePreTrainTaggerV1PredictorV1(RobertaForMultipleChoicePreTrainTaggerV1, ABC):
    def forward(
            self,
            input_ids: Tensor,
            attention_mask: Tensor = None,
            token_type_ids: Tensor = None,
            labels: Tensor = None,
            h_span_marks: Tensor = None,
            t_span_marks: Tensor = None,
            entity_pair_mask: Tensor = None,
            tagging_labels: Tensor = None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            **kwargs
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the multiple choice classification loss. Indices should be in ``[0, ...,
            num_choices-1]`` where :obj:`num_choices` is the size of the second dimension of the input tensors. (See
            :obj:`input_ids` above)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if self.token_prediction_layer != -1:
            output_hidden_states = True
        batch_size, num_choices, seq_len = input_ids.size()

        input_ids = self.fold_tensor(input_ids)
        attention_mask = self.fold_tensor(attention_mask)
        token_type_ids = self.fold_tensor(token_type_ids)

        outputs = self.roberta(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        pooled_output = outputs[0][:, 0]

        logits = self.cls(self.dropout(self.pooler(pooled_output)))
        reshaped_logits = logits.view(-1, num_choices)

        if self.token_prediction_layer == -1:
            seq_hidden = self.dropout(outputs[0].reshape(batch_size, num_choices, seq_len, -1))
        else:
            seq_hidden = outputs["hidden_states"][self.token_prediction_layer].reshape(batch_size, num_choices, seq_len, -1)
            seq_hidden = self.dropout(seq_hidden)
        # [batch, num_choices, seq_len, 2]
        tagging_logits = self.tagger(seq_hidden)

        loss = 0.
        mlm_loss = 0.
        cls_loss = 0.
        path_gen_loss = 0.
        tagging_loss = 0.
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            cls_loss = loss_fct(reshaped_logits, labels)
            loss = loss + cls_loss

            if not self.training:
                acc, true_label_num = layers.get_accuracy(reshaped_logits, labels)
                self.eval_metrics.update("acc", val=acc, n=true_label_num)
                self.eval_metrics.update("loss", val=loss.item(), n=true_label_num)
                self.eval_metrics.update("cls_loss", val=cls_loss.item(), n=true_label_num)

        if not return_dict:
            output = (reshaped_logits,) + outputs[2:] + (mlm_loss, cls_loss,)
            return ((loss,) + output) if loss is not None else output

        return TaggingOutputClass(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cls_loss=cls_loss,
            tagging_logits=tagging_logits,
        )


def extract_seq_hidden(tf_outputs: Dict, layer_idx: int):
    return tf_outputs["hidden_states"][layer_idx]


class RobertaForMultipleChoicePreTrainTaggerV2(RobertaForMultipleChoicePreTrainTaggerV1, ABC):
    def __init__(self, config: RobertaConfig,
                 rel_vocab: str = None,
                 rel_vocab_size: int = None,
                 mlp_hidden_size: int = 768,
                 mlm_alpha: float = 1.0,
                 mlm_disabled: bool = False,
                 fs_checkpoint: bool = False,
                 fs_checkpoint_offload_to_cpu: bool = False,
                 fs_checkpoint_start_layer_id: int = 0,
                 rel_emb_weights: Union[str, torch.Tensor] = None,
                 rel_gen_coff: float = 1.0,
                 tagging_coff: float = 1.0,
                 s_rel_gen_coff: float = 1.0,
                 cls_coff: float = 1.0,
                 token_prediction_layer: int = -1,
                 prior_prediction_layer: int = -1):
        super().__init__(config, rel_vocab, rel_vocab_size, mlp_hidden_size, mlm_alpha, mlm_disabled,
                         fs_checkpoint, fs_checkpoint_offload_to_cpu, fs_checkpoint_start_layer_id,
                         rel_emb_weights, rel_gen_coff, tagging_coff, token_prediction_layer)

        self.cls_coff = cls_coff
        self.s_rel_gen_coff = s_rel_gen_coff
        self.prior_prediction_layer = prior_prediction_layer

        self.sent_attn_pooler = nn.Linear(config.hidden_size, 1)
        self._init_weights(self.sent_attn_pooler)

        self.s_rel_decoder = copy.deepcopy(self.rel_decoder)
        self.s_rel_decoder.dense = nn.Linear(config.hidden_size, self.rel_decoder.dense.out_features)
        self._init_weights(self.s_rel_decoder.dense)
        self.s_rel_decoder.decoder.weight = self.rel_decoder.decoder.weight

        self.init_metric("loss", "acc", "mlm_loss", "mlm_acc", "cls_loss", "path_gen_acc", "path_gen_loss",
                         "tagging_loss", "tagging_acc", "tagging_p", "tagging_r",
                         "s_path_gen_acc", "s_path_gen_loss")

    def forward(
            self,
            input_ids: Tensor,
            attention_mask: Tensor = None,
            token_type_ids: Tensor = None,
            labels: Tensor = None,
            sent_token_index: Tensor = None,
            sent_token_mask: Tensor = None,
            h_span_marks: Tensor = None,
            t_span_marks: Tensor = None,
            entity_pair_mask: Tensor = None,
            tagging_labels: Tensor = None,
            mlm_input_ids: Tensor = None,
            mlm_attention_mask: Tensor = None,
            rel_labels: Tensor = None,
            mlm_labels: Tensor = None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            **kwargs
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the multiple choice classification loss. Indices should be in ``[0, ...,
            num_choices-1]`` where :obj:`num_choices` is the size of the second dimension of the input tensors. (See
            :obj:`input_ids` above)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = True
        batch_size, num_choices, seq_len = input_ids.size()

        input_ids = self.fold_tensor(input_ids)
        attention_mask = self.fold_tensor(attention_mask)
        token_type_ids = self.fold_tensor(token_type_ids)

        outputs = self.roberta(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        pooled_output = outputs[0][:, 0]

        logits = self.cls(self.dropout(self.pooler(pooled_output)))
        reshaped_logits = logits.view(-1, num_choices)

        seq_hidden = extract_seq_hidden(outputs, self.token_prediction_layer).reshape(batch_size, num_choices, seq_len, -1)[:, 0]
        seq_hidden = self.dropout(seq_hidden)

        tagging_logits = self.tagger(seq_hidden)

        h_hidden = torch.einsum("bst,bth->bsh", h_span_marks, seq_hidden)
        t_hidden = torch.einsum("bst,bth->bsh", t_span_marks, seq_hidden)
        rel_inputs = torch.cat([h_hidden, t_hidden], dim=-1)
        sent_rel_logits = self.rel_decoder(rel_inputs)

        if self.prior_prediction_layer != self.token_prediction_layer:
            seq_hidden = extract_seq_hidden(outputs, self.prior_prediction_layer).reshape(batch_size, num_choices, seq_len, -1)[:, 0]
            seq_hidden = self.dropout(seq_hidden)

        s_rel_inputs = attentive_pooling(self.sent_attn_pooler, seq_hidden, sent_token_index, sent_token_mask)
        s_sent_rel_logits = self.s_rel_decoder(s_rel_inputs)

        loss = 0.
        mlm_loss = 0.
        cls_loss = 0.
        path_gen_loss = 0.
        s_path_gen_loss = 0.
        tagging_loss = 0.
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            cls_loss = self.cls_coff * loss_fct(reshaped_logits, labels)
            loss = loss + cls_loss

            if mlm_labels is not None and (self.mlm_disabled is False or self.training is False):
                if mlm_attention_mask is None:
                    mlm_attention_mask = attention_mask.reshape(reshaped_logits.size(0), num_choices, -1)[:, 0]

                mlm_outputs = self.roberta(
                    mlm_input_ids,
                    attention_mask=mlm_attention_mask,
                    return_dict=return_dict
                )

                mlm_scores = self.lm_head(mlm_outputs[0])
                mlm_loss = self.mlm_alpha * loss_fct(mlm_scores.reshape(-1, self.vocab_size), mlm_labels.reshape(-1))
                if not self.mlm_disabled:
                    loss = loss + mlm_loss
            else:
                mlm_scores = None
                mlm_loss = None

            rel_labels[rel_labels == self.unk_token_id] = -1
            rel_labels = rel_labels[:, :-1]  # remove <bos> token
            rel_labels[rel_labels == self.eos_token_id] = -1
            rel_labels[entity_pair_mask] = -1

            path_gen_loss = self.rel_gen_coff * loss_fct(sent_rel_logits.reshape(-1, self.rel_vocab_size), rel_labels.reshape(-1))
            _, true_label_num = layers.get_accuracy(sent_rel_logits, rel_labels)

            s_path_gen_loss = self.s_rel_gen_coff * loss_fct(s_sent_rel_logits.reshape(-1, self.rel_vocab_size), rel_labels.reshape(-1))

            if true_label_num:
                loss = loss + path_gen_loss + s_path_gen_loss
            else:
                path_gen_loss = 0.
                s_path_gen_loss = 0.

            tagging_loss = self.tagging_coff * loss_fct(tagging_logits.reshape(-1, 2), tagging_labels.view(-1))
            loss = loss + tagging_loss

            if not self.training:
                acc, true_label_num = layers.get_accuracy(reshaped_logits, labels)
                self.eval_metrics.update("acc", val=acc, n=true_label_num)
                self.eval_metrics.update("loss", val=loss.item(), n=true_label_num)
                self.eval_metrics.update("cls_loss", val=cls_loss.item(), n=true_label_num)

                acc, true_label_num = layers.get_accuracy(sent_rel_logits, rel_labels)
                if true_label_num:
                    self.eval_metrics.update("path_gen_acc", val=acc, n=true_label_num)
                    self.eval_metrics.update("path_gen_loss", val=path_gen_loss, n=true_label_num)

                acc, true_label_num = layers.get_accuracy(s_sent_rel_logits, rel_labels)
                if true_label_num:
                    self.eval_metrics.update("s_path_gen_acc", val=acc, n=true_label_num)
                    self.eval_metrics.update("s_path_gen_loss", val=s_path_gen_loss, n=true_label_num)

                acc, true_label_num = layers.get_accuracy(tagging_logits, tagging_labels)
                self.eval_metrics.update("tagging_loss", val=tagging_loss, n=true_label_num)
                self.eval_metrics.update("tagging_acc", val=acc, n=true_label_num)

                precision, recall, batch_size = layers.get_precision_recall(tagging_logits, tagging_labels)
                self.eval_metrics.update("tagging_p", val=precision, n=batch_size)
                self.eval_metrics.update("tagging_r", val=recall, n=batch_size)

                if mlm_labels is not None:
                    acc, true_label_num = layers.get_accuracy(mlm_scores, mlm_labels)
                    self.eval_metrics.update("mlm_acc", val=acc, n=true_label_num)
                    self.eval_metrics.update("mlm_loss", val=mlm_loss.item(), n=true_label_num)

        if not return_dict:
            output = (reshaped_logits,) + outputs[2:] + (mlm_loss, cls_loss,)
            return ((loss,) + output) if loss is not None else output

        return TaggingOutputClass(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            mlm_loss=mlm_loss,
            cls_loss=cls_loss,
            path_gen_loss=path_gen_loss,
            s_path_gen_loss=s_path_gen_loss,
            tagging_loss=tagging_loss
        )


class RobertaForMultipleChoicePreTrainTaggerV3(RobertaForMultipleChoicePreTrainTaggerV1, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.init_metric("loss", "acc", "mlm_loss", "mlm_acc", "cls_loss",
                         "tagging_loss", "tagging_acc", "tagging_p", "tagging_r",
                         "rmlm_acc", "rmlm_loss")

    def forward(
            self,
            input_ids: Tensor,
            attention_mask: Tensor = None,
            token_type_ids: Tensor = None,
            labels: Tensor = None,
            h_span_marks: Tensor = None,
            t_span_marks: Tensor = None,
            entity_pair_mask: Tensor = None,
            tagging_labels: Tensor = None,
            mlm_input_ids: Tensor = None,
            mlm_attention_mask: Tensor = None,
            rel_labels: Tensor = None,
            mlm_labels: Tensor = None,
            rmlm_input_ids: Tensor = None,
            rmlm_attention_mask: Tensor = None,
            rmlm_labels: Tensor = None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            **kwargs
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the multiple choice classification loss. Indices should be in ``[0, ...,
            num_choices-1]`` where :obj:`num_choices` is the size of the second dimension of the input tensors. (See
            :obj:`input_ids` above)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if self.token_prediction_layer != -1:
            output_hidden_states = True
        batch_size, num_choices, seq_len = input_ids.size()

        input_ids = self.fold_tensor(input_ids)
        attention_mask = self.fold_tensor(attention_mask)
        token_type_ids = self.fold_tensor(token_type_ids)

        outputs = self.roberta(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        pooled_output = outputs[0][:, 0]

        logits = self.cls(self.dropout(self.pooler(pooled_output)))
        reshaped_logits = logits.view(-1, num_choices)

        if self.token_prediction_layer == -1:
            seq_hidden = self.dropout(outputs[0].reshape(batch_size, num_choices, seq_len, -1)[:, 0])
        else:
            seq_hidden = outputs["hidden_states"][self.token_prediction_layer].reshape(batch_size, num_choices, seq_len, -1)
            seq_hidden = self.dropout(seq_hidden[:, 0])

        tagging_logits = self.tagger(seq_hidden)

        rmlm_outputs = self.roberta(
            rmlm_input_ids,
            attention_mask=rmlm_attention_mask,
            return_dict=return_dict
        )
        rmlm_logits = self.lm_head(rmlm_outputs[0])

        loss = 0.
        mlm_loss = 0.
        cls_loss = 0.
        rmlm_loss = 0.
        tagging_loss = 0.
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            cls_loss = loss_fct(reshaped_logits, labels)
            loss = loss + cls_loss

            rmlm_loss = loss_fct(rmlm_logits.reshape(-1, self.vocab_size), rmlm_labels.reshape(-1))
            loss = loss + rmlm_loss

            if mlm_labels is not None and (self.mlm_disabled is False or self.training is False):
                if mlm_attention_mask is None:
                    mlm_attention_mask = attention_mask.reshape(reshaped_logits.size(0), num_choices, -1)[:, 0]

                mlm_outputs = self.roberta(
                    mlm_input_ids,
                    attention_mask=mlm_attention_mask,
                    return_dict=return_dict
                )

                mlm_scores = self.lm_head(mlm_outputs[0])
                mlm_loss = self.mlm_alpha * loss_fct(mlm_scores.reshape(-1, self.vocab_size), mlm_labels.reshape(-1))
                if not self.mlm_disabled:
                    loss = loss + mlm_loss
            else:
                mlm_scores = None
                mlm_loss = None

            tagging_loss = self.tagging_coff * loss_fct(tagging_logits.reshape(-1, 2), tagging_labels.view(-1))
            loss = loss + tagging_loss

            if not self.training:
                acc, true_label_num = layers.get_accuracy(reshaped_logits, labels)
                self.eval_metrics.update("acc", val=acc, n=true_label_num)
                self.eval_metrics.update("loss", val=loss.item(), n=true_label_num)
                self.eval_metrics.update("cls_loss", val=cls_loss.item(), n=true_label_num)

                acc, true_label_num = layers.get_accuracy(tagging_logits, tagging_labels)
                self.eval_metrics.update("tagging_loss", val=tagging_loss, n=true_label_num)
                self.eval_metrics.update("tagging_acc", val=acc, n=true_label_num)

                precision, recall, batch_size = layers.get_precision_recall(tagging_logits, tagging_labels)
                self.eval_metrics.update("tagging_p", val=precision, n=batch_size)
                self.eval_metrics.update("tagging_r", val=recall, n=batch_size)

                acc, true_label_num = layers.get_accuracy(rmlm_logits, rmlm_labels)
                self.eval_metrics.update("rmlm_loss", val=rmlm_loss, n=true_label_num)
                self.eval_metrics.update("rmlm_acc", val=acc, n=true_label_num)

                if mlm_labels is not None:
                    acc, true_label_num = layers.get_accuracy(mlm_scores, mlm_labels)
                    self.eval_metrics.update("mlm_acc", val=acc, n=true_label_num)
                    self.eval_metrics.update("mlm_loss", val=mlm_loss.item(), n=true_label_num)

        if not return_dict:
            output = (reshaped_logits,) + outputs[2:] + (mlm_loss, cls_loss,)
            return ((loss,) + output) if loss is not None else output

        return TaggingOutputClass(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            mlm_loss=mlm_loss,
            cls_loss=cls_loss,
            tagging_loss=tagging_loss,
            tagging_logits=tagging_logits,
            rmlm_loss=rmlm_loss,
        )


class RobertaForMultipleChoicePreTrainTaggerMEV1(RobertaForMultipleChoicePreTrainTaggerV1, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.me_pooler = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.hidden_size),
            nn.GELU(),
            nn.Dropout(p=getattr(self.config, "pooler_dropout", self.config.hidden_dropout_prob)),
            nn.Linear(self.config.hidden_size, self.config.hidden_size),
        )

        self.tokenizer = AutoTokenizer.from_pretrained("roberta-large")

        self.init_metric("loss", "acc", "mlm_loss", "mlm_acc", "cls_loss", "path_gen_acc", "path_gen_loss",
                         "tagging_loss", "tagging_acc", "tagging_p", "tagging_r",
                         "me_acc", "me_loss")

    def forward(
            self,
            input_ids: Tensor,
            attention_mask: Tensor = None,
            token_type_ids: Tensor = None,
            labels: Tensor = None,
            h_span_marks: Tensor = None,
            t_span_marks: Tensor = None,
            entity_pair_mask: Tensor = None,
            tagging_labels: Tensor = None,
            mlm_input_ids: Tensor = None,
            mlm_attention_mask: Tensor = None,
            rel_labels: Tensor = None,
            mlm_labels: Tensor = None,
            me_input_ids: Tensor = None,
            me_attention_mask: Tensor = None,
            me_ent_input_ids: Tensor = None,
            me_ent_mask: Tensor = None,
            me_labels: Tensor = None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            **kwargs
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the multiple choice classification loss. Indices should be in ``[0, ...,
            num_choices-1]`` where :obj:`num_choices` is the size of the second dimension of the input tensors. (See
            :obj:`input_ids` above)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if self.token_prediction_layer != -1:
            output_hidden_states = True
        batch_size, num_choices, seq_len = input_ids.size()

        input_ids = self.fold_tensor(input_ids)
        attention_mask = self.fold_tensor(attention_mask)
        token_type_ids = self.fold_tensor(token_type_ids)

        outputs = self.roberta(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        pooled_output = outputs[0][:, 0]

        logits = self.cls(self.dropout(self.pooler(pooled_output)))
        reshaped_logits = logits.view(-1, num_choices)

        if self.token_prediction_layer == -1:
            seq_hidden = self.dropout(outputs[0].reshape(batch_size, num_choices, seq_len, -1)[:, 0])
        else:
            seq_hidden = outputs["hidden_states"][self.token_prediction_layer].reshape(batch_size, num_choices, seq_len, -1)
            seq_hidden = self.dropout(seq_hidden[:, 0])

        tagging_logits = self.tagger(seq_hidden)

        h_hidden = torch.einsum("bst,bth->bsh", h_span_marks, seq_hidden)
        t_hidden = torch.einsum("bst,bth->bsh", t_span_marks, seq_hidden)
        rel_inputs = torch.cat([h_hidden, t_hidden], dim=-1)
        sent_rel_logits = self.rel_decoder(rel_inputs)

        me_outputs = self.roberta(me_input_ids, attention_mask=me_attention_mask, return_dict=return_dict)[0]
        me_mask = me_input_ids == self.tokenizer.mask_token_id
        me_states = me_outputs[me_mask, :].reshape(me_outputs.size(0), me_outputs.size(-1))

        _, ent_num, ent_len = me_ent_input_ids.size()

        ent_emb_mask = me_ent_input_ids.eq(self.tokenizer.pad_token_id)
        ent_emb_cnt = (~ent_emb_mask).sum(dim=2)
        ent_emb_cnt[ent_emb_cnt == 0] = 1.

        ent_emb = self.roberta.get_input_embeddings()(me_ent_input_ids)
        ent_emb = ent_emb.sum(dim=2) / ent_emb_cnt.unsqueeze(-1)

        me_logits = torch.einsum("bh,beh->be", me_states, ent_emb)
        me_logits = me_logits + (1 - me_ent_mask) * -10000.0

        loss = 0.
        mlm_loss = 0.
        cls_loss = 0.
        path_gen_loss = 0.
        tagging_loss = 0.
        me_loss = 0.
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            cls_loss = loss_fct(reshaped_logits, labels)
            loss = loss + cls_loss

            if mlm_labels is not None and (self.mlm_disabled is False or self.training is False):
                if mlm_attention_mask is None:
                    mlm_attention_mask = attention_mask.reshape(reshaped_logits.size(0), num_choices, -1)[:, 0]

                mlm_outputs = self.roberta(
                    mlm_input_ids,
                    attention_mask=mlm_attention_mask,
                    return_dict=return_dict
                )

                mlm_scores = self.lm_head(mlm_outputs[0])
                mlm_loss = self.mlm_alpha * loss_fct(mlm_scores.reshape(-1, self.vocab_size), mlm_labels.reshape(-1))
                if not self.mlm_disabled:
                    loss = loss + mlm_loss
            else:
                mlm_scores = None
                mlm_loss = None

            rel_labels[rel_labels == self.unk_token_id] = -1
            rel_labels = rel_labels[:, :-1]  # remove <bos> token
            rel_labels[rel_labels == self.eos_token_id] = -1
            rel_labels[entity_pair_mask] = -1

            path_gen_loss = self.rel_gen_coff * loss_fct(sent_rel_logits.reshape(-1, self.rel_vocab_size), rel_labels.reshape(-1))
            _, true_label_num = layers.get_accuracy(sent_rel_logits, rel_labels)
            if true_label_num:
                loss = loss + path_gen_loss
            else:
                path_gen_loss = 0.

            tagging_loss = self.tagging_coff * loss_fct(tagging_logits.reshape(-1, 2), tagging_labels.view(-1))
            loss = loss + tagging_loss

            me_loss = loss_fct(me_logits, me_labels)
            loss = loss + me_loss

            if not self.training:
                acc, true_label_num = layers.get_accuracy(reshaped_logits, labels)
                self.eval_metrics.update("acc", val=acc, n=true_label_num)
                self.eval_metrics.update("loss", val=loss.item(), n=true_label_num)
                self.eval_metrics.update("cls_loss", val=cls_loss.item(), n=true_label_num)

                acc, true_label_num = layers.get_accuracy(sent_rel_logits, rel_labels)
                if true_label_num:
                    self.eval_metrics.update("path_gen_acc", val=acc, n=true_label_num)
                    self.eval_metrics.update("path_gen_loss", val=path_gen_loss, n=true_label_num)

                acc, true_label_num = layers.get_accuracy(tagging_logits, tagging_labels)
                self.eval_metrics.update("tagging_loss", val=tagging_loss, n=true_label_num)
                self.eval_metrics.update("tagging_acc", val=acc, n=true_label_num)

                precision, recall, batch_size = layers.get_precision_recall(tagging_logits, tagging_labels)
                self.eval_metrics.update("tagging_p", val=precision, n=batch_size)
                self.eval_metrics.update("tagging_r", val=recall, n=batch_size)

                acc, true_label_num = layers.get_accuracy(me_logits, me_labels)
                self.eval_metrics.update("me_acc", val=acc, n=true_label_num)
                self.eval_metrics.update("me_loss", val=me_loss, n=true_label_num)

                if mlm_labels is not None:
                    acc, true_label_num = layers.get_accuracy(mlm_scores, mlm_labels)
                    self.eval_metrics.update("mlm_acc", val=acc, n=true_label_num)
                    self.eval_metrics.update("mlm_loss", val=mlm_loss.item(), n=true_label_num)

        if not return_dict:
            output = (reshaped_logits,) + outputs[2:] + (mlm_loss, cls_loss,)
            return ((loss,) + output) if loss is not None else output

        return TaggingOutputClass(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            mlm_loss=mlm_loss,
            cls_loss=cls_loss,
            path_gen_loss=path_gen_loss,
            tagging_loss=tagging_loss,
            tagging_logits=tagging_logits,
            me_loss=me_loss,
        )


class RobertaGPTForReconstruction(RobertaForMultipleChoiceForPreTrain, ABC):
    def __init__(self, config: RobertaConfig,
                 mlp_hidden_size: int = 768,
                 mlm_alpha: float = 1.0,
                 mlm_disabled: bool = False,
                 decoder: GPT2LMHeadModel = None):
        super().__init__(config, mlp_hidden_size, mlm_alpha, mlm_disabled)

        self.roberta = RobertaModel(config)
        self.lm_head = RobertaLMHead(config)
        self.dropout = nn.Dropout(p=getattr(config, "pooler_dropout", config.hidden_dropout_prob))
        self.vocab_size = config.vocab_size

        self.init_weights()

        self.gpt2: GPT2LMHeadModel = decoder
        self.decoder_config = self.gpt2.config

        self.decoder_proj = nn.Linear(config.hidden_size, self.decoder_config.n_embd)
        self._init_weights(self.decoder_proj)

        self.init_metric("loss", "reconstruct_acc", "reconstruct_loss", "mlm_loss", "mlm_acc", "")

        self.mlm_alpha = mlm_alpha
        # The option is added to disable the MLM loss but keep computing the MLM accuracy on the validation set,
        # in order to observe if there is the catastrophic forgetting problem.
        self.mlm_disabled = mlm_disabled

    def forward(
            self,
            input_ids: Tensor,
            attention_mask: Tensor = None,
            token_type_ids: Tensor = None,
            mlm_input_ids: Tensor = None,
            mlm_attention_mask: Tensor = None,
            mlm_labels: Tensor = None,
            decoder_input_ids: Tensor = None,
            decoder_attention_mask: Tensor = None,
            decoder_labels: Tensor = None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            **kwargs
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the multiple choice classification loss. Indices should be in ``[0, ...,
            num_choices-1]`` where :obj:`num_choices` is the size of the second dimension of the input tensors. (See
            :obj:`input_ids` above)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        batch_size, seq_len = input_ids.size()

        outputs = self.roberta(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        pooled_output = outputs[0][:, :1]
        rel_hidden_states = self.decoder_proj(pooled_output)

        decoder_outputs = self.gpt2(input_ids=decoder_input_ids,
                                    attention_mask=decoder_attention_mask,
                                    labels=decoder_labels,
                                    z_hidden_states=rel_hidden_states,
                                    return_dict=True)

        reconstruct_loss = decoder_outputs["loss"]
        reconstruct_logits = decoder_outputs["logits"][:, :-1].contiguous()

        loss = reconstruct_loss
        mlm_loss = 0.
        if decoder_labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)

            if mlm_labels is not None and (self.mlm_disabled is False or self.training is False):
                mlm_outputs = self.roberta(
                    mlm_input_ids,
                    attention_mask=mlm_attention_mask,
                    return_dict=return_dict
                )

                mlm_scores = self.lm_head(mlm_outputs[0])
                mlm_loss = self.mlm_alpha * loss_fct(mlm_scores.reshape(-1, self.vocab_size), mlm_labels.reshape(-1))
                if not self.mlm_disabled:
                    loss = loss + mlm_loss
            else:
                mlm_scores = None
                mlm_loss = None

            if not self.training:
                masked_decoder_labels = decoder_labels.masked_fill(decoder_labels == self.decoder_config.eos_token_id, -1)
                acc, true_label_num = layers.get_accuracy(reconstruct_logits, masked_decoder_labels)
                self.eval_metrics.update("reconstruct_acc", val=acc, n=true_label_num)
                self.eval_metrics.update("loss", val=loss.item(), n=true_label_num)
                self.eval_metrics.update("reconstruct_loss", val=reconstruct_loss.item(), n=true_label_num)

                if mlm_labels is not None:
                    acc, true_label_num = layers.get_accuracy(mlm_scores, mlm_labels)
                    self.eval_metrics.update("mlm_acc", val=acc, n=true_label_num)
                    self.eval_metrics.update("mlm_loss", val=mlm_loss.item(), n=true_label_num)

        if not return_dict:
            output = (reconstruct_logits,) + outputs[2:] + (mlm_loss, reconstruct_loss,)
            return ((loss,) + output) if loss is not None else output

        return TaggingOutputClass(
            loss=loss,
            logits=reconstruct_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            mlm_loss=mlm_loss,
        )


class RobertaGPTForReconstructionMCQA(RobertaGPTForReconstruction, ABC):
    def __init__(self, config: RobertaConfig,
                 mlp_hidden_size: int = 768,
                 mlm_alpha: float = 1.0,
                 mlm_disabled: bool = False,
                 decoder: GPT2LMHeadModel = None):
        super().__init__(config, mlp_hidden_size, mlm_alpha, mlm_disabled, decoder)

        self.init_metric("loss", "reconstruct_acc", "reconstruct_loss", "mlm_loss", "mlm_acc", "acc", "cls_loss")

    def forward(
            self,
            input_ids: Tensor,
            attention_mask: Tensor = None,
            token_type_ids: Tensor = None,
            labels: Tensor = None,
            mlm_input_ids: Tensor = None,
            mlm_attention_mask: Tensor = None,
            mlm_labels: Tensor = None,
            decoder_input_ids: Tensor = None,
            decoder_attention_mask: Tensor = None,
            decoder_labels: Tensor = None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            **kwargs
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the multiple choice classification loss. Indices should be in ``[0, ...,
            num_choices-1]`` where :obj:`num_choices` is the size of the second dimension of the input tensors. (See
            :obj:`input_ids` above)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        batch_size, num_choices, seq_len = input_ids.size()

        input_ids = self.fold_tensor(input_ids)
        attention_mask = self.fold_tensor(attention_mask)
        token_type_ids = self.fold_tensor(token_type_ids)

        outputs = self.roberta(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        pooled_output = outputs[0][:, 0]
        logits = self.cls(self.dropout(self.pooler(pooled_output)))
        reshaped_logits = logits.view(-1, num_choices)

        rel_pooled_output = pooled_output.reshape(batch_size, num_choices, -1)[:, :1]
        rel_hidden_states = self.decoder_proj(rel_pooled_output)

        decoder_outputs = self.gpt2(input_ids=decoder_input_ids,
                                    attention_mask=decoder_attention_mask,
                                    labels=decoder_labels,
                                    z_hidden_states=rel_hidden_states,
                                    return_dict=True)

        reconstruct_loss = decoder_outputs["loss"]
        reconstruct_logits = decoder_outputs["logits"][:, :-1].contiguous()

        loss = reconstruct_loss
        mlm_loss = 0.
        if decoder_labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            cls_loss = loss_fct(reshaped_logits, labels)
            loss = loss + cls_loss

            if mlm_labels is not None and (self.mlm_disabled is False or self.training is False):
                mlm_outputs = self.roberta(
                    mlm_input_ids,
                    attention_mask=mlm_attention_mask,
                    return_dict=return_dict
                )

                mlm_scores = self.lm_head(mlm_outputs[0])
                mlm_loss = self.mlm_alpha * loss_fct(mlm_scores.reshape(-1, self.vocab_size), mlm_labels.reshape(-1))
                if not self.mlm_disabled:
                    loss = loss + mlm_loss
            else:
                mlm_scores = None
                mlm_loss = None

            if not self.training:
                acc, true_label_num = layers.get_accuracy(reshaped_logits, labels)
                self.eval_metrics.update("acc", val=acc, n=true_label_num)
                self.eval_metrics.update("loss", val=loss.item(), n=true_label_num)
                self.eval_metrics.update("cls_loss", val=cls_loss.item(), n=true_label_num)

                masked_decoder_labels = decoder_labels.masked_fill(decoder_labels == self.decoder_config.eos_token_id, -1)
                acc, true_label_num = layers.get_accuracy(reconstruct_logits, masked_decoder_labels)
                self.eval_metrics.update("reconstruct_acc", val=acc, n=true_label_num)
                self.eval_metrics.update("reconstruct_loss", val=reconstruct_loss.item(), n=true_label_num)

                if mlm_labels is not None:
                    acc, true_label_num = layers.get_accuracy(mlm_scores, mlm_labels)
                    self.eval_metrics.update("mlm_acc", val=acc, n=true_label_num)
                    self.eval_metrics.update("mlm_loss", val=mlm_loss.item(), n=true_label_num)

        if not return_dict:
            output = (reconstruct_logits,) + outputs[2:] + (mlm_loss, reconstruct_loss,)
            return ((loss,) + output) if loss is not None else output

        return TaggingOutputClass(
            loss=loss,
            logits=reconstruct_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            mlm_loss=mlm_loss,
        )


def encode(input_ids, attention_mask, return_dict, encoder):
    outputs = encoder(input_ids, attention_mask=attention_mask, return_dict=return_dict)
    seq_hidden = outputs[0]
    pooled_output = seq_hidden[:, 0]
    return seq_hidden, pooled_output


class RobertaGPTForReconstructionV2(RobertaForMultipleChoiceForPreTrain, ABC):
    def __init__(self, config: RobertaConfig,
                 mlp_hidden_size: int = 768,
                 mlm_alpha: float = 1.0,
                 mlm_disabled: bool = False,
                 decoder: GPT2LMHeadModel = None,
                 decoder_config: GPT2Config = None):
        super().__init__(config, mlp_hidden_size, mlm_alpha, mlm_disabled)

        self.gpt2: GPT2LMHeadModel = decoder
        self.decoder_config = decoder_config if decoder_config is not None else self.gpt2.config

        self.decoder_proj = nn.Linear(config.hidden_size, self.decoder_config.n_embd)
        self.entity_proj = nn.Linear(config.hidden_size, self.decoder_config.n_embd)
        self._init_weights(self.decoder_proj)
        self._init_weights(self.entity_proj)

        self.init_metric("loss", "reconstruct_acc", "reconstruct_loss", "mlm_loss", "mlm_acc")

        self.mlm_alpha = mlm_alpha
        # The option is added to disable the MLM loss but keep computing the MLM accuracy on the validation set,
        # in order to observe if there is the catastrophic forgetting problem.
        self.mlm_disabled = mlm_disabled

        self.roberta_copy = None

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], *model_args, **kwargs):
        decoder = kwargs.pop("decoder")
        config = copy.deepcopy(decoder.config)

        use_copy_encoder = kwargs.pop("use_copy_encoder", False)

        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs, decoder_config=config)

        if use_copy_encoder:
            model.roberta_copy = copy.deepcopy(model.roberta)
            print(model.roberta_copy.__class__.__name__)

        model.gpt2 = decoder
        return model

    def forward(
            self,
            input_ids: Tensor,
            attention_mask: Tensor = None,
            token_type_ids: Tensor = None,
            mlm_input_ids: Tensor = None,
            mlm_attention_mask: Tensor = None,
            mlm_labels: Tensor = None,
            p_input_ids: Tensor = None,
            p_attention_mask: Tensor = None,
            decoding_input_ids: Tensor = None,
            decoding_attention_mask: Tensor = None,
            p_decoding_input_ids: Tensor = None,
            p_decoding_attention_mask: Tensor = None,
            h_span_marks: Tensor = None,
            t_span_marks: Tensor = None,
            entity_pair_mask: Tensor = None,
            p_h_span_marks: Tensor = None,
            p_t_span_marks: Tensor = None,
            p_entity_pair_mask: Tensor = None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            **kwargs
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the multiple choice classification loss. Indices should be in ``[0, ...,
            num_choices-1]`` where :obj:`num_choices` is the size of the second dimension of the input tensors. (See
            :obj:`input_ids` above)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        batch_size, seq_len = input_ids.size()

        seq_hidden, pooled_output = encode(input_ids, attention_mask, return_dict, self.roberta)
        p_seq_hidden, p_pooled_output = encode(p_input_ids, p_attention_mask, return_dict,
                                               encoder=self.roberta_copy if self.roberta_copy is not None else self.roberta)

        z = self.decoder_proj(pooled_output)
        pz = self.decoder_proj(p_pooled_output)

        equ = "bst,bth->bsh"
        s_num = h_span_marks.size(1)
        ps_num = p_h_span_marks.size(1)
        h_hidden = self.entity_proj(torch.einsum(equ, h_span_marks, seq_hidden))
        t_hidden = self.entity_proj(torch.einsum(equ, t_span_marks, seq_hidden))
        p_h_hidden = self.entity_proj(torch.einsum(equ, p_h_span_marks, p_seq_hidden))
        p_t_hidden = self.entity_proj(torch.einsum(equ, p_t_span_marks, p_seq_hidden))
        dec_inputs_embeds = torch.cat([h_hidden[:, :, None, :], t_hidden[:, :, None, :]], dim=2).reshape(batch_size, s_num * 2, -1)
        p_dec_inputs_embeds = torch.cat([p_h_hidden[:, :, None, :], p_t_hidden[:, :, None, :]], dim=2).reshape(batch_size, ps_num * 2, -1)
        dec_attention_mask = entity_pair_mask.unsqueeze(2).expand(-1, -1, 2).reshape(batch_size, s_num * 2)
        p_dec_attention_mask = p_entity_pair_mask.unsqueeze(2).expand(-1, -1, 2).reshape(batch_size, ps_num * 2)

        decoder_outputs1 = self.gpt2(inputs_embeds=dec_inputs_embeds,
                                     attention_mask=dec_attention_mask,
                                     # labels=p_decoding_input_ids,
                                     # target_attention_mask=p_decoding_attention_mask,
                                     labels=decoding_input_ids,
                                     target_attention_mask=decoding_attention_mask,
                                     z_hidden_states=pz.unsqueeze(1),
                                     return_dict=True)
        decoder_outputs2 = self.gpt2(inputs_embeds=p_dec_inputs_embeds,
                                     attention_mask=p_dec_attention_mask,
                                     # labels=decoding_input_ids,
                                     # target_attention_mask=decoding_attention_mask,  # FIXME: Fixed at 2023/02/07 23:36
                                     labels=p_decoding_input_ids,
                                     target_attention_mask=p_decoding_attention_mask,
                                     z_hidden_states=z.unsqueeze(1),
                                     return_dict=True)

        reconstruct_loss = (decoder_outputs1["loss"] + decoder_outputs2["loss"]) / 2.
        reconstruct_logits = torch.cat([decoder_outputs1["logits"][:, :-1], decoder_outputs2["logits"][:, :-1]], dim=1)

        loss = reconstruct_loss
        mlm_loss = 0.
        if decoding_input_ids is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)

            if mlm_labels is not None and (self.mlm_disabled is False or self.training is False):
                mlm_outputs = self.roberta(
                    mlm_input_ids,
                    attention_mask=mlm_attention_mask,
                    return_dict=return_dict
                )

                mlm_scores = self.lm_head(mlm_outputs[0])
                mlm_loss = self.mlm_alpha * loss_fct(mlm_scores.reshape(-1, self.vocab_size), mlm_labels.reshape(-1))
                if not self.mlm_disabled:
                    loss = loss + mlm_loss
            else:
                mlm_scores = None
                mlm_loss = None

            if not self.training:
                # masked_decoder_labels = decoder_labels.masked_fill(decoder_labels == self.decoder_config.eos_token_id, -1)
                masked_decoder_labels = torch.cat([decoding_input_ids, p_decoding_input_ids], dim=1)
                masked_decoder_labels = masked_decoder_labels.masked_fill(~torch.cat([
                    decoding_attention_mask, p_decoding_attention_mask], dim=1).bool(), -1)
                acc, true_label_num = layers.get_accuracy(reconstruct_logits, masked_decoder_labels)
                self.eval_metrics.update("reconstruct_acc", val=acc, n=true_label_num)
                self.eval_metrics.update("loss", val=loss.item(), n=true_label_num)
                self.eval_metrics.update("reconstruct_loss", val=reconstruct_loss.item(), n=true_label_num)

                if mlm_labels is not None:
                    acc, true_label_num = layers.get_accuracy(mlm_scores, mlm_labels)
                    self.eval_metrics.update("mlm_acc", val=acc, n=true_label_num)
                    self.eval_metrics.update("mlm_loss", val=mlm_loss.item(), n=true_label_num)

        if not return_dict:
            output = (reconstruct_logits,) + (mlm_loss, reconstruct_loss,)
            return ((loss,) + output) if loss is not None else output

        return TaggingOutputClass(
            loss=loss,
            logits=reconstruct_logits,
            mlm_loss=mlm_loss,
        )


def averaged_entity_hidden(seq_hidden: Tensor, weights: Tensor):
    return torch.einsum("bst,bth->bsh", weights, seq_hidden)


def extract_entity_pair_hidden(seq_hidden: Tensor, h_span_marks: Tensor, t_span_marks: Tensor, linear: nn.Module = None,
                               concat: str = "rel"):
    bsz, sent_num, seq_len = h_span_marks.size()
    h_hidden = averaged_entity_hidden(seq_hidden, h_span_marks)
    t_hidden = averaged_entity_hidden(seq_hidden, t_span_marks)

    if linear is not None:
        h_hidden = linear(h_hidden)
        t_hidden = linear(t_hidden)

    if concat == "rel":
        pair_hidden = torch.cat([h_hidden, t_hidden], dim=-1)
    elif concat == "pair":
        pair_hidden = torch.cat([h_hidden.unsqueeze(2), t_hidden.unsqueeze(2)], dim=2).reshape(bsz, sent_num * 2, -1)
    else:
        raise NotImplementedError

    return pair_hidden


class RobertaGPTForReconstructionMCQAV2(RobertaForMultipleChoiceForPreTrain, ABC):
    def __init__(self, config: RobertaConfig,
                 mlp_hidden_size: int = 768,
                 mlm_alpha: float = 1.0,
                 mlm_disabled: bool = False,
                 decoder: GPT2LMHeadModel = None,
                 decoder_config: GPT2Config = None):
        super().__init__(config, mlp_hidden_size, mlm_alpha, mlm_disabled)

        self.gpt2: GPT2LMHeadModel = decoder
        self.decoder_config = decoder_config if decoder_config is not None else self.gpt2.config

        self.decoder_proj = nn.Linear(config.hidden_size, self.decoder_config.n_embd)
        self.entity_proj = nn.Linear(config.hidden_size, self.decoder_config.n_embd)
        self._init_weights(self.decoder_proj)
        self._init_weights(self.entity_proj)

        self.init_metric("loss", "reconstruct_acc", "reconstruct_loss", "mlm_loss", "mlm_acc", "acc", "cls_loss")

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], *model_args, **kwargs):
        kai_full_state = kwargs.pop("kai_full_state", False)
        if kai_full_state:
            return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

        decoder = kwargs.pop("decoder")
        config = copy.deepcopy(decoder.config)

        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs, decoder_config=config)

        model.gpt2 = decoder
        return model

    def forward(
            self,
            input_ids: Tensor,
            attention_mask: Tensor = None,
            token_type_ids: Tensor = None,
            labels: Tensor = None,
            mlm_input_ids: Tensor = None,
            mlm_attention_mask: Tensor = None,
            mlm_labels: Tensor = None,
            p_input_ids: Tensor = None,
            p_attention_mask: Tensor = None,
            decoding_input_ids: Tensor = None,
            decoding_attention_mask: Tensor = None,
            p_decoding_input_ids: Tensor = None,
            p_decoding_attention_mask: Tensor = None,
            h_span_marks: Tensor = None,
            t_span_marks: Tensor = None,
            entity_pair_mask: Tensor = None,
            p_h_span_marks: Tensor = None,
            p_t_span_marks: Tensor = None,
            p_entity_pair_mask: Tensor = None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            **kwargs
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the multiple choice classification loss. Indices should be in ``[0, ...,
            num_choices-1]`` where :obj:`num_choices` is the size of the second dimension of the input tensors. (See
            :obj:`input_ids` above)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        batch_size, num_choices, seq_len = input_ids.size()

        input_ids = self.fold_tensor(input_ids)
        attention_mask = self.fold_tensor(attention_mask)
        # token_type_ids = self.fold_tensor(token_type_ids)

        seq_hidden, pooled_output = encode(input_ids, attention_mask, return_dict, self.roberta)

        logits = self.cls(self.dropout(self.pooler(pooled_output)))
        reshaped_logits = logits.view(-1, num_choices)

        seq_hidden = seq_hidden.reshape(batch_size, num_choices, seq_len, -1)[:, 0]
        pooled_output = pooled_output.reshape(batch_size, num_choices, -1)[:, 0]

        p_seq_hidden, p_pooled_output = encode(p_input_ids, p_attention_mask, return_dict, self.roberta)

        z = self.decoder_proj(pooled_output)
        pz = self.decoder_proj(p_pooled_output)

        dec_inputs_embeds = extract_entity_pair_hidden(seq_hidden, h_span_marks, t_span_marks, self.entity_proj, concat="pair")
        dec_attention_mask = entity_pair_mask.unsqueeze(2).expand(-1, -1, 2).reshape(batch_size, -1)

        p_dec_inputs_embeds = extract_entity_pair_hidden(p_seq_hidden, p_h_span_marks, p_t_span_marks, self.entity_proj, concat="pair")
        p_dec_attention_mask = p_entity_pair_mask.unsqueeze(2).expand(-1, -1, 2).reshape(batch_size, -1)

        decoder_outputs1 = self.gpt2(inputs_embeds=dec_inputs_embeds,
                                     attention_mask=dec_attention_mask,
                                     # labels=p_decoding_input_ids,
                                     # target_attention_mask=p_decoding_attention_mask,
                                     labels=decoding_input_ids,
                                     target_attention_mask=decoding_attention_mask,
                                     z_hidden_states=pz.unsqueeze(1),
                                     return_dict=True)
        decoder_outputs2 = self.gpt2(inputs_embeds=p_dec_inputs_embeds,
                                     attention_mask=p_dec_attention_mask,
                                     # labels=decoding_input_ids,
                                     # target_attention_mask=decoding_attention_mask,
                                     labels=p_decoding_input_ids,
                                     target_attention_mask=p_decoding_attention_mask,
                                     z_hidden_states=z.unsqueeze(1),
                                     return_dict=True)

        reconstruct_loss = (decoder_outputs1["loss"] + decoder_outputs2["loss"]) / 2.
        reconstruct_logits = torch.cat([decoder_outputs1["logits"][:, :-1], decoder_outputs2["logits"][:, :-1]], dim=1)

        loss = reconstruct_loss
        mlm_loss = 0.
        cls_loss = 0.
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            cls_loss = loss_fct(reshaped_logits, labels)
            loss = loss + cls_loss

            if mlm_labels is not None and (self.mlm_disabled is False or self.training is False):
                mlm_outputs = self.roberta(
                    mlm_input_ids,
                    attention_mask=mlm_attention_mask,
                    return_dict=return_dict
                )

                mlm_scores = self.lm_head(mlm_outputs[0])
                mlm_loss = self.mlm_alpha * loss_fct(mlm_scores.reshape(-1, self.vocab_size), mlm_labels.reshape(-1))
                if not self.mlm_disabled:
                    loss = loss + mlm_loss
            else:
                mlm_scores = None
                mlm_loss = None

            if not self.training:
                acc, true_label_num = layers.get_accuracy(reshaped_logits, labels)
                self.eval_metrics.update("acc", val=acc, n=true_label_num)
                self.eval_metrics.update("loss", val=loss.item(), n=true_label_num)
                self.eval_metrics.update("cls_loss", val=cls_loss.item(), n=true_label_num)

                masked_decoder_labels = torch.cat([decoding_input_ids, p_decoding_input_ids], dim=1)
                masked_decoder_labels = masked_decoder_labels.masked_fill(~torch.cat([
                    decoding_attention_mask, p_decoding_attention_mask], dim=1).bool(), -1)
                acc, true_label_num = layers.get_accuracy(reconstruct_logits, masked_decoder_labels)
                self.eval_metrics.update("reconstruct_acc", val=acc, n=true_label_num)
                self.eval_metrics.update("reconstruct_loss", val=reconstruct_loss.item(), n=true_label_num)

                if mlm_labels is not None:
                    acc, true_label_num = layers.get_accuracy(mlm_scores, mlm_labels)
                    self.eval_metrics.update("mlm_acc", val=acc, n=true_label_num)
                    self.eval_metrics.update("mlm_loss", val=mlm_loss.item(), n=true_label_num)

        if not return_dict:
            output = (reconstruct_logits,) + (mlm_loss, reconstruct_loss,)
            return ((loss,) + output) if loss is not None else output

        return TaggingOutputClass(
            loss=loss,
            cls_loss=cls_loss,
            logits=reconstruct_logits,
            mlm_loss=mlm_loss,
        )
