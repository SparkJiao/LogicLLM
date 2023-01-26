import copy
import pickle
from abc import ABC
from dataclasses import dataclass
from typing import Union, Dict

import torch
import transformers
from torch import nn, Tensor
from torch.nn import CrossEntropyLoss
from transformers.models.roberta.modeling_roberta import RobertaConfig
from transformers import AutoTokenizer

from general_util.logger import get_child_logger
from models.roberta import RobertaForMultipleChoiceForPreTrain, RelDecoderHead, MultipleChoicePreTrainModelOutput, attentive_pooling
from modules import layers

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
