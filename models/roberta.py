import copy
import pickle
from abc import ABC
from dataclasses import dataclass
from typing import Union

import torch
import torch.distributed as dist
from fairscale.nn.checkpoint.checkpoint_activations import checkpoint_wrapper
from torch import nn, Tensor
from torch.nn import CrossEntropyLoss
from transformers.activations import gelu
from transformers.modeling_outputs import MultipleChoiceModelOutput
from transformers.models.roberta.modeling_roberta import RobertaModel, RobertaPreTrainedModel, RobertaConfig, RobertaLMHead, \
    MaskedLMOutput, SequenceClassifierOutput, RobertaEncoder
from transformers.models.t5.modeling_t5 import T5Stack, T5Config
from transformers.utils import is_torch_fx_proxy

from general_util.logger import get_child_logger
from general_util.mixin import LogMixin
from modules import layers

logger = get_child_logger("RoBERTa")


class RobertaForMultipleChoice(RobertaPreTrainedModel, LogMixin, ABC):
    def __init__(self, config: RobertaConfig,
                 fs_checkpoint: bool = False,
                 fs_checkpoint_offload_to_cpu: bool = False,
                 fs_checkpoint_maintain_forward_counter: bool = False,
                 freeze_encoder: bool = False,
                 no_pooler: bool = False):
        super().__init__(config)

        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(p=getattr(config, "pooler_dropout", config.hidden_dropout_prob))
        self.classifier = nn.Linear(config.hidden_size, 1)

        if fs_checkpoint:
            for i in range(config.num_hidden_layers):
                self.roberta.encoder.layer[i] = checkpoint_wrapper(self.roberta.encoder.layer[i],
                                                                   offload_to_cpu=fs_checkpoint_offload_to_cpu,
                                                                   maintain_forward_counter=fs_checkpoint_maintain_forward_counter)

        self.no_pooler = no_pooler
        self.freeze_encoder = freeze_encoder
        if freeze_encoder:
            layers.freeze_module(self.roberta)

        self.init_weights()

        self.init_metric("loss", "acc")

    @staticmethod
    def fold_tensor(x: Tensor):
        if x is None:
            return x
        return x.reshape(-1, x.size(-1))

    def forward(
            self,
            input_ids: Tensor,
            attention_mask: Tensor = None,
            token_type_ids: Tensor = None,
            op_mask: Tensor = None,
            labels: Tensor = None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the multiple choice classification loss. Indices should be in ``[0, ...,
            num_choices-1]`` where :obj:`num_choices` is the size of the second dimension of the input tensors. (See
            :obj:`input_ids` above)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        num_choices = input_ids.shape[1]

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
        if self.no_pooler:
            pooled_output = outputs[0][:, 0]
        else:
            pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)

        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices) + (1.0 - op_mask.to(logits.dtype)) * -1e4

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(reshaped_logits, labels)

            if not self.training:
                acc, true_label_num = layers.get_accuracy(reshaped_logits, labels)
                self.eval_metrics.update("acc", val=acc, n=true_label_num)
                self.eval_metrics.update("loss", val=loss.item(), n=true_label_num)

        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@dataclass
class MultipleChoicePreTrainModelOutput(MultipleChoiceModelOutput):
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


@dataclass
class SequenceClassificationPreTrainModelOutput(SequenceClassifierOutput):
    mlm_loss: torch.FloatTensor = None
    mlm_acc: torch.FloatTensor = None
    cls_loss: torch.FloatTensor = None
    cls_acc: torch.FloatTensor = None


class RobertaForMultipleChoiceForPreTrain(RobertaPreTrainedModel, LogMixin, ABC):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config: RobertaConfig,
                 mlp_hidden_size: int = 768,
                 mlm_alpha: float = 1.0,
                 mlm_disabled: bool = False,
                 fs_checkpoint: bool = False,
                 fs_checkpoint_offload_to_cpu: bool = False,
                 fs_checkpoint_start_layer_id: int = 0):
        super().__init__(config)

        self.roberta = RobertaModel(config)
        self.lm_head = RobertaLMHead(config)
        self.dropout = nn.Dropout(p=getattr(config, "pooler_dropout", config.hidden_dropout_prob))
        self.vocab_size = config.vocab_size

        self.pooler = nn.Sequential(
            nn.Linear(config.hidden_size, mlp_hidden_size),
            nn.Tanh()
        )
        self.cls = nn.Linear(mlp_hidden_size, 1)

        if fs_checkpoint:
            for i in range(fs_checkpoint_start_layer_id, config.num_hidden_layers):
                self.roberta.encoder.layer[i] = checkpoint_wrapper(self.roberta.encoder.layer[i],
                                                                   offload_to_cpu=fs_checkpoint_offload_to_cpu)

        self.init_weights()

        self.init_metric("loss", "acc", "mlm_loss", "mlm_acc", "cls_loss")

        self.mlm_alpha = mlm_alpha
        # The option is added to disable the MLM loss but keep computing the MLM accuracy on the validation set,
        # in order to observe if there is the catastrophic forgetting problem.
        self.mlm_disabled = mlm_disabled

    @staticmethod
    def fold_tensor(x: Tensor):
        if x is None:
            return x
        return x.reshape(-1, x.size(-1))

    def forward(
            self,
            input_ids: Tensor,
            attention_mask: Tensor = None,
            token_type_ids: Tensor = None,
            labels: Tensor = None,
            mlm_input_ids: Tensor = None,
            mlm_attention_mask: Tensor = None,
            mlm_labels: Tensor = None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the multiple choice classification loss. Indices should be in ``[0, ...,
            num_choices-1]`` where :obj:`num_choices` is the size of the second dimension of the input tensors. (See
            :obj:`input_ids` above)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        num_choices = input_ids.shape[1]

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

        loss = 0.
        mlm_loss = 0.
        cls_loss = 0.
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

            if not self.training:
                acc, true_label_num = layers.get_accuracy(reshaped_logits, labels)
                self.eval_metrics.update("acc", val=acc, n=true_label_num)
                self.eval_metrics.update("loss", val=loss.item(), n=true_label_num)
                self.eval_metrics.update("cls_loss", val=cls_loss.item(), n=true_label_num)

                if mlm_labels is not None:
                    acc, true_label_num = layers.get_accuracy(mlm_scores, mlm_labels)
                    self.eval_metrics.update("mlm_acc", val=acc, n=true_label_num)
                    self.eval_metrics.update("mlm_loss", val=mlm_loss.item(), n=true_label_num)

        if not return_dict:
            output = (reshaped_logits,) + outputs[2:] + (mlm_loss, cls_loss,)
            return ((loss,) + output) if loss is not None else output

        return MultipleChoicePreTrainModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            mlm_loss=mlm_loss,
            cls_loss=cls_loss,
        )


class RobertaForMultipleChoiceForPreTrainWithPair(RobertaPreTrainedModel, LogMixin, ABC):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config: RobertaConfig,
                 mlp_hidden_size: int = 768,
                 mlm_alpha: float = 1.0,
                 mlm_disabled: bool = False):
        super().__init__(config)

        self.roberta = RobertaModel(config)
        self.lm_head = RobertaLMHead(config)
        self.dropout = nn.Dropout(p=getattr(config, "pooler_dropout", config.hidden_dropout_prob))
        self.vocab_size = config.vocab_size

        self.pooler = nn.Sequential(
            nn.Linear(config.hidden_size, mlp_hidden_size),
            nn.Tanh()
        )
        self.cls = nn.Linear(mlp_hidden_size, 1)

        self.pair_pooler = nn.Sequential(
            nn.Linear(config.hidden_size, mlp_hidden_size),
            nn.GELU()
        )
        self.pair_proj = nn.Linear(mlp_hidden_size, config.hidden_size)

        self.init_weights()

        self.init_metric("loss", "acc", "mlm_loss", "mlm_acc", "cls_loss", "pair_acc", "pair_loss",
                         "pair_true_label_num", "pair_value_num")

        self.mlm_alpha = mlm_alpha
        # The option is added to disable the MLM loss but keep computing the MLM accuracy on the validation set,
        # in order to observe if there is the catastrophic forgetting problem.
        self.mlm_disabled = mlm_disabled

    @staticmethod
    def fold_tensor(x: Tensor):
        if x is None:
            return x
        return x.reshape(-1, x.size(-1))

    def forward(
            self,
            input_ids: Tensor,
            attention_mask: Tensor = None,
            token_type_ids: Tensor = None,
            labels: Tensor = None,
            mlm_input_ids: Tensor = None,
            mlm_attention_mask: Tensor = None,
            mlm_labels: Tensor = None,
            pair_input_ids: Tensor = None,
            pair_attention_mask: Tensor = None,
            pair_mask: Tensor = None,
            pair_labels: Tensor = None,
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
        num_choices = input_ids.shape[1]
        batch_size = input_ids.shape[0]

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

        pair_outputs = self.roberta(pair_input_ids,
                                    attention_mask=pair_attention_mask,
                                    return_dict=return_dict)
        pair_pooled_output = pair_outputs[0][:, 0]

        pair_left = self.pair_proj(self.dropout(self.pair_pooler(pooled_output.reshape(batch_size, num_choices, -1)[:, 0])))
        pair_right = self.pair_proj(self.dropout(self.pair_pooler(pair_pooled_output)))

        align_logits_a = torch.einsum("ah,bh->ab", pair_left, pair_right)
        align_logits_b = torch.einsum("ah,bh->ab", pair_left, pair_left)
        align_logits = torch.cat([align_logits_a, align_logits_b], dim=1) + pair_mask * -10000.0

        loss = 0.
        mlm_loss = 0.
        cls_loss = 0.
        pair_loss = 0.
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

            if pair_labels is not None:
                true_label_num = (pair_labels > -1).sum().item()
                if true_label_num:
                    pair_loss = loss_fct(align_logits, pair_labels)
                    loss = loss + pair_loss

            if not self.training:
                acc, true_label_num = layers.get_accuracy(reshaped_logits, labels)
                self.eval_metrics.update("acc", val=acc, n=true_label_num)
                self.eval_metrics.update("loss", val=loss.item(), n=true_label_num)
                self.eval_metrics.update("cls_loss", val=cls_loss.item(), n=true_label_num)

                if mlm_labels is not None and self.mlm_disabled is False:
                    acc, true_label_num = layers.get_accuracy(mlm_scores, mlm_labels)
                    self.eval_metrics.update("mlm_acc", val=acc, n=true_label_num)
                    self.eval_metrics.update("mlm_loss", val=mlm_loss.item(), n=true_label_num)

                if pair_labels is not None and pair_loss > 0:
                    acc, true_label_num = layers.get_accuracy(align_logits, pair_labels)
                    self.eval_metrics.update("pair_acc", val=acc, n=true_label_num)
                    if isinstance(pair_loss, Tensor):
                        self.eval_metrics.update("pair_loss", val=pair_loss.item(), n=true_label_num)
                    else:
                        self.eval_metrics.update("pair_loss", val=0, n=true_label_num)
                    self.eval_metrics.update("pair_true_label_num", val=true_label_num / batch_size, n=batch_size)
                    self.eval_metrics.update("pair_value_num",
                                             val=(1 - pair_mask).sum(dim=-1)[pair_labels > -1].sum().item() / true_label_num,
                                             n=true_label_num)

        if not return_dict:
            output = (reshaped_logits,) + outputs[2:] + (mlm_loss, cls_loss,)
            return ((loss,) + output) if loss is not None else output

        return MultipleChoicePreTrainModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            mlm_loss=mlm_loss,
            cls_loss=cls_loss,
            pair_loss=pair_loss,
        )


class RobertaForMultipleChoiceForPreTrainWithPairFull(RobertaForMultipleChoiceForPreTrainWithPair, ABC):
    def __init__(self, config: RobertaConfig,
                 mlp_hidden_size: int = 768,
                 mlm_alpha: float = 1.0,
                 mlm_disabled: bool = False,
                 stop_key_gradient: bool = False,
                 dist_neg_sampling: bool = False):
        super().__init__(config, mlp_hidden_size, mlm_alpha, mlm_disabled)
        self.stop_key_gradient = stop_key_gradient
        self.dist_neg_sampling = dist_neg_sampling

    @staticmethod
    def fold_tensor(x: Tensor):
        if x is None:
            return x
        return x.reshape(-1, x.size(-1))

    def forward(
            self,
            input_ids: Tensor,
            attention_mask: Tensor = None,
            token_type_ids: Tensor = None,
            labels: Tensor = None,
            mlm_input_ids: Tensor = None,
            mlm_attention_mask: Tensor = None,
            mlm_labels: Tensor = None,
            pair_q_input_ids: Tensor = None,
            pair_q_attention_mask: Tensor = None,
            pair_k_input_ids: Tensor = None,
            pair_k_attention_mask: Tensor = None,
            pair_mask: Tensor = None,
            pair_labels: Tensor = None,
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
        num_choices = input_ids.shape[1]
        batch_size = input_ids.shape[0]

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

        pair_q_outputs = self.roberta(pair_q_input_ids, attention_mask=pair_q_attention_mask, return_dict=return_dict)
        pair_q_pooled_output = pair_q_outputs[0][:, 0]

        pair_k_outputs = self.roberta(pair_k_input_ids, attention_mask=pair_k_attention_mask, return_dict=return_dict)
        pair_k_pooled_output = pair_k_outputs[0][:, 0]

        pair_q = self.pair_proj(self.dropout(self.pair_pooler(pair_q_pooled_output)))
        pair_k = self.pair_proj(self.dropout(self.pair_pooler(pair_k_pooled_output)))

        if self.stop_key_gradient:
            pair_k = pair_k.detach()

        if dist.is_initialized() and self.dist_neg_sampling:
            k_list = [torch.zeros_like(pair_k) for _ in range(dist.get_world_size())]
            dist.all_gather(k_list, pair_k.contiguous())

            re_arranged_k_list = [pair_k]
            for i in range(dist.get_world_size()):
                if i == dist.get_rank():
                    continue
                re_arranged_k_list.append(k_list[i])
            pair_k = torch.cat(re_arranged_k_list, dim=0)
            ex_pair_mask = pair_mask.new_zeros(batch_size, batch_size * 3)
            pair_mask = torch.cat([pair_mask, ex_pair_mask], dim=1)

        align_logits = torch.einsum("ah,bh->ab", pair_q, pair_k)
        align_logits = align_logits + pair_mask * -10000.0

        loss = 0.
        mlm_loss = 0.
        cls_loss = 0.
        pair_loss = 0.
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

            # print(pair_labels)
            if pair_labels is not None:
                true_label_num = (pair_labels > -1).sum().item()
                if true_label_num:
                    # print(111)
                    pair_loss = loss_fct(align_logits, pair_labels)
                    # print(pair_loss)
                    loss = loss + pair_loss

            if not self.training:
                acc, true_label_num = layers.get_accuracy(reshaped_logits, labels)
                self.eval_metrics.update("acc", val=acc, n=true_label_num)
                self.eval_metrics.update("loss", val=loss.item(), n=true_label_num)
                self.eval_metrics.update("cls_loss", val=cls_loss.item(), n=true_label_num)

                if mlm_labels is not None and self.mlm_disabled is False:
                    acc, true_label_num = layers.get_accuracy(mlm_scores, mlm_labels)
                    self.eval_metrics.update("mlm_acc", val=acc, n=true_label_num)
                    self.eval_metrics.update("mlm_loss", val=mlm_loss.item(), n=true_label_num)

                if pair_labels is not None and pair_loss > 0:
                    acc, true_label_num = layers.get_accuracy(align_logits, pair_labels)
                    self.eval_metrics.update("pair_acc", val=acc, n=true_label_num)
                    if isinstance(pair_loss, Tensor):
                        self.eval_metrics.update("pair_loss", val=pair_loss.item(), n=true_label_num)
                    else:
                        self.eval_metrics.update("pair_loss", val=0, n=true_label_num)
                    self.eval_metrics.update("pair_true_label_num", val=true_label_num / batch_size, n=batch_size)
                    self.eval_metrics.update("pair_value_num",
                                             val=(1 - pair_mask).sum(dim=-1)[pair_labels > -1].sum().item() / true_label_num,
                                             n=true_label_num)

        if not return_dict:
            output = (reshaped_logits,) + outputs[2:] + (mlm_loss, cls_loss,)
            return ((loss,) + output) if loss is not None else output

        return MultipleChoicePreTrainModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            mlm_loss=mlm_loss,
            cls_loss=cls_loss,
            pair_loss=pair_loss,
        )


def attentive_pooling(linear_layer, sentence_hidden: Tensor, sent_token_index: Tensor, sent_token_mask: Tensor, sent_index: Tensor = None):
    batch, sent_num, sent_len = sent_token_index.size()
    h = sentence_hidden.size(-1)
    sent_token_hidden = torch.gather(sentence_hidden, dim=1,
                                     index=sent_token_index.unsqueeze(-1).expand(-1, -1, -1, h).reshape(
                                         batch, sent_num * sent_len, h
                                     )).reshape(batch, sent_num, sent_len, h)
    sent_token_scores = linear_layer(sent_token_hidden).squeeze(-1)
    # sent_token_scores = (1 - sent_token_mask) * torch.finfo(sentence_hidden.dtype).min + sent_token_scores
    sent_token_scores = (1 - sent_token_mask) * -10000.0 + sent_token_scores
    alpha = torch.softmax(sent_token_scores, dim=2)
    sent_hidden = torch.einsum("bst,bsth->bsh", alpha, sent_token_hidden)
    if sent_index is not None:
        sent_hidden = torch.index_select(sent_hidden.reshape(batch * sent_num, h), dim=0, index=sent_index)
    return sent_hidden


class RobertaForMultipleChoiceForPreTrainWithPairFullLocal(RobertaForMultipleChoiceForPreTrainWithPair, ABC):
    def __init__(self, config: RobertaConfig,
                 mlp_hidden_size: int = 768,
                 mlm_alpha: float = 1.0,
                 mlm_disabled: bool = False,
                 stop_key_gradient: bool = False,
                 dist_neg_sampling: bool = False):
        super().__init__(config, mlp_hidden_size, mlm_alpha, mlm_disabled)
        self.stop_key_gradient = stop_key_gradient
        self.dist_neg_sampling = dist_neg_sampling

        # self.local_pair_pooler = nn.Sequential(
        #     nn.Linear(config.hidden_size, mlp_hidden_size),
        #     nn.Tanh()
        # )
        self.local_pair_proj = nn.Linear(config.hidden_size, config.hidden_size)

        self.attn_pooler = nn.Linear(config.hidden_size, 1)

        self.post_init()

        self.init_metric("loss", "acc", "mlm_loss", "mlm_acc", "cls_loss", "pair_acc", "pair_loss",
                         "pair_true_label_num", "pair_value_num",
                         "local_ctr_loss", "local_ctr_acc", "local_ctr_true_label_num")

    @staticmethod
    def fold_tensor(x: Tensor):
        if x is None:
            return x
        return x.reshape(-1, x.size(-1))

    def forward(
            self,
            input_ids: Tensor,
            attention_mask: Tensor = None,
            token_type_ids: Tensor = None,
            labels: Tensor = None,
            mlm_input_ids: Tensor = None,
            mlm_attention_mask: Tensor = None,
            mlm_labels: Tensor = None,
            pair_q_input_ids: Tensor = None,
            pair_q_attention_mask: Tensor = None,
            pair_k_input_ids: Tensor = None,
            pair_k_attention_mask: Tensor = None,
            pair_mask: Tensor = None,
            pair_labels: Tensor = None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            q_sent_token_index: Tensor = None,
            q_sent_token_mask: Tensor = None,
            q_sent_index: Tensor = None,
            k_sent_token_index: Tensor = None,
            k_sent_token_mask: Tensor = None,
            k_sent_index: Tensor = None,
            local_ctr_labels: Tensor = None,
            local_ctr_mask: Tensor = None,
            pair_q_orig_input_ids: Tensor = None,
            pair_q_orig_attention_mask: Tensor = None,
            pair_k_orig_input_ids: Tensor = None,
            pair_k_orig_attention_mask: Tensor = None,
            **kwargs
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the multiple choice classification loss. Indices should be in ``[0, ...,
            num_choices-1]`` where :obj:`num_choices` is the size of the second dimension of the input tensors. (See
            :obj:`input_ids` above)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        num_choices = input_ids.shape[1]
        batch_size = input_ids.shape[0]

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

        pair_q_outputs = self.roberta(pair_q_input_ids, attention_mask=pair_q_attention_mask, return_dict=return_dict)
        pair_q_pooled_output = pair_q_outputs[0][:, 0]

        pair_k_outputs = self.roberta(pair_k_input_ids, attention_mask=pair_k_attention_mask, return_dict=return_dict)
        pair_k_pooled_output = pair_k_outputs[0][:, 0]

        pair_q = self.pair_proj(self.dropout(self.pair_pooler(pair_q_pooled_output)))
        pair_k = self.pair_proj(self.dropout(self.pair_pooler(pair_k_pooled_output)))

        if self.stop_key_gradient:
            pair_k = pair_k.detach()

        if dist.is_initialized() and self.dist_neg_sampling:
            k_list = [torch.zeros_like(pair_k) for _ in range(dist.get_world_size())]
            dist.all_gather(k_list, pair_k.contiguous())

            re_arranged_k_list = [pair_k]
            for i in range(dist.get_world_size()):
                if i == dist.get_rank():
                    continue
                re_arranged_k_list.append(k_list[i])
            pair_k = torch.cat(re_arranged_k_list, dim=0)
            ex_pair_mask = pair_mask.new_zeros(batch_size, batch_size * 3)
            pair_mask = torch.cat([pair_mask, ex_pair_mask], dim=1)

        align_logits = torch.einsum("ah,bh->ab", pair_q, pair_k)
        align_logits = align_logits + pair_mask * -10000.0

        # local contrastive learning
        if pair_q_orig_input_ids is not None:
            pair_q_orig_outputs = self.roberta(pair_q_orig_input_ids, attention_mask=pair_q_orig_attention_mask, return_dict=return_dict)
            pair_q_sent_h = attentive_pooling(self.attn_pooler, pair_q_orig_outputs[0], q_sent_token_index, q_sent_token_mask, q_sent_index)
        else:
            pair_q_sent_h = attentive_pooling(self.attn_pooler, pair_q_outputs[0], q_sent_token_index, q_sent_token_mask, q_sent_index)

        if pair_k_orig_input_ids is not None:
            pair_k_orig_outputs = self.roberta(pair_k_orig_input_ids, attention_mask=pair_k_orig_attention_mask, return_dict=return_dict)
            pair_k_sent_h = attentive_pooling(self.attn_pooler, pair_k_orig_outputs[0], k_sent_token_index, k_sent_token_mask, k_sent_index)
        else:
            pair_k_sent_h = attentive_pooling(self.attn_pooler, pair_k_outputs[0], k_sent_token_index, k_sent_token_mask, k_sent_index)

        pair_q_sent_h = self.local_pair_proj(self.dropout(pair_q_sent_h))
        pair_k_sent_h = self.local_pair_proj(self.dropout(pair_k_sent_h))

        if len(pair_k_sent_h.size()) == 3:
            local_ctr_logits = torch.einsum("bxh,byh->bxy", pair_q_sent_h, pair_k_sent_h)
        else:
            local_ctr_logits = torch.einsum("ah,bh->ab", pair_q_sent_h, pair_k_sent_h)
        local_ctr_logits = local_ctr_logits + (1 - local_ctr_mask) * -10000.0

        loss = 0.
        mlm_loss = 0.
        cls_loss = 0.
        pair_loss = 0.
        local_ctr_loss = 0.
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

            # print(pair_labels)
            if pair_labels is not None:
                true_label_num = (pair_labels > -1).sum().item()
                if true_label_num:
                    pair_loss = loss_fct(align_logits, pair_labels)
                    loss = loss + pair_loss

            if local_ctr_labels is not None:
                true_label_num = (local_ctr_labels > -1).sum().item()
                if true_label_num:
                    local_ctr_loss = loss_fct(local_ctr_logits.reshape(-1, local_ctr_logits.size(-1)), local_ctr_labels.reshape(-1))
                    loss = loss + local_ctr_loss

            if not self.training:
                acc, true_label_num = layers.get_accuracy(reshaped_logits, labels)
                self.eval_metrics.update("acc", val=acc, n=true_label_num)
                self.eval_metrics.update("loss", val=loss.item(), n=true_label_num)
                self.eval_metrics.update("cls_loss", val=cls_loss.item(), n=true_label_num)

                if mlm_labels is not None and self.mlm_disabled is False:
                    acc, true_label_num = layers.get_accuracy(mlm_scores, mlm_labels)
                    self.eval_metrics.update("mlm_acc", val=acc, n=true_label_num)
                    self.eval_metrics.update("mlm_loss", val=mlm_loss.item(), n=true_label_num)

                if pair_labels is not None and pair_loss > 0:
                    acc, true_label_num = layers.get_accuracy(align_logits, pair_labels)
                    self.eval_metrics.update("pair_acc", val=acc, n=true_label_num)
                    self.eval_metrics.update("pair_loss", val=pair_loss, n=true_label_num)
                    self.eval_metrics.update("pair_true_label_num", val=true_label_num / batch_size, n=batch_size)
                    self.eval_metrics.update("pair_value_num",
                                             val=(1 - pair_mask).sum(dim=-1)[pair_labels > -1].sum().item() / true_label_num,
                                             n=true_label_num)

                if local_ctr_labels is not None and local_ctr_loss > 0:
                    acc, true_label_num = layers.get_accuracy(local_ctr_logits, local_ctr_labels)
                    self.eval_metrics.update("local_ctr_acc", val=acc, n=true_label_num)
                    self.eval_metrics.update("local_ctr_loss", val=local_ctr_loss, n=true_label_num)
                    self.eval_metrics.update("local_ctr_true_label_num", val=true_label_num / batch_size, n=batch_size)

        if not return_dict:
            output = (reshaped_logits,) + outputs[2:] + (mlm_loss, cls_loss,)
            return ((loss,) + output) if loss is not None else output

        return MultipleChoicePreTrainModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            mlm_loss=mlm_loss,
            cls_loss=cls_loss,
            pair_loss=pair_loss,
            local_ctr_loss=local_ctr_loss
        )


class RobertaForMultipleChoiceForPreTrainWithPairFullTagging(RobertaForMultipleChoiceForPreTrainWithPairFull, ABC):
    def __init__(self, config: RobertaConfig,
                 mlp_hidden_size: int = 768,
                 mlm_alpha: float = 1.0,
                 mlm_disabled: bool = False):
        super().__init__(config, mlp_hidden_size, mlm_alpha, mlm_disabled)

        self.tagging_cls = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Tanh(),
            nn.Linear(config.hidden_size, 2),
        )

        self.init_metric("loss", "acc", "mlm_loss", "mlm_acc", "cls_loss", "pair_acc", "pair_loss",
                         "pair_true_label_num", "pair_value_num", "tagging_loss", "tagging_acc")

    @staticmethod
    def fold_tensor(x: Tensor):
        if x is None:
            return x
        return x.reshape(-1, x.size(-1))

    def forward(
            self,
            input_ids: Tensor,
            attention_mask: Tensor = None,
            token_type_ids: Tensor = None,
            labels: Tensor = None,
            mlm_input_ids: Tensor = None,
            mlm_attention_mask: Tensor = None,
            mlm_labels: Tensor = None,
            pair_q_input_ids: Tensor = None,
            pair_q_attention_mask: Tensor = None,
            pair_k_input_ids: Tensor = None,
            pair_k_attention_mask: Tensor = None,
            pair_mask: Tensor = None,
            pair_labels: Tensor = None,
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
        num_choices = input_ids.shape[1]
        batch_size = input_ids.shape[0]

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

        pair_q_outputs = self.roberta(pair_q_input_ids, attention_mask=pair_q_attention_mask, return_dict=return_dict)
        pair_q_pooled_output = pair_q_outputs[0][:, 0]

        pair_k_outputs = self.roberta(pair_k_input_ids, attention_mask=pair_k_attention_mask, return_dict=return_dict)
        pair_k_pooled_output = pair_k_outputs[0][:, 0]

        pair_q = self.pair_proj(self.dropout(self.pair_pooler(pair_q_pooled_output)))
        pair_k = self.pair_proj(self.dropout(self.pair_pooler(pair_k_pooled_output)))

        align_logits = torch.einsum("ah,bh->ab", pair_q, pair_k)
        align_logits = align_logits + pair_mask * -10000.0

        tagging_logits = self.tagging_cls(outputs[0].reshape(batch_size, num_choices, input_ids.size(1), -1)[:, 0])

        loss = 0.
        mlm_loss = 0.
        cls_loss = 0.
        pair_loss = 0.
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

            # print(pair_labels)
            if pair_labels is not None:
                true_label_num = (pair_labels > -1).sum().item()
                if true_label_num:
                    # print(111)
                    pair_loss = loss_fct(align_logits, pair_labels)
                    # print(pair_loss)
                    loss = loss + pair_loss

            if tagging_labels is not None:
                tagging_loss = loss_fct(tagging_logits.reshape(-1, 2), tagging_labels.reshape(-1))
                loss = loss + tagging_loss

            if not self.training:
                acc, true_label_num = layers.get_accuracy(reshaped_logits, labels)
                self.eval_metrics.update("acc", val=acc, n=true_label_num)
                self.eval_metrics.update("loss", val=loss.item(), n=true_label_num)
                self.eval_metrics.update("cls_loss", val=cls_loss.item(), n=true_label_num)

                if mlm_labels is not None and self.mlm_disabled is False:
                    acc, true_label_num = layers.get_accuracy(mlm_scores, mlm_labels)
                    self.eval_metrics.update("mlm_acc", val=acc, n=true_label_num)
                    self.eval_metrics.update("mlm_loss", val=mlm_loss.item(), n=true_label_num)

                if pair_labels is not None and pair_loss > 0:
                    acc, true_label_num = layers.get_accuracy(align_logits, pair_labels)
                    self.eval_metrics.update("pair_acc", val=acc, n=true_label_num)
                    if isinstance(pair_loss, Tensor):
                        self.eval_metrics.update("pair_loss", val=pair_loss.item(), n=true_label_num)
                    else:
                        self.eval_metrics.update("pair_loss", val=0, n=true_label_num)
                    self.eval_metrics.update("pair_true_label_num", val=true_label_num / batch_size, n=batch_size)
                    self.eval_metrics.update("pair_value_num",
                                             val=(1 - pair_mask).sum(dim=-1)[pair_labels > -1].sum().item() / true_label_num,
                                             n=true_label_num)

                if tagging_labels is not None:
                    acc, true_label_num = layers.get_accuracy(tagging_logits, tagging_labels)
                    self.eval_metrics.update("tagging_loss", tagging_loss.item(), n=true_label_num)
                    self.eval_metrics.update("tagging_acc", acc, n=true_label_num)

        if not return_dict:
            output = (reshaped_logits,) + outputs[2:] + (mlm_loss, cls_loss,)
            return ((loss,) + output) if loss is not None else output

        return MultipleChoicePreTrainModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            mlm_loss=mlm_loss,
            cls_loss=cls_loss,
            pair_loss=pair_loss,
            tagging_loss=tagging_loss,
        )


class RobertaForSequenceClassificationForPreTrain(RobertaPreTrainedModel, LogMixin, ABC):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config: RobertaConfig,
                 mlp_hidden_size: int = 768,
                 mlm_alpha: float = 1.0,
                 mlm_disabled: bool = False,
                 fs_checkpoint: bool = False,
                 fs_checkpoint_offload_to_cpu: bool = False,
                 fs_checkpoint_start_layer_id: int = 0):
        super().__init__(config)

        self.roberta = RobertaModel(config)
        self.lm_head = RobertaLMHead(config)
        self.dropout = nn.Dropout(p=getattr(config, "pooler_dropout", config.hidden_dropout_prob))
        self.vocab_size = config.vocab_size

        self.pooler = nn.Sequential(
            nn.Linear(config.hidden_size, mlp_hidden_size),
            nn.Tanh()
        )
        self.cls = nn.Linear(mlp_hidden_size, 2)  # ``1`` for ``true`` and ``0`` for ``false``

        if fs_checkpoint:
            for i in range(fs_checkpoint_start_layer_id, config.num_hidden_layers):
                self.roberta.encoder.layer[i] = checkpoint_wrapper(self.roberta.encoder.layer[i],
                                                                   offload_to_cpu=fs_checkpoint_offload_to_cpu)

        self.init_weights()

        self.init_metric("loss", "acc", "mlm_loss", "mlm_acc", "cls_loss")

        self.mlm_alpha = mlm_alpha
        # The option is added to disable the MLM loss but keep computing the MLM accuracy on the validation set,
        # in order to observe if there is the catastrophic forgetting problem.
        self.mlm_disabled = mlm_disabled

    def forward(
            self,
            input_ids: Tensor,
            attention_mask: Tensor = None,
            token_type_ids: Tensor = None,
            labels: Tensor = None,
            mlm_input_ids: Tensor = None,
            mlm_attention_mask: Tensor = None,
            mlm_labels: Tensor = None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the multiple choice classification loss. Indices should be in ``[0, ...,
            num_choices-1]`` where :obj:`num_choices` is the size of the second dimension of the input tensors. (See
            :obj:`input_ids` above)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

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
        reshaped_logits = logits.view(-1, self.config.num_labels)

        loss = 0.
        mlm_loss = 0.
        cls_loss = 0.
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            cls_loss = loss_fct(reshaped_logits, labels)
            loss = loss + cls_loss

            if mlm_labels is not None and (self.mlm_disabled is False or self.training is False):
                # if mlm_attention_mask is None:
                #     mlm_attention_mask = attention_mask.reshape(reshaped_logits.size(0), num_choices, -1)[:, 0]

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

                if mlm_labels is not None:
                    acc, true_label_num = layers.get_accuracy(mlm_scores, mlm_labels)
                    self.eval_metrics.update("mlm_acc", val=acc, n=true_label_num)
                    self.eval_metrics.update("mlm_loss", val=mlm_loss.item(), n=true_label_num)

        if not return_dict:
            output = (reshaped_logits,) + outputs[2:] + (mlm_loss, cls_loss,)
            return ((loss,) + output) if loss is not None else output

        return SequenceClassificationPreTrainModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            mlm_loss=mlm_loss,
            cls_loss=cls_loss,
        )


class RobertaForMaskedLM(RobertaPreTrainedModel, LogMixin, ABC):
    _keys_to_ignore_on_load_missing = [r"position_ids", r"lm_head.decoder.bias"]

    def __init__(self, config: RobertaConfig,
                 fs_checkpoint: bool = False,
                 fs_checkpoint_offload_to_cpu: bool = False):
        super().__init__(config)

        if config.is_decoder:
            logger.warning(
                "If you want to use `RobertaForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        self.roberta = RobertaModel(config)
        self.lm_head = RobertaLMHead(config)
        self.vocab_size = config.vocab_size

        if fs_checkpoint:
            for i in range(config.num_hidden_layers):
                self.roberta.encoder.layer[i] = checkpoint_wrapper(self.roberta.encoder.layer[i],
                                                                   offload_to_cpu=fs_checkpoint_offload_to_cpu)

        self.init_weights()

        self.init_metric("loss", "acc")

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
            config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``
        kwargs (:obj:`Dict[str, any]`, optional, defaults to `{}`):
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
        prediction_scores = self.lm_head(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.vocab_size), labels.view(-1))

            if not self.training:
                acc, true_label_num = layers.get_accuracy(prediction_scores, labels)
                self.eval_metrics.update("acc", acc, n=true_label_num)
                self.eval_metrics.update("loss", masked_lm_loss.item(), n=true_label_num)

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class RobertaForMultipleChoiceInMLM(RobertaPreTrainedModel, LogMixin, ABC):
    _keys_to_ignore_on_load_missing = [r"position_ids", r"lm_head.decoder.bias"]

    def __init__(self, config: RobertaConfig,
                 fs_checkpoint: bool = False,
                 fs_checkpoint_offload_to_cpu: bool = False):
        super().__init__(config)

        if config.is_decoder:
            logger.warning(
                "If you want to use `RobertaForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        self.roberta = RobertaModel(config)
        self.lm_head = RobertaLMHead(config)
        self.vocab_size = config.vocab_size

        if fs_checkpoint:
            for i in range(config.num_hidden_layers):
                self.roberta.encoder.layer[i] = checkpoint_wrapper(self.roberta.encoder.layer[i],
                                                                   offload_to_cpu=fs_checkpoint_offload_to_cpu)

        self.init_weights()

        self.init_metric("loss", "acc")

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            mlm_labels=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
            config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``
        kwargs (:obj:`Dict[str, any]`, optional, defaults to `{}`):
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
        prediction_scores = self.lm_head(sequence_output)

        seq_len = input_ids.shape[1]
        batch_size = labels.size(0)

        loss_fct = CrossEntropyLoss(ignore_index=-1, reduction="none")
        masked_lm_loss = -loss_fct(prediction_scores.view(-1, self.vocab_size), mlm_labels.view(-1)).reshape(-1, seq_len)

        # batch_size * choice_num
        logits = masked_lm_loss.sum(dim=-1) / (mlm_labels != -1).sum(dim=-1)
        logits = logits.reshape(batch_size, -1)

        loss = None
        if labels is not None:

            loss_fct = CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(logits, labels)

            if not self.training:
                acc, true_label_num = layers.get_accuracy(logits, labels)
                self.eval_metrics.update("acc", acc, n=true_label_num)
                self.eval_metrics.update("loss", loss.item(), n=true_label_num)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return MaskedLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class RobertaForMultipleChoiceForZeroShot(RobertaPreTrainedModel, LogMixin, ABC):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config: RobertaConfig,
                 mlp_hidden_size: int = 768,
                 fs_checkpoint: bool = False,
                 fs_checkpoint_offload_to_cpu: bool = False,
                 fs_checkpoint_start_layer_id: int = 0,
                 freeze_encoder: bool = False,
                 freeze_pooler: bool = False,
                 override_pooler: bool = True):
        super().__init__(config)

        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(p=getattr(config, "pooler_dropout", config.hidden_dropout_prob))
        self.override_pooler = override_pooler

        if self.override_pooler:
            self.pooler = nn.Sequential(
                nn.Linear(config.hidden_size, mlp_hidden_size),
                nn.Tanh()
            )
            self.cls = nn.Linear(mlp_hidden_size, 1)
        else:
            self.n_pooler = nn.Sequential(
                nn.Linear(config.hidden_size, mlp_hidden_size),
                nn.Tanh()
            )
            self.classifier = nn.Linear(mlp_hidden_size, 1)

        if fs_checkpoint:
            for i in range(fs_checkpoint_start_layer_id, config.num_hidden_layers):
                self.roberta.encoder.layer[i] = checkpoint_wrapper(self.roberta.encoder.layer[i],
                                                                   offload_to_cpu=fs_checkpoint_offload_to_cpu)

        self.freeze_pooler = freeze_pooler
        if self.freeze_pooler:
            if self.override_pooler:
                layers.freeze_module(self.pooler)
            else:
                layers.freeze_module(self.n_pooler)

        self.freeze_encoder = freeze_encoder
        if self.freeze_encoder:
            layers.freeze_module(self.roberta)

        self.init_weights()

        self.init_metric("loss", "acc")

    @staticmethod
    def fold_tensor(x: Tensor):
        if x is None:
            return x
        return x.reshape(-1, x.size(-1))

    def forward(
            self,
            input_ids: Tensor,
            attention_mask: Tensor = None,
            token_type_ids: Tensor = None,
            labels: Tensor = None,
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
        num_choices = input_ids.shape[1]

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

        if self.override_pooler:
            logits = self.cls(self.dropout(self.pooler(pooled_output)))
        else:
            logits = self.classifier(self.dropout(self.n_pooler(pooled_output)))
        reshaped_logits = logits.view(-1, num_choices)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            cls_loss = loss_fct(reshaped_logits, labels)
            loss = cls_loss

            if not self.training:
                acc, true_label_num = layers.get_accuracy(reshaped_logits, labels)
                self.eval_metrics.update("acc", val=acc, n=true_label_num)
                self.eval_metrics.update("loss", val=loss.item(), n=true_label_num)

        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class RobertaForMultipleChoicePrompt(RobertaPreTrainedModel, LogMixin, ABC):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config: RobertaConfig,
                 mlp_hidden_size: int = 768,
                 prompt_mlp_hidden_size: int = 768,
                 fs_checkpoint: bool = False,
                 fs_checkpoint_offload_to_cpu: bool = False,
                 fs_checkpoint_maintain_forward_counter: bool = False,
                 fs_checkpoint_start_layer_id: int = 0,
                 freeze_encoder: bool = False,
                 freeze_pooler: bool = False):
        super().__init__(config)

        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.prompt_mlp = nn.Sequential(
            nn.Linear(config.hidden_size, prompt_mlp_hidden_size),
            nn.Tanh(),
            nn.Linear(prompt_mlp_hidden_size, config.hidden_size),
        )

        self.pooler = nn.Sequential(
            nn.Linear(config.hidden_size, mlp_hidden_size),
            nn.Tanh()
        )
        self.cls = nn.Linear(mlp_hidden_size, 1)

        if fs_checkpoint:
            for i in range(fs_checkpoint_start_layer_id, config.num_hidden_layers):
                self.roberta.encoder.layer[i] = checkpoint_wrapper(self.roberta.encoder.layer[i],
                                                                   offload_to_cpu=fs_checkpoint_offload_to_cpu,
                                                                   maintain_forward_counter=fs_checkpoint_maintain_forward_counter)

        self.freeze_pooler = freeze_pooler
        if self.freeze_pooler:
            for param in self.pooler.parameters():
                param.requires_grad = False
        self.freeze_encoder = freeze_encoder
        if self.freeze_encoder:
            for name, param in self.roberta.named_parameters():
                if 'embeddings.word_embeddings' not in name:
                    param.requires_grad = False

        self.init_weights()

        self.init_metric("loss", "acc")

    @staticmethod
    def fold_tensor(x: Tensor):
        if x is None:
            return x
        return x.reshape(-1, x.size(-1))

    def forward(
            self,
            input_ids: Tensor,
            attention_mask: Tensor = None,
            token_type_ids: Tensor = None,
            labels: Tensor = None,
            prefix_pos: Tensor = None,
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
        num_choices = input_ids.shape[1]

        input_ids = self.fold_tensor(input_ids)
        attention_mask = self.fold_tensor(attention_mask)
        token_type_ids = self.fold_tensor(token_type_ids)
        prefix_pos = self.fold_tensor(prefix_pos)

        embed_layer = self.roberta.embeddings.word_embeddings
        input_embeds = embed_layer(input_ids)

        ex_prefix_pos = prefix_pos.unsqueeze(-1).expand(-1, -1, input_embeds.size(-1))
        prefix_embed = torch.gather(input_embeds, index=ex_prefix_pos, dim=1)
        prefix_embed = self.prompt_mlp(prefix_embed)
        input_embeds = torch.scatter(input_embeds, dim=1, index=ex_prefix_pos, src=prefix_embed.to(dtype=input_embeds.dtype))

        if self.freeze_encoder:
            input_embeds = layers.keep_grad_prompt(input_embeds, prefix_pos)

        outputs = self.roberta(
            inputs_embeds=input_embeds,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        pooled_output = outputs[0][:, 0]

        logits = self.cls(self.dropout(self.pooler(pooled_output)))
        reshaped_logits = logits.view(-1, num_choices)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            cls_loss = loss_fct(reshaped_logits, labels)
            loss = cls_loss

            if not self.training:
                acc, true_label_num = layers.get_accuracy(reshaped_logits, labels)
                self.eval_metrics.update("acc", val=acc, n=true_label_num)
                self.eval_metrics.update("loss", val=loss.item(), n=true_label_num)

        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class RobertaForSequenceClassification(RobertaPreTrainedModel, LogMixin, ABC):
    def __init__(self, config: RobertaConfig,
                 fs_checkpoint: bool = False,
                 fs_checkpoint_offload_to_cpu: bool = False,
                 fs_checkpoint_maintain_forward_counter: bool = False,
                 freeze_encoder: bool = False,
                 no_pooler: bool = False):
        super().__init__(config)

        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(p=getattr(config, "pooler_dropout", config.hidden_dropout_prob))
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        if fs_checkpoint:
            for i in range(config.num_hidden_layers):
                self.roberta.encoder.layer[i] = checkpoint_wrapper(self.roberta.encoder.layer[i],
                                                                   offload_to_cpu=fs_checkpoint_offload_to_cpu,
                                                                   maintain_forward_counter=fs_checkpoint_maintain_forward_counter)

        self.no_pooler = no_pooler
        self.freeze_encoder = freeze_encoder
        if freeze_encoder:
            layers.freeze_module(self.roberta)

        self.init_weights()

        self.init_metric("loss", "acc")

    def forward(
            self,
            input_ids: Tensor,
            attention_mask: Tensor = None,
            token_type_ids: Tensor = None,
            labels: Tensor = None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the multiple choice classification loss. Indices should be in ``[0, ...,
            num_choices-1]`` where :obj:`num_choices` is the size of the second dimension of the input tensors. (See
            :obj:`input_ids` above)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        if self.no_pooler:
            pooled_output = outputs[0][:, 0]
        else:
            pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)

        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(logits, labels)

            if not self.training:
                acc, true_label_num = layers.get_accuracy(logits, labels)
                self.eval_metrics.update("acc", val=acc, n=true_label_num)
                self.eval_metrics.update("loss", val=loss.item(), n=true_label_num)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class RobertaForMultipleChoicePath(RobertaForMultipleChoice, ABC):
    def __init__(self, config: RobertaConfig, num_decoder_layers):
        super().__init__(config)

        self.t5_config = T5Config()
        self.t5_config.is_decoder = True
        self.t5_config.is_encoder_decoder = False
        self.t5_config.num_layers = num_decoder_layers

        self.enc_proj = nn.Linear(config.hidden_size, self.t5_config.d_model)

        self.decoder = T5Stack(self.t5_config)
        self.decoder.post_init()

        self.c = nn.Parameter(torch.Tensor(1, 1, 1, self.t5_config.d_model))
        self.c.data.normal_(mean=0.0, std=self.config.initializer_range)
        self.register_buffer("embed_pad", torch.zeros(1, 1, 1, self.t5_config.d_model))

        self.weight_q = nn.Linear(self.t5_config.d_model, self.t5_config.d_model, bias=False)
        self.weight_k = nn.Linear(self.t5_config.d_model, self.t5_config.d_model, bias=False)
        self.weight_o = nn.Linear(self.t5_config.d_model, self.t5_config.d_model, bias=False)

        self.classifier = nn.Linear(self.t5_config.d_model, 1)

        self.model_parallel = False
        self.device_map = None

        self.init_weights()

    def forward(self,
                input_ids: Tensor,
                attention_mask: Tensor = None,
                token_type_ids: Tensor = None,
                op_mask: Tensor = None,
                labels: Tensor = None,
                part_index: Tensor = None,
                part_token_mask: Tensor = None,
                part_occur_mask: Tensor = None,
                part_mask: Tensor = None,
                pos_index: Tensor = None,
                pos_token_mask: Tensor = None,
                pos_occur_mask: Tensor = None,
                pos_mask: Tensor = None,
                part_decoder_input_ids: Tensor = None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                use_cache=None,
                ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        use_cache = use_cache if use_cache is not None else self.t5_config.use_cache
        num_choices = input_ids.shape[1]
        batch_size = input_ids.size(0)
        seq_len = input_ids.size(2)

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
        seq_outputs = self.enc_proj(outputs[0].reshape(batch_size, num_choices, seq_len, -1))

        part_hidden = self.parse_span_rep(seq_outputs, part_index, part_token_mask, part_occur_mask)
        pos_hidden = self.parse_span_rep(seq_outputs, pos_index, pos_token_mask, pos_occur_mask)

        sample_num = part_decoder_input_ids.size(2)
        part_num = part_mask.size(2)
        part_decoder_input_ids = part_decoder_input_ids.reshape(batch_size, num_choices, sample_num * part_num)

        part_decoder_input_embeds = torch.gather(part_hidden, dim=2,
                                                 index=part_decoder_input_ids.unsqueeze(-1).expand(-1, -1, -1, seq_outputs.size(-1)))
        part_decoder_input_embeds = part_decoder_input_embeds.reshape(batch_size, num_choices * sample_num, part_num, -1)

        embed_pad = self.embed_pad.expand(batch_size, num_choices * sample_num, -1, -1)
        part_decoder_input_embeds = torch.cat([part_decoder_input_embeds, embed_pad], dim=2)

        part_c_index = part_mask.sum(dim=2)[:, :, None, None, None].expand(
            -1, -1, sample_num, 1, seq_outputs.size(-1)).reshape(batch_size, num_choices * sample_num, 1, seq_outputs.size(-1))
        c = self.c.expand(batch_size, num_choices * sample_num, 1, -1)

        part_decoder_input_embeds = torch.scatter(part_decoder_input_embeds, dim=2,
                                                  index=part_c_index,
                                                  src=c)

        part_decoder_input_embeds = part_decoder_input_embeds.reshape(batch_size * num_choices * sample_num, part_num + 1, -1)

        pos_num = pos_mask.size(2)
        pos_hidden = pos_hidden.unsqueeze(2).expand(-1, -1, sample_num, -1, -1).reshape(batch_size * num_choices * sample_num,
                                                                                        pos_num, self.t5_config.d_model)
        pos_mask = pos_mask.unsqueeze(2).expand(-1, -1, sample_num, -1).reshape(batch_size * num_choices * sample_num, pos_num)

        decoder_outputs = self.decoder(
            input_ids=None,
            inputs_embeds=part_decoder_input_embeds,
            encoder_hidden_states=pos_hidden,
            encoder_attention_mask=pos_mask,
            use_cache=use_cache,
            return_dict=return_dict,
        )

        hidden = torch.gather(decoder_outputs[0].reshape(batch_size, num_choices * sample_num, part_num + 1, self.t5_config.d_model),
                              dim=2, index=part_c_index).reshape(batch_size, num_choices, sample_num, self.t5_config.d_model)

        w_q = self.weight_q(seq_outputs[:, :, 0])
        w_k = self.weight_k(hidden)
        sim = torch.einsum("bnd,bnsd->bns", w_q, w_k).softmax(dim=-1)
        hidden = self.weight_o(torch.einsum("bns,bnsd->bnd", sim, hidden))

        pooled_output = self.dropout(hidden)

        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices) + (1.0 - op_mask.to(logits.dtype)) * -1e4

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(reshaped_logits, labels)

            if not self.training:
                acc, true_label_num = layers.get_accuracy(reshaped_logits, labels)
                self.eval_metrics.update("acc", val=acc, n=true_label_num)
                self.eval_metrics.update("loss", val=loss.item(), n=true_label_num)

        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @staticmethod
    def parse_span_rep(seq_outputs: Tensor, index: Tensor, token_mask: Tensor, occur_mask: Tensor):

        # print(f"Index: {index.size()}")
        # print(f"Token mask: {token_mask.size()}")
        # print(f"Occur mask: {occur_mask.size()}")
        # print(f"Seq outputs: {seq_outputs.size()}")

        batch_size, option_num, max_span_num, max_span_occur_num, max_span_len = index.size()
        assert batch_size == seq_outputs.size(0)
        assert option_num == seq_outputs.size(1)
        h = seq_outputs.size(-1)

        flat_index = index.reshape(batch_size, option_num, -1, 1)
        flat_rep = torch.gather(seq_outputs, dim=2, index=flat_index.expand(-1, -1, -1, h))
        flat_rep = flat_rep.reshape(batch_size, option_num, max_span_num, max_span_occur_num, max_span_len, h)

        # print(f"Flat rep: {flat_rep.size()}")

        # Sub-work pooling
        true_token_num = token_mask.sum(dim=4, keepdim=True).to(flat_rep.dtype)
        true_token_num[true_token_num == 0] = 1.
        flat_rep = flat_rep.sum(dim=4) / true_token_num  # (batch_size, option_num, max_span_num, max_span_occur_num, h)

        # Occurrence pooling
        true_occur_num = occur_mask.sum(dim=3, keepdim=True).to(flat_rep.dtype)
        true_occur_num[true_occur_num == 0] = 1.
        flat_rep = flat_rep.sum(dim=3) / true_occur_num  # (batch_size, option_num, max_span_num, h)

        return flat_rep


class RobertaForMultipleChoicePathV2(RobertaForMultipleChoice, ABC):
    def __init__(self, config: RobertaConfig, num_decoder_layers: int, num_extra_encoder_layers: int):
        super().__init__(config)

        self.t5_config = T5Config()
        self.t5_config.is_decoder = True
        self.t5_config.is_encoder_decoder = False
        self.t5_config.num_layers = num_decoder_layers

        self.enc_proj = nn.Linear(config.hidden_size, self.t5_config.d_model)

        self.decoder = T5Stack(self.t5_config)
        self.decoder.post_init()

        self.dec_proj = nn.Linear(self.t5_config.d_model, config.hidden_size)

        ex_enc_config = copy.deepcopy(config)
        ex_enc_config.num_hidden_layers = num_extra_encoder_layers
        self.ex_enc_config = ex_enc_config
        self.ex_enc = RobertaEncoder(self.ex_enc_config)

        self.classifier = nn.Linear(config.hidden_size, 1)

        self.model_parallel = False
        self.device_map = None

        self.init_weights()

    def forward(self,
                input_ids: Tensor,
                attention_mask: Tensor = None,
                token_type_ids: Tensor = None,
                op_mask: Tensor = None,
                labels: Tensor = None,
                part_index: Tensor = None,
                part_token_mask: Tensor = None,
                part_occur_mask: Tensor = None,
                part_mask: Tensor = None,
                pos_index: Tensor = None,
                pos_token_mask: Tensor = None,
                pos_occur_mask: Tensor = None,
                pos_mask: Tensor = None,
                part_decoder_input_ids: Tensor = None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                use_cache=None,
                ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        use_cache = use_cache if use_cache is not None else self.t5_config.use_cache
        num_choices = input_ids.shape[1]
        batch_size = input_ids.size(0)
        seq_len = input_ids.size(2)

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
        seq_outputs = self.enc_proj(outputs[0].reshape(batch_size, num_choices, seq_len, -1))

        # (batch_size, option_num, max_span_num, h)
        part_hidden = self.parse_span_rep(seq_outputs, part_index, part_token_mask, part_occur_mask)
        pos_hidden = self.parse_span_rep(seq_outputs, pos_index, pos_token_mask, pos_occur_mask)

        sample_num = part_decoder_input_ids.size(2)
        part_num = part_mask.size(2)
        part_decoder_input_ids = part_decoder_input_ids.reshape(batch_size, num_choices, sample_num * part_num)

        part_decoder_input_embeds = torch.gather(part_hidden, dim=2,
                                                 index=part_decoder_input_ids.unsqueeze(-1).expand(-1, -1, -1, seq_outputs.size(-1)))
        part_decoder_input_embeds = part_decoder_input_embeds.reshape(batch_size * num_choices * sample_num, part_num, -1)

        pos_num = pos_mask.size(2)
        pos_hidden = pos_hidden.unsqueeze(2).expand(-1, -1, sample_num, -1, -1).reshape(batch_size * num_choices * sample_num,
                                                                                        pos_num, self.t5_config.d_model)
        pos_mask = pos_mask.unsqueeze(2).expand(-1, -1, sample_num, -1).reshape(batch_size * num_choices * sample_num, pos_num)

        decoder_outputs = self.decoder(
            input_ids=None,
            inputs_embeds=part_decoder_input_embeds,
            encoder_hidden_states=pos_hidden,
            encoder_attention_mask=pos_mask,
            use_cache=use_cache,
            return_dict=return_dict,
        )
        part_assign_hidden = self.dec_proj(decoder_outputs[0].reshape(batch_size * num_choices, sample_num * part_num, -1))

        hidden_states = torch.cat([outputs[0], part_assign_hidden], dim=1)
        attention_mask = torch.cat([
            attention_mask,
            attention_mask.new_ones(batch_size * num_choices, sample_num * part_num)
        ], dim=1)
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, hidden_states.size()[:-1],
                                                                                 hidden_states.device)
        head_mask = self.get_head_mask(None, self.ex_enc_config.num_hidden_layers)

        ex_encoder_outputs = self.ex_enc(
            hidden_states,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = self.dropout(ex_encoder_outputs[0][:, 0])
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices) + (1.0 - op_mask.to(logits.dtype)) * -1e4

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(reshaped_logits, labels)

            if not self.training:
                acc, true_label_num = layers.get_accuracy(reshaped_logits, labels)
                self.eval_metrics.update("acc", val=acc, n=true_label_num)
                self.eval_metrics.update("loss", val=loss.item(), n=true_label_num)

        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @staticmethod
    def parse_span_rep(seq_outputs: Tensor, index: Tensor, token_mask: Tensor, occur_mask: Tensor):

        # print(f"Index: {index.size()}")
        # print(f"Token mask: {token_mask.size()}")
        # print(f"Occur mask: {occur_mask.size()}")
        # print(f"Seq outputs: {seq_outputs.size()}")

        batch_size, option_num, max_span_num, max_span_occur_num, max_span_len = index.size()
        assert batch_size == seq_outputs.size(0)
        assert option_num == seq_outputs.size(1)
        h = seq_outputs.size(-1)

        flat_index = index.reshape(batch_size, option_num, -1, 1)
        flat_rep = torch.gather(seq_outputs, dim=2, index=flat_index.expand(-1, -1, -1, h))
        flat_rep = flat_rep.reshape(batch_size, option_num, max_span_num, max_span_occur_num, max_span_len, h)

        # print(f"Flat rep: {flat_rep.size()}")

        # Sub-work pooling
        true_token_num = token_mask.sum(dim=4, keepdim=True).to(flat_rep.dtype)
        true_token_num[true_token_num == 0] = 1.
        flat_rep = flat_rep.sum(dim=4) / true_token_num  # (batch_size, option_num, max_span_num, max_span_occur_num, h)

        # Occurrence pooling
        true_occur_num = occur_mask.sum(dim=3, keepdim=True).to(flat_rep.dtype)
        true_occur_num[true_occur_num == 0] = 1.
        flat_rep = flat_rep.sum(dim=3) / true_occur_num  # (batch_size, option_num, max_span_num, h)

        return flat_rep


class RobertaForMultipleChoicePathV3(RobertaForMultipleChoice, ABC):
    def __init__(self, config: RobertaConfig, num_decoder_layers: int, num_extra_encoder_layers: int):
        super().__init__(config)

        self.t5_config = T5Config()
        self.t5_config.is_decoder = True
        self.t5_config.is_encoder_decoder = False
        self.t5_config.num_layers = num_decoder_layers

        self.enc_proj = nn.Linear(config.hidden_size, self.t5_config.d_model)

        self.decoder = T5Stack(self.t5_config)
        self.decoder.post_init()

        self.dec_proj = nn.Linear(self.t5_config.d_model, config.hidden_size)

        ex_enc_config = copy.deepcopy(config)
        ex_enc_config.num_hidden_layers = num_extra_encoder_layers
        self.ex_enc_config = ex_enc_config
        self.ex_enc = RobertaEncoder(self.ex_enc_config)

        self.linear1 = nn.Linear(config.hidden_size, 1)

        # self.classifier = nn.Linear(config.hidden_size, 1)
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, 1),
        )

        self.model_parallel = False
        self.device_map = None

        self.init_weights()

    def forward(self,
                input_ids: Tensor,
                attention_mask: Tensor = None,
                token_type_ids: Tensor = None,
                op_mask: Tensor = None,
                labels: Tensor = None,
                part_index: Tensor = None,
                part_token_mask: Tensor = None,
                part_occur_mask: Tensor = None,
                part_mask: Tensor = None,
                pos_index: Tensor = None,
                pos_token_mask: Tensor = None,
                pos_occur_mask: Tensor = None,
                pos_mask: Tensor = None,
                part_decoder_input_ids: Tensor = None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                use_cache=None,
                ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        use_cache = use_cache if use_cache is not None else self.t5_config.use_cache
        num_choices = input_ids.shape[1]
        batch_size = input_ids.size(0)
        seq_len = input_ids.size(2)

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
        seq_outputs = self.enc_proj(outputs[0].reshape(batch_size, num_choices, seq_len, -1))

        # (batch_size, option_num, max_span_num, h)
        part_hidden = self.parse_span_rep(seq_outputs, part_index, part_token_mask, part_occur_mask)
        pos_hidden = self.parse_span_rep(seq_outputs, pos_index, pos_token_mask, pos_occur_mask)

        sample_num = part_decoder_input_ids.size(2)
        part_num = part_mask.size(2)
        part_decoder_input_ids = part_decoder_input_ids.reshape(batch_size, num_choices, sample_num * part_num)

        part_decoder_input_embeds = torch.gather(part_hidden, dim=2,
                                                 index=part_decoder_input_ids.unsqueeze(-1).expand(-1, -1, -1, seq_outputs.size(-1)))
        part_decoder_input_embeds = part_decoder_input_embeds.reshape(batch_size * num_choices * sample_num, part_num, -1)

        pos_num = pos_mask.size(2)
        pos_hidden = pos_hidden.unsqueeze(2).expand(-1, -1, sample_num, -1, -1).reshape(batch_size * num_choices * sample_num,
                                                                                        pos_num, self.t5_config.d_model)
        pos_mask = pos_mask.unsqueeze(2).expand(-1, -1, sample_num, -1).reshape(batch_size * num_choices * sample_num, pos_num)

        decoder_outputs = self.decoder(
            input_ids=None,
            inputs_embeds=part_decoder_input_embeds,
            encoder_hidden_states=pos_hidden,
            encoder_attention_mask=pos_mask,
            use_cache=use_cache,
            return_dict=return_dict,
        )
        # part_assign_hidden = self.dec_proj(decoder_outputs[0].reshape(batch_size * num_choices, sample_num * part_num, -1))
        part_assign_hidden = self.dec_proj(decoder_outputs[0])
        extended_hidden_states = outputs[0].unsqueeze(1).expand(-1, sample_num, -1, -1).reshape(-1, seq_len, outputs[0].size(-1))
        extended_hidden_states = torch.cat([extended_hidden_states, part_assign_hidden], dim=1)

        # hidden_states = torch.cat([outputs[0], part_assign_hidden], dim=1)
        attention_mask = torch.cat([
            attention_mask.unsqueeze(1).expand(-1, sample_num, -1).reshape(-1, seq_len),
            attention_mask.new_ones(batch_size * num_choices * sample_num, part_num)
        ], dim=1)
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, extended_hidden_states.size()[:-1],
                                                                                 extended_hidden_states.device)
        head_mask = self.get_head_mask(None, self.ex_enc_config.num_hidden_layers)

        ex_encoder_outputs = self.ex_enc(
            extended_hidden_states,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        alpha = torch.softmax(self.linear1(ex_encoder_outputs[0][:, 0]).reshape(batch_size * num_choices, sample_num), dim=-1)
        weighted_hidden = torch.einsum("bs,bsh->bh", alpha, ex_encoder_outputs[0][:, 0])

        pooled_output = self.dropout(weighted_hidden)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices) + (1.0 - op_mask.to(logits.dtype)) * -1e4

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(reshaped_logits, labels)

            if not self.training:
                acc, true_label_num = layers.get_accuracy(reshaped_logits, labels)
                self.eval_metrics.update("acc", val=acc, n=true_label_num)
                self.eval_metrics.update("loss", val=loss.item(), n=true_label_num)

        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @staticmethod
    def parse_span_rep(seq_outputs: Tensor, index: Tensor, token_mask: Tensor, occur_mask: Tensor):

        # print(f"Index: {index.size()}")
        # print(f"Token mask: {token_mask.size()}")
        # print(f"Occur mask: {occur_mask.size()}")
        # print(f"Seq outputs: {seq_outputs.size()}")

        batch_size, option_num, max_span_num, max_span_occur_num, max_span_len = index.size()
        assert batch_size == seq_outputs.size(0)
        assert option_num == seq_outputs.size(1)
        h = seq_outputs.size(-1)

        flat_index = index.reshape(batch_size, option_num, -1, 1)
        flat_rep = torch.gather(seq_outputs, dim=2, index=flat_index.expand(-1, -1, -1, h))
        flat_rep = flat_rep.reshape(batch_size, option_num, max_span_num, max_span_occur_num, max_span_len, h)

        # print(f"Flat rep: {flat_rep.size()}")

        # Sub-work pooling
        true_token_num = token_mask.sum(dim=4, keepdim=True).to(flat_rep.dtype)
        true_token_num[true_token_num == 0] = 1.
        flat_rep = flat_rep.sum(dim=4) / true_token_num  # (batch_size, option_num, max_span_num, max_span_occur_num, h)

        # Occurrence pooling
        true_occur_num = occur_mask.sum(dim=3, keepdim=True).to(flat_rep.dtype)
        true_occur_num[true_occur_num == 0] = 1.
        flat_rep = flat_rep.sum(dim=3) / true_occur_num  # (batch_size, option_num, max_span_num, h)

        return flat_rep


class RobertaForMultipleChoicePreTrainWPathGenV1(RobertaForMultipleChoiceForPreTrain, ABC):
    def __init__(self, config: RobertaConfig,
                 num_decoder_layers: int,
                 rel_vocab: str = None,
                 rel_vocab_size: int = None,
                 mlp_hidden_size: int = 768,
                 mlm_alpha: float = 1.0,
                 mlm_disabled: bool = False,
                 fs_checkpoint: bool = False,
                 fs_checkpoint_offload_to_cpu: bool = False,
                 fs_checkpoint_start_layer_id: int = 0,
                 decoder_config_path: str = None,
                 rel_gen_coff: float = 1.0,
                 ctr_coff: float = 1.0, ):
        super().__init__(config, mlp_hidden_size, mlm_alpha, mlm_disabled,
                         fs_checkpoint, fs_checkpoint_offload_to_cpu, fs_checkpoint_start_layer_id)

        if rel_vocab is not None:
            self.rel_vocab = pickle.load(open(rel_vocab, "rb"))
            self.rel_vocab_size = len(set(list(self.rel_vocab.values())))
            self.unk_token_id = -2
            for token, token_id in self.rel_vocab.items():
                if token == "<unk>":
                    self.unk_token_id = token_id
                    break
                assert token_id < self.rel_vocab_size
        elif rel_vocab_size is not None:
            self.rel_vocab_size = rel_vocab_size
            self.unk_token_id = -2
        else:
            raise RuntimeError()
        self.eos_token_id = self.rel_vocab_size
        self.pad_token_id = self.rel_vocab_size + 1

        self.t5_config = T5Config() if decoder_config_path is None else T5Config.from_pretrained(decoder_config_path)
        self.t5_config.is_decoder = True
        self.t5_config.is_encoder_decoder = False
        self.t5_config.add_cross_attention = True
        self.t5_config.num_layers = num_decoder_layers
        self.t5_config.eos_token_id = self.eos_token_id
        self.t5_config.pad_token_id = self.pad_token_id

        self.enc_proj = nn.Linear(config.hidden_size, self.t5_config.d_model)

        self.rel_embed = nn.Embedding(self.rel_vocab_size + 2, self.t5_config.d_model)
        logger.info(f"Relation embedding size: {self.rel_embed.weight.size()}")
        self.decoder = T5Stack(self.t5_config, self.rel_embed)
        self.decoder.post_init()

        self.seq2seq_head = nn.Linear(self.t5_config.d_model, self.rel_vocab_size + 2, bias=False)

        self.rel_gen_coff = rel_gen_coff
        self.ctr_coff = ctr_coff

        self.init_weights()
        self.tie_weights()
        self.init_metric("loss", "acc", "mlm_loss", "mlm_acc", "cls_loss", "path_gen_acc", "path_gen_loss")

    def get_input_embeddings(self):
        return self.rel_embed

    def set_input_embeddings(self, new_embeddings):
        self.rel_embed = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def set_output_embeddings(self, new_embeddings):
        self.seq2seq_head = new_embeddings

    def get_output_embeddings(self):
        return self.seq2seq_head

    def forward(
            self,
            input_ids: Tensor,
            attention_mask: Tensor = None,
            token_type_ids: Tensor = None,
            labels: Tensor = None,
            mlm_input_ids: Tensor = None,
            mlm_attention_mask: Tensor = None,
            rel_labels: Tensor = None,
            mlm_labels: Tensor = None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            input_ids_dropout: Tensor = None,
            attention_mask_dropout: Tensor = None,
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

        if rel_labels.size(0) == batch_size * num_choices:
            rel_labels = rel_labels.reshape(batch_size, num_choices, -1)[:, 0].contiguous()

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

        # Seq2Seq
        if input_ids_dropout is not None:
            dropout_outputs = self.roberta(
                input_ids_dropout,
                attention_mask=attention_mask_dropout,
                return_dict=return_dict,
            )
            encoder_hidden_states = self.enc_proj(dropout_outputs[0])
            encoder_attention_mask = attention_mask_dropout
        else:
            encoder_hidden_states = self.enc_proj(outputs[0].reshape(batch_size, num_choices, seq_len, -1)[:, 0])
            encoder_attention_mask = attention_mask.reshape(batch_size, num_choices, seq_len)[:, 0]

        decoder_input_ids = self._shift_right(rel_labels)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            return_dict=return_dict
        )

        lm_logits = self.seq2seq_head(decoder_outputs[0])

        loss = 0.
        mlm_loss = 0.
        cls_loss = 0.
        path_gen_loss = 0.
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            cls_loss = self.ctr_coff * loss_fct(reshaped_logits, labels)
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
            path_gen_loss = self.rel_gen_coff * loss_fct(lm_logits.view(-1, lm_logits.size(-1)), rel_labels.view(-1))
            tmp_a, tmp_b = layers.get_accuracy(lm_logits, rel_labels)
            if tmp_b > 0:
                loss = loss + path_gen_loss
            else:
                path_gen_loss = 0.

            if not self.training:
                acc, true_label_num = layers.get_accuracy(reshaped_logits, labels)
                self.eval_metrics.update("acc", val=acc, n=true_label_num)
                self.eval_metrics.update("loss", val=loss.item(), n=true_label_num)
                self.eval_metrics.update("cls_loss", val=cls_loss.item(), n=true_label_num)

                acc, true_label_num = layers.get_accuracy(lm_logits, rel_labels)
                self.eval_metrics.update("path_gen_acc", val=acc, n=true_label_num)
                self.eval_metrics.update("path_gen_loss", val=path_gen_loss, n=true_label_num)

                if mlm_labels is not None:
                    acc, true_label_num = layers.get_accuracy(mlm_scores, mlm_labels)
                    self.eval_metrics.update("mlm_acc", val=acc, n=true_label_num)
                    self.eval_metrics.update("mlm_loss", val=mlm_loss.item(), n=true_label_num)

        if not return_dict:
            output = (reshaped_logits,) + outputs[2:] + (mlm_loss, cls_loss,)
            return ((loss,) + output) if loss is not None else output

        return MultipleChoicePreTrainModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            mlm_loss=mlm_loss,
            cls_loss=cls_loss,
            path_gen_loss=path_gen_loss,
        )

    def _shift_right(self, input_ids):
        decoder_start_token_id = pad_token_id = self.pad_token_id

        assert decoder_start_token_id is not None, (
            "self.model.config.decoder_start_token_id has to be defined. In T5 it is usually set to the pad_token_id."
            " See T5 docs for more information"
        )

        # shift inputs to the right
        if is_torch_fx_proxy(input_ids):
            # Item assignment is not supported natively for proxies.
            shifted_input_ids = torch.full(input_ids.shape[:-1] + (1,), decoder_start_token_id)
            shifted_input_ids = torch.cat([shifted_input_ids, input_ids[..., :-1]], dim=-1)
        else:
            shifted_input_ids = input_ids.new_zeros(input_ids.shape)
            shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
            shifted_input_ids[..., 0] = decoder_start_token_id

        assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -1, pad_token_id)

        assert torch.all(shifted_input_ids >= 0).item(), "Verify that `shifted_input_ids` has only positive values"

        return shifted_input_ids


class RelDecoderHead(nn.Module):
    def __init__(self, input_size, hidden_size, vocab_size):
        super().__init__()
        self.dense = nn.Linear(input_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)

        self.decoder = nn.Linear(hidden_size, vocab_size)
        self.bias = nn.Parameter(torch.zeros(vocab_size))
        self.decoder.bias = self.bias

    def forward(self, x):
        x = self.dense(x)
        x = gelu(x)
        x = self.layer_norm(x)

        x = self.decoder(x)
        return x


def retrieve_rel_weights_from_model_buffer(model_path, key: str = "quantizer.embed"):
    state_dict = torch.load(model_path, map_location="cpu")
    return state_dict[key]


class RobertaForMultipleChoicePreTrainWPathGenV2(RobertaForMultipleChoiceForPreTrain, ABC):
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
                 rel_gen_coff: float = 1.0, ):
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

        self.attn_pooler = nn.Linear(config.hidden_size, 1)
        self._init_weights(self.attn_pooler)

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
            self.rel_decoder = RelDecoderHead(config.hidden_size, rel_emb_weights.size(1), self.rel_vocab_size)

            self._init_weights(self.rel_decoder)

            self.rel_decoder.decoder.weight.data.copy_(rel_emb_weights)
            assert self.rel_decoder.decoder.weight.requires_grad
        else:
            self.rel_decoder = RobertaLMHead(rel_decoding_config)
            self._init_weights(self.rel_decoder)

        self.rel_gen_coff = rel_gen_coff

        self.init_metric("loss", "acc", "mlm_loss", "mlm_acc", "cls_loss", "path_gen_acc", "path_gen_loss")

    def forward(
            self,
            input_ids: Tensor,
            attention_mask: Tensor = None,
            token_type_ids: Tensor = None,
            labels: Tensor = None,
            mlm_input_ids: Tensor = None,
            mlm_attention_mask: Tensor = None,
            rel_labels: Tensor = None,
            mlm_labels: Tensor = None,
            sent_token_index: Tensor = None,
            sent_token_mask: Tensor = None,
            sent_mask: Tensor = None,
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

        pos_seq_hidden = outputs[0].reshape(batch_size, num_choices, seq_len, -1)[:, 0]
        sent_num, sent_len = sent_token_index.size(1), sent_token_index.size(2)
        h = pos_seq_hidden.size(-1)
        sent_token_hidden = torch.gather(pos_seq_hidden, dim=1,
                                         index=sent_token_index.unsqueeze(-1).expand(-1, -1, -1, h).reshape(
                                             batch_size, sent_num * sent_len, h)).reshape(batch_size, sent_num, sent_len, h)
        sent_token_scores = self.attn_pooler(sent_token_hidden).squeeze(-1)
        sent_token_scores = (1 - sent_token_mask) * torch.finfo(self.dtype).min + sent_token_scores
        sent_hidden = self.dropout(torch.einsum("bst,bsth->bsh", torch.softmax(sent_token_scores, dim=2), sent_token_hidden))
        sent_rel_logits = self.rel_decoder(sent_hidden)

        loss = 0.
        mlm_loss = 0.
        cls_loss = 0.
        path_gen_loss = 0.
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

            # print(rel_labels.size(), sent_rel_logits.size())
            rel_labels[rel_labels == self.unk_token_id] = -1
            rel_labels = rel_labels[:, :-1]  # remove <bos> token
            rel_labels[rel_labels == self.eos_token_id] = -1
            path_gen_loss = self.rel_gen_coff * loss_fct(sent_rel_logits.reshape(-1, sent_rel_logits.size(-1)), rel_labels.reshape(-1))
            loss = loss + path_gen_loss

            if not self.training:
                acc, true_label_num = layers.get_accuracy(reshaped_logits, labels)
                self.eval_metrics.update("acc", val=acc, n=true_label_num)
                self.eval_metrics.update("loss", val=loss.item(), n=true_label_num)
                self.eval_metrics.update("cls_loss", val=cls_loss.item(), n=true_label_num)

                acc, true_label_num = layers.get_accuracy(sent_rel_logits, rel_labels)
                self.eval_metrics.update("path_gen_acc", val=acc, n=true_label_num)
                self.eval_metrics.update("path_gen_loss", val=path_gen_loss.item(), n=true_label_num)

                if mlm_labels is not None:
                    acc, true_label_num = layers.get_accuracy(mlm_scores, mlm_labels)
                    self.eval_metrics.update("mlm_acc", val=acc, n=true_label_num)
                    self.eval_metrics.update("mlm_loss", val=mlm_loss.item(), n=true_label_num)

        if not return_dict:
            output = (reshaped_logits,) + outputs[2:] + (mlm_loss, cls_loss,)
            return ((loss,) + output) if loss is not None else output

        return MultipleChoicePreTrainModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            mlm_loss=mlm_loss,
            cls_loss=cls_loss,
            path_gen_loss=path_gen_loss,
        )


class RobertaPathGenV1PreTrain(RobertaForMultipleChoiceForPreTrain, ABC):
    def __init__(self, config: RobertaConfig,
                 num_decoder_layers: int,
                 rel_vocab: str,
                 mlp_hidden_size: int = 768,
                 mlm_alpha: float = 1.0,
                 mlm_disabled: bool = False,
                 fs_checkpoint: bool = False,
                 fs_checkpoint_offload_to_cpu: bool = False,
                 fs_checkpoint_start_layer_id: int = 0,
                 decoder_config_path: str = None):
        super().__init__(config, mlp_hidden_size, mlm_alpha, mlm_disabled,
                         fs_checkpoint, fs_checkpoint_offload_to_cpu, fs_checkpoint_start_layer_id)

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

        self.t5_config = T5Config() if decoder_config_path is None else T5Config.from_pretrained(decoder_config_path)
        self.t5_config.is_decoder = True
        self.t5_config.is_encoder_decoder = False
        self.t5_config.add_cross_attention = True
        self.t5_config.num_layers = num_decoder_layers
        self.t5_config.eos_token_id = self.eos_token_id
        self.t5_config.pad_token_id = self.pad_token_id

        self.enc_proj = nn.Linear(config.hidden_size, self.t5_config.d_model)

        self.rel_embed = nn.Embedding(self.rel_vocab_size + 2, self.t5_config.d_model)
        logger.info(f"Relation embedding size: {self.rel_embed.weight.size()}")
        self.decoder = T5Stack(self.t5_config, self.rel_embed)
        self.decoder.post_init()

        self.seq2seq_head = nn.Linear(self.t5_config.d_model, self.rel_vocab_size + 2, bias=False)

        self.init_weights()
        self.tie_weights()
        self.init_metric("loss", "acc", "mlm_loss", "mlm_acc", "cls_loss", "path_gen_acc", "path_gen_loss")

    def get_input_embeddings(self):
        return self.rel_embed

    def set_input_embeddings(self, new_embeddings):
        self.rel_embed = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def set_output_embeddings(self, new_embeddings):
        self.seq2seq_head = new_embeddings

    def get_output_embeddings(self):
        return self.seq2seq_head

    def forward(
            self,
            input_ids: Tensor,
            attention_mask: Tensor = None,
            token_type_ids: Tensor = None,
            labels: Tensor = None,
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
        # batch_size, num_choices, seq_len = input_ids.size()
        batch_size, seq_len = input_ids.size()

        # input_ids = self.fold_tensor(input_ids)
        # attention_mask = self.fold_tensor(attention_mask)
        # token_type_ids = self.fold_tensor(token_type_ids)
        # We just overlook the multiple choice data.
        # input_ids = input_ids[:, 0]
        # attention_mask = attention_mask[:, 0]
        # token_type_ids = token_type_ids[:, 0] if token_type_ids is not None else None

        outputs = self.roberta(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        pooled_output = outputs[0][:, 0]

        # logits = self.cls(self.dropout(self.pooler(pooled_output)))
        # reshaped_logits = logits.view(-1, num_choices)

        # Seq2Seq
        encoder_hidden_states = self.enc_proj(outputs[0])
        # encoder_attention_mask = attention_mask.reshape(batch_size, num_choices, seq_len)[:, 0]
        decoder_input_ids = self._shift_right(rel_labels)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask,
            return_dict=return_dict
        )

        lm_logits = self.seq2seq_head(decoder_outputs[0])

        loss = 0.
        mlm_loss = 0.
        cls_loss = 0.
        path_gen_loss = 0.
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            # cls_loss = loss_fct(reshaped_logits, labels)
            # loss = loss + cls_loss

            if mlm_labels is not None and (self.mlm_disabled is False or self.training is False):
                if mlm_attention_mask is None:
                    # mlm_attention_mask = attention_mask.reshape(reshaped_logits.size(0), num_choices, -1)[:, 0]
                    mlm_attention_mask = attention_mask

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
            path_gen_loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), rel_labels.view(-1))
            loss = loss + path_gen_loss

            if not self.training:
                # acc, true_label_num = layers.get_accuracy(reshaped_logits, labels)
                # self.eval_metrics.update("acc", val=acc, n=true_label_num)
                # self.eval_metrics.update("loss", val=loss.item(), n=true_label_num)
                # self.eval_metrics.update("cls_loss", val=cls_loss.item(), n=true_label_num)

                acc, true_label_num = layers.get_accuracy(lm_logits, rel_labels)
                self.eval_metrics.update("path_gen_acc", val=acc, n=true_label_num)
                self.eval_metrics.update("path_gen_loss", val=path_gen_loss.item(), n=true_label_num)
                self.eval_metrics.update("loss", val=loss.item(), n=true_label_num)

                if mlm_labels is not None:
                    acc, true_label_num = layers.get_accuracy(mlm_scores, mlm_labels)
                    self.eval_metrics.update("mlm_acc", val=acc, n=true_label_num)
                    self.eval_metrics.update("mlm_loss", val=mlm_loss.item(), n=true_label_num)

        if not return_dict:
            output = (lm_logits,) + outputs[2:] + (mlm_loss, cls_loss,)
            return ((loss,) + output) if loss is not None else output

        return MultipleChoicePreTrainModelOutput(
            loss=loss,
            # logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            mlm_loss=mlm_loss,
            cls_loss=cls_loss,
            path_gen_loss=path_gen_loss,
        )

    def _shift_right(self, input_ids):
        decoder_start_token_id = pad_token_id = self.pad_token_id

        assert decoder_start_token_id is not None, (
            "self.model.config.decoder_start_token_id has to be defined. In T5 it is usually set to the pad_token_id."
            " See T5 docs for more information"
        )

        # shift inputs to the right
        if is_torch_fx_proxy(input_ids):
            # Item assignment is not supported natively for proxies.
            shifted_input_ids = torch.full(input_ids.shape[:-1] + (1,), decoder_start_token_id)
            shifted_input_ids = torch.cat([shifted_input_ids, input_ids[..., :-1]], dim=-1)
        else:
            shifted_input_ids = input_ids.new_zeros(input_ids.shape)
            shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
            shifted_input_ids[..., 0] = decoder_start_token_id

        assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -1, pad_token_id)

        assert torch.all(shifted_input_ids >= 0).item(), "Verify that `shifted_input_ids` has only positive values"

        return shifted_input_ids


class RobertaForMultipleChoicePreTrainWPathGenV3(RobertaForMultipleChoiceForPreTrain, ABC):
    def __init__(self, config: RobertaConfig,
                 num_decoder_layers: int,
                 rel_vocab: str,
                 mlp_hidden_size: int = 768,
                 mlm_alpha: float = 1.0,
                 mlm_disabled: bool = False,
                 fs_checkpoint: bool = False,
                 fs_checkpoint_offload_to_cpu: bool = False,
                 fs_checkpoint_start_layer_id: int = 0):
        super().__init__(config, mlp_hidden_size, mlm_alpha, mlm_disabled,
                         fs_checkpoint, fs_checkpoint_offload_to_cpu, fs_checkpoint_start_layer_id)

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

        self.decoder_config = copy.deepcopy(config)
        self.decoder_config.is_decoder = True
        self.decoder_config.add_cross_attention = True
        self.decoder_config.num_hidden_layers = num_decoder_layers
        self.decoder_config.tie_word_embeddings = True

        # self.rel_embed = nn.Embedding(self.rel_vocab_size + 2, config.hidden_size)
        self.rel_position_embed = nn.Embedding(10, config.hidden_size)
        self.rel_embed_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # logger.info(f"Relation embedding size: {self.rel_embed.weight.size()}")
        self.decoder = RobertaEncoder(self.decoder_config)
        # self.decoder.post_init()

        self.seq2seq_head = nn.Linear(config.hidden_size, self.rel_vocab_size)
        logger.info(f"Relation embedding size: {self.seq2seq_head.weight.size()}")

        self.init_weights()
        # self.tie_weights()
        self.init_metric("loss", "acc", "mlm_loss", "mlm_acc", "cls_loss", "path_gen_acc", "path_gen_loss")

    # def get_position_ids(self):
    #     return self.roberta.embeddings.position_ids

    # def get_input_embeddings(self):
    #     return self.rel_embed

    # def set_input_embeddings(self, new_embeddings):
    #     self.rel_embed = new_embeddings
    #     self.encoder.set_input_embeddings(new_embeddings)
    #     self.decoder.set_input_embeddings(new_embeddings)

    # def set_output_embeddings(self, new_embeddings):
    #     self.seq2seq_head = new_embeddings

    # def get_output_embeddings(self):
    #     return self.seq2seq_head

    def forward(
            self,
            input_ids: Tensor,
            attention_mask: Tensor = None,
            token_type_ids: Tensor = None,
            labels: Tensor = None,
            mlm_input_ids: Tensor = None,
            mlm_attention_mask: Tensor = None,
            rel_labels: Tensor = None,
            mlm_labels: Tensor = None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            input_ids_dropout: Tensor = None,
            attention_mask_dropout: Tensor = None,
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

        input_shape = rel_labels.size()
        rel_num = input_shape[1]
        rel_position_ids = torch.arange(rel_num, dtype=torch.long, device=pooled_output.device).unsqueeze(0).expand(input_shape)
        rel_position_embeddings = self.rel_position_embed(rel_position_ids)
        decoder_input_embeds = pooled_output.reshape(batch_size, num_choices, -1)[:, :1].expand(-1, rel_num, -1)
        decoder_input_embeds = self.dropout(self.rel_embed_layer_norm(decoder_input_embeds + rel_position_embeddings))

        rel_attention_mask = self.invert_attention_mask(rel_labels <= -1)

        # Seq2Seq
        if input_ids_dropout is not None:
            dropout_outputs = self.roberta(
                input_ids_dropout,
                attention_mask=attention_mask_dropout,
                return_dict=return_dict,
            )
            encoder_hidden_states = dropout_outputs[0]
            encoder_attention_mask = attention_mask_dropout
        else:
            encoder_hidden_states = outputs[0].reshape(batch_size, num_choices, seq_len, -1)[:, 0]
            encoder_attention_mask = attention_mask.reshape(batch_size, num_choices, seq_len)[:, 0]

        head_mask = self.get_head_mask(None, self.config.num_hidden_layers)
        decoder_outputs = self.decoder(
            hidden_states=decoder_input_embeds,
            attention_mask=rel_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=self.invert_attention_mask(encoder_attention_mask),
            return_dict=return_dict
        )

        lm_logits = self.seq2seq_head(decoder_outputs[0])

        loss = 0.
        mlm_loss = 0.
        cls_loss = 0.
        path_gen_loss = 0.
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
            rel_labels[rel_labels == self.eos_token_id] = -1
            path_gen_loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), rel_labels.view(-1))
            loss = loss + path_gen_loss

            if not self.training:
                acc, true_label_num = layers.get_accuracy(reshaped_logits, labels)
                self.eval_metrics.update("acc", val=acc, n=true_label_num)
                self.eval_metrics.update("loss", val=loss.item(), n=true_label_num)
                self.eval_metrics.update("cls_loss", val=cls_loss.item(), n=true_label_num)

                acc, true_label_num = layers.get_accuracy(lm_logits, rel_labels)
                self.eval_metrics.update("path_gen_acc", val=acc, n=true_label_num)
                self.eval_metrics.update("path_gen_loss", val=path_gen_loss.item(), n=true_label_num)

                if mlm_labels is not None:
                    acc, true_label_num = layers.get_accuracy(mlm_scores, mlm_labels)
                    self.eval_metrics.update("mlm_acc", val=acc, n=true_label_num)
                    self.eval_metrics.update("mlm_loss", val=mlm_loss.item(), n=true_label_num)

        if not return_dict:
            output = (reshaped_logits,) + outputs[2:] + (mlm_loss, cls_loss,)
            return ((loss,) + output) if loss is not None else output

        return MultipleChoicePreTrainModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            mlm_loss=mlm_loss,
            cls_loss=cls_loss,
            path_gen_loss=path_gen_loss,
        )


class RobertaForMultipleChoicePreTrainWPathGenV4(RobertaForMultipleChoiceForPreTrain, ABC):
    def __init__(self, config: RobertaConfig,
                 num_decoder_layers: int,
                 rel_vocab: str = None,
                 rel_vocab_size: int = None,
                 mlp_hidden_size: int = 768,
                 mlm_alpha: float = 1.0,
                 mlm_disabled: bool = False,
                 fs_checkpoint: bool = False,
                 fs_checkpoint_offload_to_cpu: bool = False,
                 fs_checkpoint_start_layer_id: int = 0,
                 decoder_config_path: str = None,
                 concat_ent: bool = False):
        super().__init__(config, mlp_hidden_size, mlm_alpha, mlm_disabled,
                         fs_checkpoint, fs_checkpoint_offload_to_cpu, fs_checkpoint_start_layer_id)

        if rel_vocab is not None:
            self.rel_vocab = pickle.load(open(rel_vocab, "rb"))
            self.rel_vocab_size = len(set(list(self.rel_vocab.values())))
            self.unk_token_id = -2
            for token, token_id in self.rel_vocab.items():
                if token == "<unk>":
                    self.unk_token_id = token_id
                    break
                assert token_id < self.rel_vocab_size
        elif rel_vocab_size is not None:
            self.rel_vocab_size = rel_vocab_size
            self.unk_token_id = -2
        else:
            raise RuntimeError()
        self.eos_token_id = self.rel_vocab_size
        self.pad_token_id = self.rel_vocab_size + 1

        self.t5_config = T5Config() if decoder_config_path is None else T5Config.from_pretrained(decoder_config_path)
        self.t5_config.is_decoder = True
        self.t5_config.is_encoder_decoder = False
        self.t5_config.add_cross_attention = True
        self.t5_config.num_layers = num_decoder_layers
        self.t5_config.eos_token_id = self.eos_token_id
        self.t5_config.pad_token_id = self.pad_token_id

        self.enc_proj = nn.Linear(config.hidden_size, self.t5_config.d_model)

        self.rel_embed = nn.Embedding(self.rel_vocab_size + 2, self.t5_config.d_model)
        logger.info(f"Relation embedding size: {self.rel_embed.weight.size()}")
        self.decoder = T5Stack(self.t5_config, self.rel_embed)
        self.decoder.post_init()

        self.seq2seq_head = nn.Linear(self.t5_config.d_model, self.rel_vocab_size + 2, bias=False)

        self.ent_decoder_config = copy.deepcopy(config)
        self.ent_decoder_config.is_decoder = True
        self.ent_decoder_config.add_cross_attention = True
        self.ent_decoder_config.num_hidden_layers = num_decoder_layers
        self.ent_decoder_config.eos_token_id = config.sep_token_id

        self.ent_decoder = RobertaModel(self.ent_decoder_config)
        self.ent_decoder.init_weights()

        self.ent_decode_head = RobertaLMHead(self.ent_decoder_config)

        self.ent_decode_head.decoder.weight = self.roberta.embeddings.word_embeddings.weight
        self.ent_decoder.embeddings.word_embeddings.weight = self.roberta.embeddings.word_embeddings.weight
        self.ent_decoder.embeddings.position_embeddings.weight = self.roberta.embeddings.position_embeddings.weight

        self.init_weights()
        self.tie_weights()
        self.init_metric("loss", "acc", "mlm_loss", "mlm_acc", "cls_loss", "path_gen_acc", "path_gen_loss", "ent_gen_loss", "ent_gen_acc")

        self.concat_ent = concat_ent

    def get_input_embeddings(self):
        return self.rel_embed

    def set_input_embeddings(self, new_embeddings):
        self.rel_embed = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def set_output_embeddings(self, new_embeddings):
        self.seq2seq_head = new_embeddings

    def get_output_embeddings(self):
        return self.seq2seq_head

    def forward(
            self,
            input_ids: Tensor,
            attention_mask: Tensor = None,
            token_type_ids: Tensor = None,
            labels: Tensor = None,
            mlm_input_ids: Tensor = None,
            mlm_attention_mask: Tensor = None,
            rel_labels: Tensor = None,
            mlm_labels: Tensor = None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            input_ids_dropout: Tensor = None,
            attention_mask_dropout: Tensor = None,
            entity_mentions: Tensor = None,
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

        # prepare encoder outputs
        if input_ids_dropout is not None:
            dropout_outputs = self.roberta(
                input_ids_dropout,
                attention_mask=attention_mask_dropout,
                return_dict=return_dict,
            )
            encoder_hidden_states = self.enc_proj(dropout_outputs[0])
            encoder_attention_mask = attention_mask_dropout
        else:
            encoder_hidden_states = self.enc_proj(outputs[0].reshape(batch_size, num_choices, seq_len, -1)[:, 0])
            encoder_attention_mask = attention_mask.reshape(batch_size, num_choices, seq_len)[:, 0]

        # entity mentions
        ent_decoder_input_ids = self._shift_right(entity_mentions, self.config.pad_token_id)
        ent_decoder_outputs = self.ent_decoder(
            input_ids=ent_decoder_input_ids,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            return_dict=return_dict
        )
        ent_logits = self.ent_decode_head(ent_decoder_outputs[0])

        ent_id_attention_mask = ent_decoder_input_ids != self.config.pad_token_id
        ent_id_attention_mask[:, 0] = 1
        if self.concat_ent:
            encoder_hidden_states = torch.cat([encoder_hidden_states, ent_decoder_outputs[0]], dim=1)
            encoder_attention_mask = torch.cat([encoder_attention_mask, ent_id_attention_mask], dim=1)

        # Seq2Seq
        decoder_input_ids = self._shift_right(rel_labels, self.pad_token_id)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            return_dict=return_dict
        )

        lm_logits = self.seq2seq_head(decoder_outputs[0])

        loss = 0.
        mlm_loss = 0.
        cls_loss = 0.
        path_gen_loss = 0.
        ent_gen_loss = 0.
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
            path_gen_loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), rel_labels.view(-1))
            loss = loss + path_gen_loss

            entity_mentions[entity_mentions == self.config.pad_token_id] = -1
            ent_gen_loss = loss_fct(ent_logits.view(-1, ent_logits.size(-1)), entity_mentions.view(-1))
            loss = loss + ent_gen_loss

            if not self.training:
                acc, true_label_num = layers.get_accuracy(reshaped_logits, labels)
                self.eval_metrics.update("acc", val=acc, n=true_label_num)
                self.eval_metrics.update("loss", val=loss.item(), n=true_label_num)
                self.eval_metrics.update("cls_loss", val=cls_loss.item(), n=true_label_num)

                acc, true_label_num = layers.get_accuracy(lm_logits, rel_labels)
                self.eval_metrics.update("path_gen_acc", val=acc, n=true_label_num)
                self.eval_metrics.update("path_gen_loss", val=path_gen_loss.item(), n=true_label_num)

                acc, true_label_num = layers.get_accuracy(ent_logits, entity_mentions)
                self.eval_metrics.update("ent_gen_acc", val=acc, n=true_label_num)
                self.eval_metrics.update("ent_gen_loss", val=ent_gen_loss.item(), n=true_label_num)

                if mlm_labels is not None:
                    acc, true_label_num = layers.get_accuracy(mlm_scores, mlm_labels)
                    self.eval_metrics.update("mlm_acc", val=acc, n=true_label_num)
                    self.eval_metrics.update("mlm_loss", val=mlm_loss.item(), n=true_label_num)

        if not return_dict:
            output = (reshaped_logits,) + outputs[2:] + (mlm_loss, cls_loss,)
            return ((loss,) + output) if loss is not None else output

        return MultipleChoicePreTrainModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            mlm_loss=mlm_loss,
            cls_loss=cls_loss,
            path_gen_loss=path_gen_loss,
            ent_gen_loss=ent_gen_loss,
        )

    def _shift_right(self, input_ids, pad_token_id):
        decoder_start_token_id = pad_token_id

        assert decoder_start_token_id is not None, (
            "self.model.config.decoder_start_token_id has to be defined. In T5 it is usually set to the pad_token_id."
            " See T5 docs for more information"
        )

        # shift inputs to the right
        if is_torch_fx_proxy(input_ids):
            # Item assignment is not supported natively for proxies.
            shifted_input_ids = torch.full(input_ids.shape[:-1] + (1,), decoder_start_token_id)
            shifted_input_ids = torch.cat([shifted_input_ids, input_ids[..., :-1]], dim=-1)
        else:
            shifted_input_ids = input_ids.new_zeros(input_ids.shape)
            shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
            shifted_input_ids[..., 0] = decoder_start_token_id

        assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -1, pad_token_id)

        assert torch.all(shifted_input_ids >= 0).item(), "Verify that `shifted_input_ids` has only positive values"

        return shifted_input_ids


class RobertaForMultipleChoicePreTrainWPathGenV5(RobertaForMultipleChoiceForPreTrain, ABC):
    def __init__(self, config: RobertaConfig,
                 num_decoder_layers: int,
                 rel_vocab: str,
                 mlp_hidden_size: int = 768,
                 mlm_alpha: float = 1.0,
                 mlm_disabled: bool = False,
                 fs_checkpoint: bool = False,
                 fs_checkpoint_offload_to_cpu: bool = False,
                 fs_checkpoint_start_layer_id: int = 0,
                 decoder_config_path: str = None,
                 rel_gen_coff: float = 1.0, ):
        super().__init__(config, mlp_hidden_size, mlm_alpha, mlm_disabled,
                         fs_checkpoint, fs_checkpoint_offload_to_cpu, fs_checkpoint_start_layer_id)

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

        self.t5_config = T5Config() if decoder_config_path is None else T5Config.from_pretrained(decoder_config_path)
        self.t5_config.is_decoder = True
        self.t5_config.is_encoder_decoder = False
        self.t5_config.add_cross_attention = True
        self.t5_config.num_layers = num_decoder_layers
        self.t5_config.eos_token_id = self.eos_token_id
        self.t5_config.pad_token_id = self.pad_token_id

        self.enc_proj = nn.Linear(config.hidden_size, self.t5_config.d_model)

        self.rel_embed = nn.Embedding(self.rel_vocab_size + 2, self.t5_config.d_model)
        logger.info(f"Relation embedding size: {self.rel_embed.weight.size()}")
        self.decoder = T5Stack(self.t5_config, self.rel_embed)
        self.decoder.post_init()

        self.seq2seq_head = nn.Linear(self.t5_config.d_model, self.rel_vocab_size + 2, bias=False)
        self.rel_embed_pooler = nn.Linear(self.t5_config.d_model, 1)
        self.rel_embed_cls = nn.Linear(self.t5_config.d_model, 1)

        self.rel_gen_coff = rel_gen_coff

        self.init_weights()
        self.tie_weights()
        self.init_metric("loss", "acc", "mlm_loss", "mlm_acc", "cls_loss", "path_gen_acc", "path_gen_loss", "rel_ctr_loss", "rel_acc")

    def get_input_embeddings(self):
        return self.rel_embed

    def set_input_embeddings(self, new_embeddings):
        self.rel_embed = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def set_output_embeddings(self, new_embeddings):
        self.seq2seq_head = new_embeddings

    def get_output_embeddings(self):
        return self.seq2seq_head

    def forward(
            self,
            input_ids: Tensor,
            attention_mask: Tensor = None,
            token_type_ids: Tensor = None,
            labels: Tensor = None,
            mlm_input_ids: Tensor = None,
            mlm_attention_mask: Tensor = None,
            rel_labels: Tensor = None,
            mlm_labels: Tensor = None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            input_ids_dropout: Tensor = None,
            attention_mask_dropout: Tensor = None,
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

        # Seq2Seq
        # if input_ids_dropout is not None:
        #     dropout_outputs = self.roberta(
        #         input_ids_dropout,
        #         attention_mask=attention_mask_dropout,
        #         return_dict=return_dict,
        #     )
        #     encoder_hidden_states = self.enc_proj(dropout_outputs[0])
        #     encoder_attention_mask = attention_mask_dropout
        # else:
        encoder_hidden_states = self.enc_proj(outputs[0])

        decoder_input_ids = self._shift_right(rel_labels)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask,
            return_dict=return_dict
        )

        lm_logits = self.seq2seq_head(decoder_outputs[0])

        # Relation embedding normalization.
        rel_mask = (rel_labels == -1) | (rel_labels == self.eos_token_id)
        rel_ids = rel_labels.masked_fill(rel_mask, self.eos_token_id)
        rel_embeddings = self.dropout(self.rel_embed(rel_ids))
        rel_h = torch.einsum("bs,bsh->bh", torch.softmax(
            self.rel_embed_pooler(rel_embeddings).squeeze(-1) + rel_mask.to(dtype=self.dtype) * torch.finfo(self.dtype).min, dim=1),
                             rel_embeddings)
        rel_logits = self.rel_embed_cls(rel_h).reshape(-1, num_choices)

        loss = 0.
        mlm_loss = 0.
        cls_loss = 0.
        path_gen_loss = 0.
        rel_ctr_loss = 0.
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
            path_gen_loss = self.rel_gen_coff * loss_fct(lm_logits.view(-1, lm_logits.size(-1)), rel_labels.view(-1))
            loss = loss + path_gen_loss

            rel_ctr_loss = loss_fct(rel_logits, labels)
            loss = loss + rel_ctr_loss

            if not self.training:
                acc, true_label_num = layers.get_accuracy(reshaped_logits, labels)
                self.eval_metrics.update("acc", val=acc, n=true_label_num)
                self.eval_metrics.update("loss", val=loss.item(), n=true_label_num)
                self.eval_metrics.update("cls_loss", val=cls_loss.item(), n=true_label_num)

                acc, true_label_num = layers.get_accuracy(lm_logits, rel_labels)
                self.eval_metrics.update("path_gen_acc", val=acc, n=true_label_num)
                self.eval_metrics.update("path_gen_loss", val=path_gen_loss.item(), n=true_label_num)

                acc, true_label_num = layers.get_accuracy(rel_logits, labels)
                self.eval_metrics.update("rel_acc", val=acc, n=true_label_num)
                self.eval_metrics.update("rel_ctr_loss", val=rel_ctr_loss.item(), n=true_label_num)

                if mlm_labels is not None:
                    acc, true_label_num = layers.get_accuracy(mlm_scores, mlm_labels)
                    self.eval_metrics.update("mlm_acc", val=acc, n=true_label_num)
                    self.eval_metrics.update("mlm_loss", val=mlm_loss.item(), n=true_label_num)

        if not return_dict:
            output = (reshaped_logits,) + outputs[2:] + (mlm_loss, cls_loss,)
            return ((loss,) + output) if loss is not None else output

        return MultipleChoicePreTrainModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            mlm_loss=mlm_loss,
            cls_loss=cls_loss,
            path_gen_loss=path_gen_loss,
            rel_ctr_loss=rel_ctr_loss,
        )

    def _shift_right(self, input_ids):
        decoder_start_token_id = pad_token_id = self.pad_token_id

        assert decoder_start_token_id is not None, (
            "self.model.config.decoder_start_token_id has to be defined. In T5 it is usually set to the pad_token_id."
            " See T5 docs for more information"
        )

        # shift inputs to the right
        if is_torch_fx_proxy(input_ids):
            # Item assignment is not supported natively for proxies.
            shifted_input_ids = torch.full(input_ids.shape[:-1] + (1,), decoder_start_token_id)
            shifted_input_ids = torch.cat([shifted_input_ids, input_ids[..., :-1]], dim=-1)
        else:
            shifted_input_ids = input_ids.new_zeros(input_ids.shape)
            shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
            shifted_input_ids[..., 0] = decoder_start_token_id

        assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -1, pad_token_id)

        assert torch.all(shifted_input_ids >= 0).item(), "Verify that `shifted_input_ids` has only positive values"

        return shifted_input_ids
