from abc import ABC
from typing import Dict

import torch
from torch import nn, Tensor
from torch.nn import CrossEntropyLoss
from transformers.models.roberta.modeling_roberta import RobertaConfig

from general_util.logger import get_child_logger
from models.retriever import RetrieverMixin
from models.roberta import RobertaForMultipleChoiceForPreTrain, MultipleChoicePreTrainModelOutput
from modules import layers

logger = get_child_logger("RoBERTa.Pair")


# @dataclass
# class TaggingOutputClass(MultipleChoicePreTrainModelOutput):


class RobertaForMultipleChoicePreTrainPairV1(RobertaForMultipleChoiceForPreTrain, RetrieverMixin, ABC):
    def __init__(self, config: RobertaConfig,
                 mlp_hidden_size: int = 768,
                 mlm_alpha: float = 1.0,
                 mlm_disabled: bool = False,
                 fs_checkpoint: bool = False,
                 fs_checkpoint_offload_to_cpu: bool = False,
                 fs_checkpoint_start_layer_id: int = 0,
                 dist_func: str = "mlp"):
        super().__init__(config, mlp_hidden_size, mlm_alpha, mlm_disabled,
                         fs_checkpoint, fs_checkpoint_offload_to_cpu, fs_checkpoint_start_layer_id)

        self.dist_func = dist_func
        if self.dist_func == "mlp":
            self.scorer = nn.Sequential(nn.Linear(config.hidden_size * 2, mlp_hidden_size),
                                        nn.Tanh(),
                                        nn.Dropout(p=getattr(config, "pooler_dropout", config.hidden_dropout_prob)),
                                        nn.Linear(mlp_hidden_size, 1))
        elif self.dist_func == "dot":
            self.q = nn.Linear(config.hidden_size, config.hidden_size)
            self.k = nn.Linear(config.hidden_size, config.hidden_size)
        else:
            raise RuntimeError()

        self.init_metric("loss", "acc", "mlm_loss", "mlm_acc", "cls_loss", "pair_acc", "pair_loss")

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
            pair_token_type_ids: Tensor = None,
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

        pair_num = pair_input_ids.size(1) - 1
        pair_input_ids = self.fold_tensor(pair_input_ids)
        pair_attention_mask = self.fold_tensor(pair_attention_mask)
        pair_token_type_ids = self.fold_tensor(pair_token_type_ids)
        pair_outputs = self.roberta(
            pair_input_ids,
            token_type_ids=pair_token_type_ids,
            attention_mask=pair_attention_mask,
            return_dict=return_dict,
        )
        pair_pooled_outputs = pair_outputs[0][:, 0].reshape(batch_size, pair_num + 1, -1)
        if self.dist_func == "mlp":
            pair_q = pair_pooled_outputs[:, :1].expand(-1, pair_num, -1)
            pair_hidden = torch.cat([pair_q, pair_pooled_outputs[:, 1:]], dim=-1)
            pair_logits = self.scorer(pair_hidden).squeeze(-1)
        elif self.dist_func == "dot":
            pair_q = self.q(pair_pooled_outputs[:, 0])
            pair_k = self.k(pair_pooled_outputs[:, 1:])
            pair_logits = torch.einsum("bh,bsh->bs", pair_q, pair_k)
        else:
            raise RuntimeError()
        pair_logits = pair_logits + (1 - pair_mask) * -10000.0

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

            pair_loss = loss_fct(pair_logits, pair_labels)
            _, true_label_num = layers.get_accuracy(pair_logits, pair_labels)
            if true_label_num:
                loss = loss + pair_loss

            if not self.training:
                acc, true_label_num = layers.get_accuracy(reshaped_logits, labels)
                self.eval_metrics.update("acc", val=acc, n=true_label_num)
                self.eval_metrics.update("loss", val=loss.item(), n=true_label_num)
                self.eval_metrics.update("cls_loss", val=cls_loss.item(), n=true_label_num)

                acc, true_label_num = layers.get_accuracy(pair_logits, pair_labels)
                if true_label_num:
                    self.eval_metrics.update("pair_acc", val=acc, n=true_label_num)
                    self.eval_metrics.update("pair_loss", val=pair_loss, n=true_label_num)

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
            pair_loss=pair_loss
        )

    @torch.no_grad()
    def encode_index(self, input_ids, attention_mask, token_type_ids):
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )
        hidden_states = outputs[0][:, 0]
        if self.dist_func == "mlp":
            pass
        elif self.dist_func == "dot":
            hidden_states = self.k(hidden_states)
        else:
            raise RuntimeError()
        hidden_states = hidden_states.cpu()
        if self.cached_indices is None:
            self.cached_indices = hidden_states
        else:
            self.cached_indices = torch.cat([self.cached_indices, hidden_states], dim=0)

    def encode_query(self, input_ids, attention_mask, token_type_ids):
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )
        hidden_states = outputs[0][:, 0]
        if self.dist_func == "dot":
            hidden_states = self.q(hidden_states)
        return hidden_states

    # def build_index(self, data_loader: DataLoader, *args, **kwargs):
    #     for batch in data_loader:
    #         self.encode_index(**batch_to_device(batch))

    @torch.no_grad()
    def search(self, input_ids, attention_mask, token_type_ids):
        query = self.encode_query(input_ids, attention_mask, token_type_ids)
        scores = []
        for k_id, k_hidden_states in enumerate(self.cached_indices):
            k_hidden_states = k_hidden_states.to(device=query.device)
            if self.dist_func == "mlp":
                pair_hidden = torch.cat([query,
                                         k_hidden_states.unsqueeze(0).expand(query.size(0), -1)], dim=1)
                scores.append(self.scorer(pair_hidden))
            elif self.dist_func == "dot":
                scores.append(torch.einsum("bh,h->b", query, k_hidden_states).unsqueeze(-1))
            else:
                raise RuntimeError()
        scores = torch.cat(scores, dim=1)
        return scores


class RobertaForMultipleChoicePreTrainPairV2(RobertaForMultipleChoicePreTrainPairV1, ABC):
    def __init__(self, config: RobertaConfig,
                 mlp_hidden_size: int = 768,
                 mlm_alpha: float = 1.0,
                 mlm_disabled: bool = False,
                 fs_checkpoint: bool = False,
                 fs_checkpoint_offload_to_cpu: bool = False,
                 fs_checkpoint_start_layer_id: int = 0,
                 dist_func: str = "mlp",
                 vem: nn.Module = None):
        super().__init__(config, mlp_hidden_size, mlm_alpha, mlm_disabled,
                         fs_checkpoint, fs_checkpoint_offload_to_cpu, fs_checkpoint_start_layer_id, dist_func)

        self.vem = vem
        if self.dist_func == "dot":
            self.q = None
            self.k = None
        self.scorer = nn.Linear(config.hidden_size, config.hidden_size)

        self.init_metric("loss", "acc", "mlm_loss", "mlm_acc", "cls_loss", "pair_acc", "pair_loss")

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
            pair_token_type_ids: Tensor = None,
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

        # pair_num = pair_input_ids.size(1) - 1
        pair_input_ids = pair_input_ids[:, :2]
        pair_attention_mask = pair_attention_mask[:, :2]
        pair_input_ids = self.fold_tensor(pair_input_ids)
        pair_attention_mask = self.fold_tensor(pair_attention_mask)
        pair_token_type_ids = self.fold_tensor(pair_token_type_ids)
        pair_outputs = self.roberta(
            pair_input_ids,
            token_type_ids=pair_token_type_ids,
            attention_mask=pair_attention_mask,
            return_dict=return_dict,
        )
        pair_pooled_outputs = pair_outputs[0][:, 0].reshape(batch_size, 2, -1)
        pair_q = self.scorer(pair_pooled_outputs[:, 0])
        pair_k = self.scorer(pair_pooled_outputs[:, 1])
        pair_logits = self.vem(pair_q, pair_k)

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

            pair_loss = loss_fct(pair_logits, pair_labels)
            _, true_label_num = layers.get_accuracy(pair_logits, pair_labels)
            if true_label_num:
                loss = loss + pair_loss

            if not self.training:
                acc, true_label_num = layers.get_accuracy(reshaped_logits, labels)
                self.eval_metrics.update("acc", val=acc, n=true_label_num)
                self.eval_metrics.update("loss", val=loss.item(), n=true_label_num)
                self.eval_metrics.update("cls_loss", val=cls_loss.item(), n=true_label_num)

                acc, true_label_num = layers.get_accuracy(pair_logits, pair_labels)
                if true_label_num:
                    self.eval_metrics.update("pair_acc", val=acc, n=true_label_num)
                    self.eval_metrics.update("pair_loss", val=pair_loss, n=true_label_num)

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
            pair_loss=pair_loss
        )


class RobertaForMultipleChoicePreTrainPairOneTowerV1(RobertaForMultipleChoiceForPreTrain, RetrieverMixin, ABC):
    def __init__(self, config: RobertaConfig,
                 mlp_hidden_size: int = 768,
                 mlm_alpha: float = 1.0,
                 mlm_disabled: bool = False,
                 fs_checkpoint: bool = False,
                 fs_checkpoint_offload_to_cpu: bool = False,
                 fs_checkpoint_start_layer_id: int = 0,
                 ):
        super().__init__(config, mlp_hidden_size, mlm_alpha, mlm_disabled,
                         fs_checkpoint, fs_checkpoint_offload_to_cpu, fs_checkpoint_start_layer_id)

        self.scorer = nn.Sequential(
            nn.Linear(config.hidden_size, mlp_hidden_size),
            nn.Tanh(),
            nn.Dropout(p=getattr(config, "pooler_dropout", config.hidden_dropout_prob)),
            nn.Linear(mlp_hidden_size, 1)
        )

        self.init_metric("loss", "acc", "mlm_loss", "mlm_acc", "cls_loss", "pair_acc", "pair_loss")

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
            pair_token_type_ids: Tensor = None,
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

        pair_num = pair_input_ids.size(1)
        pair_input_ids = self.fold_tensor(pair_input_ids)
        pair_attention_mask = self.fold_tensor(pair_attention_mask)
        pair_token_type_ids = self.fold_tensor(pair_token_type_ids)
        pair_outputs = self.roberta(
            pair_input_ids,
            token_type_ids=pair_token_type_ids,
            attention_mask=pair_attention_mask,
            return_dict=return_dict,
        )
        pair_pooled_outputs = pair_outputs[0][:, 0]
        pair_logits = self.scorer(pair_pooled_outputs).reshape(batch_size, pair_num)
        pair_logits = pair_logits + (1 - pair_mask) * -10000.0

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

            pair_loss = loss_fct(pair_logits, pair_labels)
            _, true_label_num = layers.get_accuracy(pair_logits, pair_labels)
            if true_label_num:
                loss = loss + pair_loss

            if not self.training:
                acc, true_label_num = layers.get_accuracy(reshaped_logits, labels)
                self.eval_metrics.update("acc", val=acc, n=true_label_num)
                self.eval_metrics.update("loss", val=loss.item(), n=true_label_num)
                self.eval_metrics.update("cls_loss", val=cls_loss.item(), n=true_label_num)

                acc, true_label_num = layers.get_accuracy(pair_logits, pair_labels)
                if true_label_num:
                    self.eval_metrics.update("pair_acc", val=acc, n=true_label_num)
                    self.eval_metrics.update("pair_loss", val=pair_loss, n=true_label_num)

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
            pair_loss=pair_loss
        )


def extract_seq_hidden(tf_outputs: Dict, layer_idx: int):
    return tf_outputs["hidden_states"][layer_idx]
