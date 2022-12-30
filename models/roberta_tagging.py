import copy
import pickle
from abc import ABC
from dataclasses import dataclass
from typing import Union

import torch
import transformers
from torch import nn, Tensor
from torch.nn import CrossEntropyLoss
from transformers.models.roberta.modeling_roberta import RobertaConfig

from general_util.logger import get_child_logger
from models.roberta import RobertaForMultipleChoiceForPreTrain, RelDecoderHead, MultipleChoicePreTrainModelOutput
from modules import layers

logger = get_child_logger("RoBERTa.Tagger")


@dataclass
class TaggingOutputClass(MultipleChoicePreTrainModelOutput):
    tagging_loss: torch.FloatTensor = None
    tagging_acc: torch.FloatTensor = None
    tagging_precision: torch.FloatTensor = None
    tagging_recall: torch.FloatTensor = None


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
                 tagging_coff: float = 1.0):
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

        self.init_metric("loss", "acc", "mlm_loss", "mlm_acc", "cls_loss", "path_gen_acc", "path_gen_loss",
                         "tagging_loss", "tagging_acc", "tagging_p", "tagging_r")

        self.tokenizer = transformers.AutoTokenizer.from_pretrained("pretrained-models/roberta-large")

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
        batch_size, num_choices, seq_len = input_ids.size()

        # print(input_ids.size())

        input_ids = self.fold_tensor(input_ids)
        attention_mask = self.fold_tensor(attention_mask)
        token_type_ids = self.fold_tensor(token_type_ids)

        # print(self.tokenizer.decode(input_ids[0]))
        # print(self.tokenizer.decode(input_ids[1]))

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

        seq_hidden = self.dropout(outputs[0].reshape(batch_size, num_choices, seq_len, -1)[:, 0])

        tagging_logits = self.tagger(seq_hidden)

        # print(h_span_marks.size())
        # print(h_span_marks[0].tolist())
        # print(tagging_labels[0].tolist())

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
            # if entity_pair_mask is not None:
            rel_labels[entity_pair_mask] = -1

            # path_gen_loss = self.rel_gen_coff * no_reduce_loss_fct(sent_rel_logits.reshape(-1, sent_rel_logits.size(-1)),
            #                                                        rel_labels.reshape(-1))
            # nan_mask = torch.isnan(path_gen_loss)
            # path_gen_loss[nan_mask] = 0.
            # path_gen_loss = path_gen_loss.sum() / (~nan_mask).sum().item()
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
            tagging_loss=tagging_loss
        )
