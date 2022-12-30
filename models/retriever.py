from abc import ABC

import torch
from torch import Tensor
from models.roberta import RobertaForMultipleChoiceForPreTrainWithPairFull
from dataclasses import dataclass
from transformers.modeling_outputs import MultipleChoiceModelOutput


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
    retrieval_hidden_states: torch.FloatTensor = None


class RobertaMCPretrainPairInference(RobertaForMultipleChoiceForPreTrainWithPairFull, ABC):
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
        pair_q = self.pair_proj(self.dropout(self.pair_pooler(pooled_output)))

        return MultipleChoicePreTrainModelOutput(
            retrieval_hidden_states=pair_q
        )
