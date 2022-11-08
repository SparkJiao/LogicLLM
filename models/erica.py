from abc import ABC

from transformers.models.roberta.modeling_roberta import RobertaModel, RobertaConfig, MaskedLMOutput
import torch
import torch.distributed as dist
from typing import Optional, Tuple
from torch import Tensor
from dataclasses import dataclass
from general_util.logger import get_child_logger


@dataclass
class ERICAOutput(MaskedLMOutput):
    entity_hidden: Optional[torch.FloatTensor] = None
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class ERICAPredictor(RobertaModel, ABC):
    def __init__(self, config: RobertaConfig):
        super().__init__(config)

    def forward(self, input_ids: Optional[torch.LongTensor] = None,
                attention_mask: Optional[torch.FloatTensor] = None,
                token_type_ids: Optional[torch.LongTensor] = None,
                ent_seq_mapping: Optional[torch.FloatTensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                head_mask: Optional[torch.FloatTensor] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                encoder_hidden_states: Optional[torch.FloatTensor] = None,
                encoder_attention_mask: Optional[torch.FloatTensor] = None,
                labels: Optional[torch.LongTensor] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None, ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # outputs = self.roberta(
        #     input_ids,
        #     attention_mask=attention_mask,
        #     return_dict=return_dict,
        # )
        outputs = super().forward(input_ids, attention_mask, return_dict=return_dict)
        sequence_output = outputs[0]
        ent_hidden = torch.einsum("bes,bsh->beh", ent_seq_mapping, sequence_output)

        if not return_dict:
            output = (ent_hidden,) + outputs[2:]
            return output

        return ERICAOutput(
            entity_hidden=ent_hidden,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
