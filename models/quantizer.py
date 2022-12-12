from abc import ABC
from typing import Optional

import torch
from torch import nn

from transformers.models.roberta.modeling_roberta import RobertaModel, RobertaPreTrainedModel, RobertaForMaskedLM, RobertaConfig


class Quantizer(nn.Module):
    def get_code_indices(self, hidden_states: torch.Tensor):
        raise NotImplementedError

    def quantize(self, code_indices: torch.LongTensor):
        raise NotImplementedError


class RobertaPriorModel(RobertaPreTrainedModel, ABC):
    def __init__(self, config: RobertaConfig,
                 quantizer: Quantizer,
                 ):
        super().__init__(config)

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        # self.lm_head = RobertaLMHead(config)
        self.quantizer = quantizer

        # The LM head weights require special treatment only when they are tied with the word embeddings
        # self.update_keys_to_ignore(config, ["lm_head.decoder.weight"])

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

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

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            return_dict=return_dict,
        )
        # outputs = super().forward(input_ids, attention_mask, return_dict=return_dict)
        sequence_output = outputs[0]
        ent_hidden = torch.einsum("bes,bsh->beh", ent_seq_mapping, sequence_output)

        if not return_dict:
            output = (ent_hidden,) + outputs[2:]
            return output

        # return ERICAOutput(
        #     entity_hidden=ent_hidden,
        #     hidden_states=outputs.hidden_states,
        #     attentions=outputs.attentions,
        # )
