from abc import ABC
from typing import Optional, Tuple, Union, Dict, List

import torch
from torch import nn, Tensor
from torch.nn import functional
from nltk import word_tokenize
from nltk.translate.bleu_score import sentence_bleu
from torch import Tensor
from torch import distributed as dist
from torch.nn import CrossEntropyLoss
from transformers.models.bart.modeling_bart import (
    BartConfig,
    BartModel,
    BartForConditionalGeneration,
    BartPretrainedModel,
    Seq2SeqModelOutput,
    Seq2SeqLMOutput,
    BaseModelOutput,
    shift_tokens_right,
    BartEncoder,
    BartDecoder,
)
from transformers.utils import ModelOutput

from general_util.logger import get_child_logger
from general_util.mixin import LogMixin
from modules import layers
from dataclasses import dataclass

logger = get_child_logger("BART")


@dataclass
class DVAEModelOutput(Seq2SeqModelOutput):
    z_encoder: torch.FloatTensor = None
    z_decoder: torch.FloatTensor = None
    vq_loss: torch.FloatTensor = None
    commitment_loss: torch.FloatTensor = None


@dataclass
class DVAESeq2SeqLMOutput(Seq2SeqLMOutput):
    vq_loss: torch.FloatTensor = None
    commitment_loss: torch.FloatTensor = None
    mlm_loss: torch.FloatTensor = None
    z_encoder_mean: torch.FloatTensor = None
    z_decoder_mean: torch.FloatTensor = None
    code_indices: torch.LongTensor = None


@dataclass
class DVAEInferenceOutput(ModelOutput):
    code_indices: torch.LongTensor = None


def l2norm(t):
    return functional.normalize(t, p=2, dim=-1)


class BartModelCompressedCrossAttention(BartModel, ABC):
    def __init__(self, config: BartConfig, codebook_size: int, code_dim: int = 128, add_cross_attn: bool = True):
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.encoder = BartEncoder(config, self.shared)
        self.decoder = BartDecoder(config, self.shared)

        self.add_cross_attn = add_cross_attn

        self.codebook_size = codebook_size
        if code_dim == config.d_model:
            self.codebook = nn.Embedding(codebook_size, config.d_model)
            self.code_proj = nn.Identity()
        else:
            self.codebook = nn.Embedding(codebook_size, code_dim)
            self.code_proj = nn.Linear(code_dim, config.d_model)

        # self.codebook = nn.Parameter(torch.FloatTensor(codebook_size, config.d_model))
        # self.codebook_cls = nn.Linear(codebook_size, config.d_model)

        # Initialize weights and apply final processing
        self.post_init()
        # std = self.config.init_std
        # self.codebook.data.normal_(mean=0.0, std=std)

        self.mse_loss = torch.nn.MSELoss()

    # def get_input_embeddings(self):
    #     return self.shared
    #
    # def set_input_embeddings(self, value):
    #     self.shared = value
    #     self.encoder.embed_tokens = self.shared
    #     self.decoder.embed_tokens = self.shared
    #
    # def get_encoder(self):
    #     return self.encoder
    #
    # def get_decoder(self):
    #     return self.decoder

    def get_code_indices(self, flat_x):
        # compute L2 distance
        emb_weight = self.code_proj(self.codebook.weight)
        distances = (
                torch.sum(flat_x ** 2, dim=1, keepdim=True) +
                torch.sum(emb_weight ** 2, dim=1) -
                2. * torch.matmul(flat_x, emb_weight.t())
        )  # [N, M]
        encoding_indices = torch.argmin(distances, dim=1)  # [N,]
        return encoding_indices

    def quantize(self, encoding_indices):
        """Returns embedding tensor for a batch of indices."""
        return self.code_proj(self.codebook(encoding_indices))

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
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Seq2SeqModelOutput]:

        # different to other models, Bart automatically creates decoder_input_ids from
        # input_ids if no decoder_input_ids are provided
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            if input_ids is None:
                raise ValueError(
                    "If no `decoder_input_ids` or `decoder_inputs_embeds` are "
                    "passed, `input_ids` cannot be `None`. Please pass either "
                    "`input_ids` or `decoder_input_ids` or `decoder_inputs_embeds`."
                )

            decoder_input_ids = shift_tokens_right(
                input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        sentence_representation = encoder_outputs[0][:, 0]
        # # Use only the latent code as the encoder outputs.
        # encoder_hidden_states = sentence_representation.unsqueeze(1)
        # encoder_attention_mask = attention_mask[:, :1]

        code_indices = self.get_code_indices(sentence_representation)
        code_embedding = self.quantize(code_indices)

        # VQ loss
        vq_loss = self.mse_loss(code_embedding, sentence_representation.detach())
        # commitment loss
        commitment_loss = self.mse_loss(sentence_representation, code_embedding.detach())
        # assert code_embedding.requires_grad
        # assert sentence_representation.requires_grad

        # straight-through estimator
        code_embedding = sentence_representation + (code_embedding - sentence_representation).detach()

        if self.add_cross_attn:
            # Use only the latent code as the encoder outputs.
            encoder_hidden_states = code_embedding.unsqueeze(1)
            encoder_attention_mask = attention_mask[:, :1]
        else:
            encoder_hidden_states = None
            encoder_attention_mask = None

        # print(encoder_hidden_states.size())
        # print(encoder_attention_mask.size())

        # Generate `input_embeds` here to concat the latent code.
        decoder_inputs_embeds = self.decoder.embed_tokens(decoder_input_ids) * self.decoder.embed_scale
        # decoder_inputs_embeds = torch.cat([code_embedding.unsqueeze(1) * self.decoder.embed_scale, decoder_inputs_embeds], dim=1)
        decoder_inputs_embeds[:, 0] = code_embedding * self.decoder.embed_scale

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            # input_ids=decoder_input_ids,
            input_ids=None,
            attention_mask=decoder_attention_mask,
            # encoder_hidden_states=encoder_outputs[0],
            # encoder_attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return DVAEModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
            z_encoder=sentence_representation,
            z_decoder=code_embedding,
            vq_loss=vq_loss,
            commitment_loss=commitment_loss,
        )


class BartDVAE(BartForConditionalGeneration, ABC, LogMixin):
    base_model_prefix = "model"
    _keys_to_ignore_on_load_missing = [r"final_logits_bias", r"lm_head.weight"]

    def __init__(self, config: BartConfig, codebook_size, code_dim: int = 128, vqvae_type: str = "compressed", freeze_decoder: bool = False,
                 commitment_cost: float = 0.25, alpha: float = 1.0, decay: float = 0., epsilon: float = 1e-5,
                 add_cross_attn: bool = True, add_norm: bool = False, detach_extra_outputs: bool = False, mlp_after: bool = False,
                 add_point_wise: bool = False,
                 inference_stage: bool = False):
        super().__init__(config)

        if config.model_type != "bart":
            logger.warning(f"Check if `model_type` is set as the parameter to initialize the model.")
            vqvae_type = config.model_type

        if vqvae_type == "w_csa":
            self.model = BartModelCodebook(config, codebook_size=codebook_size, code_dim=code_dim)
        elif vqvae_type == "compressed":
            self.model = BartModelCompressedCrossAttention(config, codebook_size=codebook_size, code_dim=code_dim,
                                                           add_cross_attn=add_cross_attn)
        elif vqvae_type == "w_csa_ema":
            self.model = BartModelCodebookEMA(config, codebook_size, decay, code_dim, epsilon, add_cross_attn, add_norm,
                                              detach_extra_outputs, mlp_after, add_point_wise)
        elif vqvae_type == "bottleneck_ema":
            self.model = BartModelCodebookEMABottleNeck(config, codebook_size, decay, code_dim, epsilon, add_norm, detach_extra_outputs,
                                                        mlp_after, add_point_wise)
        else:
            raise NotImplementedError()
        print(self.model.__class__.__name__)
        if freeze_decoder:
            for param in self.model.decoder.parameters():
                param.requires_grad = False

        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

        self.commitment_cost = commitment_cost
        self.alpha = alpha

        self.inference_stage = inference_stage

        self.init_metric("acc", "loss", "mlm_loss", "vq_loss", "commit_loss", "ent_mark_acc")

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
            partial_mask: Optional[torch.BoolTensor] = None,
            ent_mark_mask: Optional[torch.BoolTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            **kwargs,
    ) -> Union[Tuple, Seq2SeqLMOutput, DVAEInferenceOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if use_cache:
                logger.warning("The `use_cache` argument is changed to `False` since `labels` is provided.")
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )
        if self.inference_stage:
            kwargs["inference"] = True

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
            **kwargs
        )

        if self.inference_stage:
            return outputs

        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias
        if lm_logits.size(1) == labels.size(1) + 1:
            lm_logits = lm_logits[:, 1:]

        loss = None
        masked_lm_loss = None
        vq_loss = None
        commitment_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            labels[labels == self.config.pad_token_id] = -100
            if partial_mask is not None:
                labels.masked_fill_(partial_mask, -100)

            masked_lm_loss = loss_fct(lm_logits.reshape(-1, self.config.vocab_size), labels.view(-1))

            vq_loss = self.alpha * outputs.vq_loss
            commitment_loss = self.alpha * self.commitment_cost * outputs.commitment_loss

            loss = masked_lm_loss + vq_loss + commitment_loss

            if not self.training:
                acc, true_label_num = layers.get_accuracy(lm_logits, labels, pad_id=-100)
                self.eval_metrics.update("acc", val=acc, n=true_label_num)
                self.eval_metrics.update("loss", val=loss.item(), n=lm_logits.size(0))
                if isinstance(vq_loss, Tensor):
                    self.eval_metrics.update("vq_loss", val=vq_loss.item(), n=lm_logits.size(0))
                else:
                    self.eval_metrics.update("vq_loss", val=vq_loss, n=lm_logits.size(0))
                self.eval_metrics.update("commit_loss", val=commitment_loss.item(), n=lm_logits.size(0))
                self.eval_metrics.update("mlm_loss", val=masked_lm_loss.item(), n=true_label_num)

                if ent_mark_mask is not None:
                    ent_mark_labels = labels.masked_fill(~ent_mark_mask, -100)
                    acc, true_label_num = layers.get_accuracy(lm_logits, ent_mark_labels, pad_id=-100)
                    self.eval_metrics.update("ent_mark_acc", acc, n=true_label_num)

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return DVAESeq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
            vq_loss=vq_loss,
            commitment_loss=commitment_loss,
            mlm_loss=masked_lm_loss,
            z_encoder_mean=torch.mean(outputs.z_encoder.detach()),
            z_decoder_mean=torch.mean(outputs.z_decoder.detach()),
        )


class BartModelCodebook(BartModel, ABC):
    def __init__(self, config: BartConfig, codebook_size: int, code_dim: int = 128):
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.encoder = BartEncoder(config, self.shared)
        self.decoder = BartDecoder(config, self.shared)

        self.codebook_size = codebook_size
        if code_dim == config.d_model:
            self.codebook = nn.Embedding(codebook_size, config.d_model)
            self.code_proj = nn.Identity()
        else:
            self.codebook = nn.Embedding(codebook_size, code_dim)
            self.code_proj = nn.Linear(code_dim, config.d_model)

        for dec_layer in self.decoder.layers:
            for param in dec_layer.encoder_attn.parameters():
                param.requires_grad = False
            for param in dec_layer.encoder_attn_layer_norm.parameters():
                param.requires_grad = False

        # Initialize weights and apply final processing
        self.post_init()

        self.mse_loss = torch.nn.MSELoss()

    def get_code_indices(self, flat_x):
        # compute L2 distance
        emb_weight = self.code_proj(self.codebook.weight)
        distances = (
                torch.sum(flat_x ** 2, dim=1, keepdim=True) +
                torch.sum(emb_weight ** 2, dim=1) -
                2. * torch.matmul(flat_x, emb_weight.t())
        )  # [N, M]
        encoding_indices = torch.argmin(distances, dim=1)  # [N,]
        return encoding_indices

    def quantize(self, encoding_indices):
        """Returns embedding tensor for a batch of indices."""
        return self.code_proj(self.codebook(encoding_indices))

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
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Seq2SeqModelOutput]:

        # different to other models, Bart automatically creates decoder_input_ids from
        # input_ids if no decoder_input_ids are provided
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            if input_ids is None:
                raise ValueError(
                    "If no `decoder_input_ids` or `decoder_inputs_embeds` are "
                    "passed, `input_ids` cannot be `None`. Please pass either "
                    "`input_ids` or `decoder_input_ids` or `decoder_inputs_embeds`."
                )

            decoder_input_ids = shift_tokens_right(
                input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        sentence_representation = encoder_outputs[0][:, 0]

        code_indices = self.get_code_indices(sentence_representation)
        code_embedding = self.quantize(code_indices)

        # Should note that first compute the VQ loss and commitment loss to update the code embeddings
        # and then perform the straight-through estimate.
        # VQ loss
        vq_loss = self.mse_loss(code_embedding, sentence_representation.detach())
        # commitment loss
        commitment_loss = self.mse_loss(sentence_representation, code_embedding.detach())
        # assert code_embedding.requires_grad
        # assert sentence_representation.requires_grad

        # straight-through estimator
        code_embedding = sentence_representation + (code_embedding - sentence_representation).detach()

        # Generate `input_embeds` here to concat the latent code.
        decoder_inputs_embeds = self.decoder.embed_tokens(decoder_input_ids) * self.decoder.embed_scale
        decoder_inputs_embeds[:, 0] = code_embedding * self.decoder.embed_scale

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            # input_ids=decoder_input_ids,
            input_ids=None,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return DVAEModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
            z_encoder=sentence_representation,
            z_decoder=code_embedding,
            vq_loss=vq_loss,
            commitment_loss=commitment_loss,
        )


def ema_inplace(moving_avg: Tensor, new: Tensor, decay: float):
    moving_avg.data.mul_(decay).add_(new, alpha=(1 - decay))


"""
Reference:
BEiT-v2: https://github.com/microsoft/unilm/blob/master/beit2/norm_ema_quantizer.py#L123-L205 (TODO)
pytorch-vq-vae: https://github.com/zalandoresearch/pytorch-vq-vae/blob/master/vq-vae.ipynb (Follows this one currently.)
"""


class BartModelCodebookEMA(BartModel, ABC):
    def __init__(self, config: BartConfig, codebook_size: int, decay: float, code_dim: int = 128, epsilon: float = 1e-5,
                 add_cross_attn: bool = True, add_norm: bool = False, detach_extra_outputs: bool = False, mlp_after: bool = False,
                 add_point_wise: bool = False):
        super().__init__(config)
        assert decay > 0
        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.encoder = BartEncoder(config, self.shared)
        self.decoder = BartDecoder(config, self.shared)

        self.codebook_size = codebook_size
        if code_dim == config.d_model:
            self.vq_linear1 = nn.Identity()
            self.codebook = nn.Embedding(codebook_size, config.d_model)
            self.vq_linear2 = nn.Identity()
        else:
            self.vq_linear1 = nn.Linear(config.d_model, code_dim)
            self.codebook = nn.Embedding(codebook_size, code_dim)
            if mlp_after:
                self.vq_linear2 = nn.Sequential(nn.Linear(code_dim, config.d_model),
                                                nn.GELU(),
                                                nn.Linear(config.d_model, config.d_model))
            else:
                self.vq_linear2 = nn.Linear(code_dim, config.d_model)
        self.register_buffer("ema_cluster_size", torch.zeros(codebook_size))
        self.ema_weight = nn.Parameter(torch.Tensor(codebook_size, code_dim))
        self.ema_weight.data.normal_(mean=0.0, std=self.config.init_std)
        self.decay = decay
        self.epsilon = epsilon

        self.add_cross_attn = add_cross_attn
        for dec_layer in self.decoder.layers:
            for param in dec_layer.encoder_attn.parameters():
                param.requires_grad = False
            for param in dec_layer.encoder_attn_layer_norm.parameters():
                param.requires_grad = False

        self.add_norm = add_norm
        self.detach_extra_outputs = detach_extra_outputs
        self.add_point_wise = add_point_wise

        if dist.is_available() and dist.is_initialized():
            print("ddp is enable, so use ddp_reduce to sync the statistic_code_usage for each gpu!")
            self.all_reduce_fn = dist.all_reduce
        else:
            self.all_reduce_fn = nn.Identity()

        # Initialize weights and apply final processing
        self.post_init()
        self.mse_loss = torch.nn.MSELoss()

    def get_code_indices(self, flat_x):
        # compute L2 distance
        emb_weight = self.codebook.weight
        distances = (
                torch.sum(flat_x ** 2, dim=1, keepdim=True) +
                torch.sum(emb_weight ** 2, dim=1) -
                2. * torch.matmul(flat_x, emb_weight.t())
        )  # [N, M]
        encoding_indices = torch.argmin(distances, dim=1)  # [N,]
        return encoding_indices

    def quantize(self, encoding_indices):
        """Returns embedding tensor for a batch of indices."""
        return self.codebook(encoding_indices)

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
            extra_input_ids: Optional[torch.FloatTensor] = None,
            extra_attention_mask: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            inference: bool = False
    ) -> Union[Tuple, Seq2SeqModelOutput, DVAEInferenceOutput]:

        # different to other models, Bart automatically creates decoder_input_ids from
        # input_ids if no decoder_input_ids are provided
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            if input_ids is None:
                raise ValueError(
                    "If no `decoder_input_ids` or `decoder_inputs_embeds` are "
                    "passed, `input_ids` cannot be `None`. Please pass either "
                    "`input_ids` or `decoder_input_ids` or `decoder_inputs_embeds`."
                )

            decoder_input_ids = shift_tokens_right(
                input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        sentence_representation = encoder_outputs[0][:, 0]
        sentence_representation = self.vq_linear1(sentence_representation)
        if self.add_norm:
            sentence_representation = l2norm(sentence_representation)

        code_indices = self.get_code_indices(sentence_representation)
        if inference:
            return DVAEInferenceOutput(code_indices=code_indices)
        code_embedding = self.quantize(code_indices)

        encodings = functional.one_hot(code_indices, self.codebook_size).type(code_embedding.dtype)

        # Should note that first compute the VQ loss and commitment loss to update the code embeddings
        # and then perform the straight-through estimate.
        # # VQ loss
        # vq_loss = self.mse_loss(code_embedding, sentence_representation.detach())
        # commitment loss
        commitment_loss = self.mse_loss(sentence_representation, code_embedding.detach())

        # Update code embedding using EMA
        if self.training:
            bins = encodings.sum(dim=0)  # [codebook_size]
            self.all_reduce_fn(bins)
            # [codebook_size] <- [batch_size, codebook_size]
            # Calculate the amount of each code in current mini-batch
            self.ema_cluster_size = self.ema_cluster_size * self.decay + (1 - self.decay) * bins
            # ema_inplace(self.ema_cluster_size, bins, self.decay)

            # Laplace smoothing of the cluster size
            n = torch.sum(self.ema_cluster_size.data)
            self.ema_cluster_size = ((self.ema_cluster_size + self.epsilon) / (n + self.codebook_size * self.epsilon) * n)
            # zero_mask = (bins == 0)
            # bins = bins.masked_fill_(zero_mask, 1.)

            dw = torch.einsum("bn,bh->nh", encodings, sentence_representation)  # [codebook_size, h]
            self.all_reduce_fn(dw)

            # dw_normalized = dw / bins.unsqueeze(1)
            # dw_normalized = torch.where(zero_mask, self.ema_weight.data, dw_normalized)
            # self.ema_weight = nn.Parameter(self.ema_weight * self.decay + (1 - self.decay) * dw)
            self.ema_weight.data.copy_(self.ema_weight * self.decay + (1 - self.decay) * dw)

            # self.codebook.weight = nn.Parameter(self.ema_weight / self.ema_cluster_size.unsqueeze(1))
            if self.add_norm:
                self.codebook.weight.data.copy_(l2norm(self.ema_weight / self.ema_cluster_size.unsqueeze(1)))
            else:
                self.codebook.weight.data.copy_(self.ema_weight / self.ema_cluster_size.unsqueeze(1))

        # straight-through estimator
        code_embedding = sentence_representation + (code_embedding - sentence_representation).detach()
        code_embedding = self.vq_linear2(code_embedding)

        # print(self.decoder.embed_tokens.weight.size())
        # Generate `input_embeds` here to concat the latent code.
        decoder_inputs_embeds = self.decoder.embed_tokens(decoder_input_ids) * self.decoder.embed_scale
        if self.add_point_wise:
            decoder_inputs_embeds = decoder_inputs_embeds + (code_embedding * self.decoder.embed_scale).unsqueeze(1)
        decoder_inputs_embeds[:, 0] = code_embedding * self.decoder.embed_scale

        if self.add_cross_attn:
            if extra_input_ids is not None:
                extra_encoder_outputs = self.encoder(
                    input_ids=extra_input_ids,
                    attention_mask=extra_attention_mask,
                    return_dict=return_dict,
                )
                encoder_hidden_states = extra_encoder_outputs[0]
                encoder_attention_mask = extra_attention_mask
                if self.detach_extra_outputs:
                    encoder_hidden_states = encoder_hidden_states.detach()
            else:
                encoder_hidden_states = encoder_outputs[0]
                encoder_attention_mask = attention_mask
        else:
            encoder_hidden_states = encoder_attention_mask = None

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            # input_ids=decoder_input_ids,
            input_ids=None,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return DVAEModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
            z_encoder=sentence_representation,
            z_decoder=code_embedding,
            vq_loss=0.,
            commitment_loss=commitment_loss,
        )


class BartModelCodebookEMABottleNeck(BartModel, ABC):
    def __init__(self, config: BartConfig, codebook_size: int, decay: float, code_dim: int = 128, epsilon: float = 1e-5,
                 add_norm: bool = False, detach_extra_outputs: bool = False, mlp_after: bool = False, add_point_wise: bool = False):
        super().__init__(config)
        assert decay > 0
        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.encoder = BartEncoder(config, self.shared)
        self.decoder = BartDecoder(config, self.shared)

        self.codebook_size = codebook_size
        if code_dim == config.d_model:
            self.vq_linear1 = nn.Identity()
            self.codebook = nn.Embedding(codebook_size, config.d_model)
            self.vq_linear2 = nn.Identity()
        else:
            self.vq_linear1 = nn.Linear(config.d_model, code_dim)
            self.codebook = nn.Embedding(codebook_size, code_dim)
            if mlp_after:
                self.vq_linear2 = nn.Sequential(nn.Linear(code_dim, config.d_model),
                                                nn.GELU(),
                                                nn.Linear(config.d_model, config.d_model))
            else:
                self.vq_linear2 = nn.Linear(code_dim, config.d_model)
        self.register_buffer("ema_cluster_size", torch.zeros(codebook_size))
        self.ema_weight = nn.Parameter(torch.Tensor(codebook_size, code_dim))
        self.ema_weight.data.normal_(mean=0.0, std=self.config.init_std)
        self.decay = decay
        self.epsilon = epsilon

        self.add_norm = add_norm
        self.detach_extra_outputs = detach_extra_outputs
        self.add_point_wise = add_point_wise

        if dist.is_available() and dist.is_initialized():
            print("ddp is enable, so use ddp_reduce to sync the statistic_code_usage for each gpu!")
            self.all_reduce_fn = dist.all_reduce
        else:
            self.all_reduce_fn = nn.Identity()

        # Initialize weights and apply final processing
        self.post_init()
        self.mse_loss = torch.nn.MSELoss()

    def get_code_indices(self, flat_x):
        # compute L2 distance
        emb_weight = self.codebook.weight
        distances = (
                torch.sum(flat_x ** 2, dim=1, keepdim=True) +
                torch.sum(emb_weight ** 2, dim=1) -
                2. * torch.matmul(flat_x, emb_weight.t())
        )  # [N, M]
        encoding_indices = torch.argmin(distances, dim=1)  # [N,]
        return encoding_indices

    def quantize(self, encoding_indices):
        """Returns embedding tensor for a batch of indices."""
        return self.codebook(encoding_indices)

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
            extra_input_ids: Optional[torch.FloatTensor] = None,
            extra_attention_mask: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            inference: bool = False,
    ) -> Union[Tuple, Seq2SeqModelOutput, DVAEInferenceOutput]:

        # different to other models, Bart automatically creates decoder_input_ids from
        # input_ids if no decoder_input_ids are provided
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            if input_ids is None:
                raise ValueError(
                    "If no `decoder_input_ids` or `decoder_inputs_embeds` are "
                    "passed, `input_ids` cannot be `None`. Please pass either "
                    "`input_ids` or `decoder_input_ids` or `decoder_inputs_embeds`."
                )

            decoder_input_ids = shift_tokens_right(
                input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        sentence_representation = encoder_outputs[0][:, 0]
        sentence_representation = self.vq_linear1(sentence_representation)
        if self.add_norm:
            sentence_representation = l2norm(sentence_representation)

        code_indices = self.get_code_indices(sentence_representation)
        if inference:
            return DVAEInferenceOutput(code_indices=code_indices)
        code_embedding = self.quantize(code_indices)

        encodings = functional.one_hot(code_indices, self.codebook_size).type(code_embedding.dtype)

        # Should note that first compute the VQ loss and commitment loss to update the code embeddings
        # and then perform the straight-through estimate.
        # # VQ loss
        # vq_loss = self.mse_loss(code_embedding, sentence_representation.detach())
        # commitment loss
        commitment_loss = self.mse_loss(sentence_representation, code_embedding.detach())

        # Update code embedding using EMA
        if self.training:
            bins = encodings.sum(dim=0)  # [codebook_size]
            self.all_reduce_fn(bins)
            # [codebook_size] <- [batch_size, codebook_size]
            # Calculate the amount of each code in current mini-batch
            self.ema_cluster_size = self.ema_cluster_size * self.decay + (1 - self.decay) * bins
            # ema_inplace(self.ema_cluster_size, bins, self.decay)

            # Laplace smoothing of the cluster size
            n = torch.sum(self.ema_cluster_size.data)
            self.ema_cluster_size = ((self.ema_cluster_size + self.epsilon) / (n + self.codebook_size * self.epsilon) * n)
            # zero_mask = (bins == 0)
            # bins = bins.masked_fill_(zero_mask, 1.)

            dw = torch.einsum("bn,bh->nh", encodings, sentence_representation)  # [codebook_size, h]
            self.all_reduce_fn(dw)

            # dw_normalized = dw / bins.unsqueeze(1)
            # dw_normalized = torch.where(zero_mask, self.ema_weight.data, dw_normalized)
            self.ema_weight.data.copy_(self.ema_weight * self.decay + (1 - self.decay) * dw)

            if self.add_norm:
                self.codebook.weight.data.copy_(l2norm(self.ema_weight / self.ema_cluster_size.unsqueeze(1)))
            else:
                self.codebook.weight.data.copy_(self.ema_weight / self.ema_cluster_size.unsqueeze(1))

        # straight-through estimator
        code_embedding = sentence_representation + (code_embedding - sentence_representation).detach()
        code_embedding = self.vq_linear2(code_embedding)

        # Generate `input_embeds` here to concat the latent code.
        decoder_inputs_embeds = self.decoder.embed_tokens(decoder_input_ids) * self.decoder.embed_scale
        if self.add_point_wise:
            decoder_inputs_embeds = decoder_inputs_embeds + (code_embedding * self.decoder.embed_scale).unsqueeze(1)
        decoder_inputs_embeds[:, 0] = code_embedding * self.decoder.embed_scale
        # decoder_inputs_embeds[:, 0] = 0  # For test only.

        if extra_input_ids is not None:
            extra_outputs = self.encoder(
                input_ids=extra_input_ids,
                attention_mask=extra_attention_mask,
                return_dict=return_dict,
            )
            orig_representation = extra_outputs[0][:, :1]
            if self.detach_extra_outputs:
                orig_representation = orig_representation.detach()
            decoder_inputs_embeds = torch.cat([decoder_inputs_embeds[:, :1], orig_representation, decoder_inputs_embeds[:, 1:]], dim=1)

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            # input_ids=decoder_input_ids,
            input_ids=None,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return DVAEModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
            z_encoder=sentence_representation,
            z_decoder=code_embedding,
            vq_loss=0.,
            commitment_loss=commitment_loss,
        )

