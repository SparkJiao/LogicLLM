import math
from abc import ABC
from dataclasses import dataclass
from typing import List

import torch
from torch import nn, Tensor, _softmax_backward_data
from torch.nn import CrossEntropyLoss, LayerNorm
from transformers.modeling_outputs import MultipleChoiceModelOutput
from transformers.models.deberta.modeling_deberta import DebertaPreTrainedModel, DebertaModel, DebertaConfig, \
    ContextPooler, StableDropout, ACT2FN, DebertaEncoder, build_relative_position, c2p_dynamic_expand, p2c_dynamic_expand, \
    pos_dynamic_expand, SequenceClassifierOutput

from general_util.logger import get_child_logger
from general_util.mixin import LogMixin
from modules import layers

logger = get_child_logger("DeBERTa")


class XSoftmax(torch.autograd.Function):
    """
    Masked Softmax which is optimized for saving memory

    Args:
        input (`torch.tensor`): The input tensor that will apply softmax.
        mask (`torch.IntTensor`):
            The mask matrix where 0 indicate that element will be ignored in the softmax calculation.
        dim (int): The dimension that will apply softmax

    Example:

    ```python
    >>> import torch
    >>> from transformers.models.deberta.modeling_deberta import XSoftmax

    >>> # Make a tensor
    >>> x = torch.randn([4, 20, 100])

    >>> # Create a mask
    >>> mask = (x > 0).int()

    >>> # Specify the dimension to apply softmax
    >>> dim = -1

    >>> y = XSoftmax.apply(x, mask, dim)
    ```"""

    @staticmethod
    def forward(self, input, mask, dim):
        self.dim = dim
        rmask = ~(mask.bool())

        output = input.masked_fill(rmask, float("-inf"))
        output = torch.softmax(output, self.dim)
        output.masked_fill_(rmask, 0)
        self.save_for_backward(output)
        return output

    @staticmethod
    def backward(self, grad_output):
        (output,) = self.saved_tensors
        inputGrad = _softmax_backward_data(grad_output, output, self.dim, output.dtype)
        return inputGrad, None, None

    @staticmethod
    def symbolic(g, self, mask, dim):
        import torch.onnx.symbolic_helper as sym_help
        from torch.onnx.symbolic_opset9 import masked_fill, softmax

        mask_cast_value = g.op("Cast", mask, to_i=sym_help.cast_pytorch_to_onnx["Long"])
        r_mask = g.op(
            "Cast",
            g.op("Sub", g.op("Constant", value_t=torch.tensor(1, dtype=torch.int64)), mask_cast_value),
            to_i=sym_help.cast_pytorch_to_onnx["Byte"],
        )
        output = masked_fill(g, self, r_mask, g.op("Constant", value_t=torch.tensor(float("-inf"))))
        output = softmax(g, output, dim)
        return masked_fill(g, output, r_mask, g.op("Constant", value_t=torch.tensor(0, dtype=torch.uint8)))


class DisentangledSelfAttention(nn.Module):
    """
    Disentangled self-attention module

    Parameters:
        config (`str`):
            A model config class instance with the configuration to build a new model. The schema is similar to
            *BertConfig*, for more details, please refer [`DebertaConfig`]

    """

    def __init__(self, config: DebertaConfig):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.in_proj = nn.Linear(config.hidden_size, self.all_head_size * 3, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(self.all_head_size, dtype=torch.float))
        self.v_bias = nn.Parameter(torch.zeros(self.all_head_size, dtype=torch.float))
        self.pos_att_type = config.pos_att_type if config.pos_att_type is not None else []

        self.relative_attention = getattr(config, "relative_attention", False)
        self.talking_head = getattr(config, "talking_head", False)

        if self.talking_head:
            self.head_logits_proj = nn.Linear(config.num_attention_heads, config.num_attention_heads, bias=False)
            self.head_weights_proj = nn.Linear(config.num_attention_heads, config.num_attention_heads, bias=False)

        if self.relative_attention:
            self.max_relative_positions = getattr(config, "max_relative_positions", -1)
            if self.max_relative_positions < 1:
                self.max_relative_positions = config.max_position_embeddings
            self.pos_dropout = StableDropout(config.hidden_dropout_prob)

            if "c2p" in self.pos_att_type:
                self.pos_proj = nn.Linear(config.hidden_size, self.all_head_size, bias=False)
            if "p2c" in self.pos_att_type:
                self.pos_q_proj = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = StableDropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, -1)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
            self,
            hidden_states,
            attention_mask,
            output_attentions=False,
            query_states=None,
            relative_pos=None,
            rel_embeddings=None,
    ):
        """
        Call the module

        Args:
            hidden_states (`torch.FloatTensor`):
                Input states to the module usually the output from previous layer, it will be the Q,K and V in
                *Attention(Q,K,V)*

            attention_mask (`torch.ByteTensor`):
                An attention mask matrix of shape [*B*, *N*, *N*] where *B* is the batch size, *N* is the maximum
                sequence length in which element [i,j] = *1* means the *i* th token in the input can attend to the *j*
                th token.

            output_attentions (`bool`, optional):
                Whether return the attention matrix.

            query_states (`torch.FloatTensor`, optional):
                The *Q* state in *Attention(Q,K,V)*.

            relative_pos (`torch.LongTensor`):
                The relative position encoding between the tokens in the sequence. It's of shape [*B*, *N*, *N*] with
                values ranging in [*-max_relative_positions*, *max_relative_positions*].

            rel_embeddings (`torch.FloatTensor`):
                The embedding of relative distances. It's a tensor of shape [\\(2 \\times
                \\text{max_relative_positions}\\), *hidden_size*].


        """
        if query_states is None:
            qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
            query_layer, key_layer, value_layer = self.transpose_for_scores(qp).chunk(3, dim=-1)
        else:

            def linear(w, b, x):
                if b is not None:
                    return torch.matmul(x, w.t()) + b.t()
                else:
                    return torch.matmul(x, w.t())  # + b.t()

            ws = self.in_proj.weight.chunk(self.num_attention_heads * 3, dim=0)
            qkvw = [torch.cat([ws[i * 3 + k] for i in range(self.num_attention_heads)], dim=0) for k in range(3)]
            qkvb = [None] * 3

            q = linear(qkvw[0], qkvb[0], query_states)
            k, v = [linear(qkvw[i], qkvb[i], hidden_states) for i in range(1, 3)]
            query_layer, key_layer, value_layer = [self.transpose_for_scores(x) for x in [q, k, v]]

        query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
        value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])

        rel_att = None
        # Take the dot product between "query" and "key" to get the raw attention scores.
        scale_factor = 1 + len(self.pos_att_type)
        scale = math.sqrt(query_layer.size(-1) * scale_factor)
        query_layer = query_layer / scale
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        if self.relative_attention:
            rel_embeddings = self.pos_dropout(rel_embeddings)
            rel_att = self.disentangled_att_bias(query_layer, key_layer, relative_pos, rel_embeddings, scale_factor)

        if rel_att is not None:
            attention_scores = attention_scores + rel_att

        # bxhxlxd
        if self.talking_head:
            attention_scores = self.head_logits_proj(attention_scores.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
        attention_probs = self.dropout(attention_probs)
        if self.talking_head:
            attention_probs = self.head_weights_proj(attention_probs.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (-1,)
        context_layer = context_layer.view(*new_context_layer_shape)
        if output_attentions:
            return context_layer, attention_probs
        else:
            return context_layer

    def disentangled_att_bias(self, query_layer, key_layer, relative_pos, rel_embeddings, scale_factor):
        if relative_pos is None:
            q = query_layer.size(-2)
            relative_pos = build_relative_position(q, key_layer.size(-2), query_layer.device)
        if relative_pos.dim() == 2:
            relative_pos = relative_pos.unsqueeze(0).unsqueeze(0)
        elif relative_pos.dim() == 3:
            relative_pos = relative_pos.unsqueeze(1)
        # bxhxqxk
        elif relative_pos.dim() != 4:
            raise ValueError(f"Relative position ids must be of dim 2 or 3 or 4. {relative_pos.dim()}")

        att_span = min(max(query_layer.size(-2), key_layer.size(-2)), self.max_relative_positions)
        relative_pos = relative_pos.long().to(query_layer.device)
        rel_embeddings = rel_embeddings[
                         self.max_relative_positions - att_span: self.max_relative_positions + att_span, :
                         ].unsqueeze(0)

        score = 0

        # content->position
        if "c2p" in self.pos_att_type:
            pos_key_layer = self.pos_proj(rel_embeddings)
            pos_key_layer = self.transpose_for_scores(pos_key_layer)
            c2p_att = torch.matmul(query_layer, pos_key_layer.transpose(-1, -2))
            c2p_pos = torch.clamp(relative_pos + att_span, 0, att_span * 2 - 1)
            c2p_att = torch.gather(c2p_att, dim=-1, index=c2p_dynamic_expand(c2p_pos, query_layer, relative_pos))
            score += c2p_att

        # position->content
        if "p2c" in self.pos_att_type:
            pos_query_layer = self.pos_q_proj(rel_embeddings)
            pos_query_layer = self.transpose_for_scores(pos_query_layer)
            pos_query_layer /= math.sqrt(pos_query_layer.size(-1) * scale_factor)
            if query_layer.size(-2) != key_layer.size(-2):
                r_pos = build_relative_position(key_layer.size(-2), key_layer.size(-2), query_layer.device)
            else:
                r_pos = relative_pos
            p2c_pos = torch.clamp(-r_pos + att_span, 0, att_span * 2 - 1)
            p2c_att = torch.matmul(key_layer, pos_query_layer.transpose(-1, -2))
            p2c_att = torch.gather(
                p2c_att, dim=-1, index=p2c_dynamic_expand(p2c_pos, query_layer, key_layer)
            ).transpose(-1, -2)

            if query_layer.size(-2) != key_layer.size(-2):
                pos_index = relative_pos[:, :, :, 0].unsqueeze(-1)
                p2c_att = torch.gather(p2c_att, dim=-2, index=pos_dynamic_expand(pos_index, p2c_att, key_layer))
            score += p2c_att

        return score


class BertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding_size = getattr(config, 'embedding_size', config.hidden_size)
        self.dense = nn.Linear(config.hidden_size, self.embedding_size)
        self.transform_act_fn = ACT2FN[config.hidden_act] if isinstance(config.hidden_act, str) else config.hidden_act

        self.LayerNorm = LayerNorm(self.embedding_size, config.layer_norm_eps, elementwise_affine=True)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        # hidden_states = masked_layer_norm(self.LayerNorm, hidden_states)
        hidden_states = self.LayerNorm(hidden_states)

        logits = self.decoder(hidden_states)
        return logits


# Copied from https://github.com/microsoft/DeBERTa/blob/master/DeBERTa/apps/models/masked_language_model.py#L30-L82
class EnhancedMaskDecoder(torch.nn.Module):
    def __init__(self, config: DebertaConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size

        self.position_biased_input = getattr(config, 'position_biased_input', True)
        self.lm_head = BertLMPredictionHead(config)

    def forward(self, encoded_layers: List[Tensor], z_states: Tensor, attention_mask: Tensor, encoder: DebertaEncoder,
                mlm_labels: Tensor, relative_pos=None):
        mlm_ctx_layers = self.emd_context_layer(encoded_layers, z_states, attention_mask, encoder, relative_pos=relative_pos)
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-1)

        mlm_labels = mlm_labels.reshape(-1)
        mlm_index = (mlm_labels > 0).nonzero().view(-1)
        mlm_ctx_states = mlm_ctx_layers[-1].reshape(-1, mlm_ctx_layers[-1].size(-1)).index_select(0, index=mlm_index)
        mlm_target_ids = mlm_labels.index_select(0, index=mlm_index)
        mlm_logits = self.lm_head(mlm_ctx_states)
        mlm_loss = loss_fct(mlm_logits.reshape(-1, self.vocab_size), mlm_target_ids)

        return mlm_logits, mlm_target_ids, mlm_loss

    def emd_context_layer(self, encoded_layers: List[Tensor], z_states: Tensor, attention_mask: Tensor,
                          encoder: DebertaEncoder, relative_pos=None):
        attention_mask = encoder.get_attention_mask(attention_mask)
        hidden_states = encoded_layers[-2]
        if not self.position_biased_input:
            enc_layers = [encoder.layer[-1] for _ in range(2)]
            z_states = z_states + hidden_states
            query_states = z_states
            query_mask = attention_mask
            outputs = []
            rel_embeddings = encoder.get_rel_embedding()

            for layer in enc_layers:
                # TODO: pass relative pos ids
                output = layer(hidden_states, query_mask, query_states=query_states, relative_pos=relative_pos,
                               rel_embeddings=rel_embeddings)
                query_states = output
                outputs.append(query_states)
        else:
            outputs = [encoded_layers[-1]]
            raise RuntimeError()  # For debug. ``position_biased_input==False``

        return outputs


# Copied from transformers.models.deberta.modeling_deberta.ContextPooler and modify the pooler hidden size
class ContextPoolerE(nn.Module):
    def __init__(self, config, mlp_hidden_size: 768):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, mlp_hidden_size)
        self.dropout = StableDropout(config.pooler_dropout)
        self.config = config

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.

        context_token = hidden_states[:, 0]
        context_token = self.dropout(context_token)
        pooled_output = self.dense(context_token)
        pooled_output = ACT2FN[self.config.pooler_hidden_act](pooled_output)
        return pooled_output

    @property
    def output_dim(self):
        return self.config.hidden_size


@dataclass
class MultipleChoicePreTrainModelOutput(MultipleChoiceModelOutput):
    mlm_loss: torch.FloatTensor = None
    mlm_acc: torch.FloatTensor = None
    cls_loss: torch.FloatTensor = None
    cls_acc: torch.FloatTensor = None


class DebertaForMultipleChoicePreTrain(DebertaPreTrainedModel, LogMixin, ABC):
    def __init__(self, config: DebertaConfig,
                 mlp_hidden_size: int = 768,
                 use_stable_embedding: bool = False,
                 fs_checkpoint: bool = False,
                 fs_checkpoint_offload_to_cpu: bool = False,
                 fs_checkpoint_start_layer_id: int = 0):
        super().__init__(config)

        config.update({
            "mlp_hidden_size": mlp_hidden_size,
        })

        self.config = config

        self.deberta = DebertaModel(config)
        # Hack here. Since ``position_based_input==False``, the weights won't be loaded.
        self.embedding_size = getattr(config, "embedding_size", config.hidden_size)
        # self.deberta.embeddings.position_embeddings = nn.Embedding(config.max_position_embeddings, self.embedding_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, self.embedding_size)
        self.lm_predictions = EnhancedMaskDecoder(config)

        for layer in self.deberta.encoder.layer:
            layer.attention.self = DisentangledSelfAttention(config)

        self.pooler = ContextPoolerE(config, mlp_hidden_size=mlp_hidden_size)
        self.dropout = StableDropout(config.hidden_dropout_prob)
        self.cls = nn.Linear(mlp_hidden_size, 1)

        self.init_weights()
        self.init_metric("loss", "acc", "mlm_loss", "mlm_acc", "cls_loss")

    def get_output_embeddings(self):
        return self.lm_predictions.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        self.lm_predictions.lm_head.decoder = new_embeddings

    def get_input_embeddings(self):
        return self.deberta.embeddings.word_embeddings

    def get_position_embeddings(self, seq_length):
        position_ids = self.deberta.embeddings.position_ids[:, :seq_length].to(self.position_embeddings.weight.device)
        position_embeddings = self.position_embeddings(position_ids)
        return position_embeddings

    @staticmethod
    def fold_tensor(x: Tensor):
        if x is None:
            return x
        return x.reshape(-1, x.size(-1))

    def mlm_forward(self, mlm_input_ids: Tensor, mlm_attention_mask: Tensor, mlm_labels: Tensor = None, return_dict=None):

        mlm_outputs = self.deberta(
            mlm_input_ids,
            attention_mask=mlm_attention_mask,
            output_hidden_states=True,
            return_dict=return_dict
        )

        encoded_layers = mlm_outputs[1]
        z_states = self.get_position_embeddings(mlm_input_ids.size(1))

        mlm_logits, mlm_target_ids, mlm_loss = self.lm_predictions(encoded_layers=encoded_layers,
                                                                   z_states=z_states,
                                                                   attention_mask=mlm_attention_mask,
                                                                   encoder=self.deberta.encoder,
                                                                   mlm_labels=mlm_labels,
                                                                   relative_pos=None)
        return mlm_logits, mlm_target_ids, mlm_loss

    def forward(self,
                input_ids: Tensor,
                attention_mask: Tensor = None,
                token_type_ids: Tensor = None,
                labels: Tensor = None,
                mlm_input_ids: Tensor = None,
                mlm_attention_mask: Tensor = None,
                mlm_labels: Tensor = None,
                return_dict=None,
                **kwargs):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
            config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        num_choices = input_ids.shape[1]

        input_ids = self.fold_tensor(input_ids)
        attention_mask = self.fold_tensor(attention_mask)
        token_type_ids = self.fold_tensor(token_type_ids)

        outputs = self.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=return_dict,
        )

        logits = self.cls(self.dropout(self.pooler(outputs[0])))
        reshaped_logits = logits.view(-1, num_choices)

        loss = 0.
        mlm_loss = 0.
        cls_loss = 0.
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            cls_loss = loss_fct(reshaped_logits, labels)
            loss = loss + cls_loss

            if mlm_labels is not None:
                if mlm_attention_mask is None:
                    mlm_attention_mask = attention_mask.reshape(reshaped_logits.size(0), num_choices, -1)[:, 0]

                mlm_scores, mlm_labels, mlm_loss = self.mlm_forward(mlm_input_ids, mlm_attention_mask, mlm_labels, return_dict=return_dict)
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
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return MultipleChoicePreTrainModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            mlm_loss=mlm_loss.detach(),
            cls_loss=cls_loss.detach(),
        )


class DebertaForMultipleChoice(DebertaPreTrainedModel, LogMixin, ABC):
    def __init__(self, config: DebertaConfig, override_pooler: bool = False):
        super().__init__(config)

        self.config = config
        self.override_pooler = override_pooler

        self.deberta = DebertaModel(config)
        for layer in self.deberta.encoder.layer:
            layer.attention.self = DisentangledSelfAttention(config)

        if self.override_pooler:
            mlp_hidden_size = getattr(config, "mlp_hidden_size", config.hidden_size)
            self.pooler = ContextPoolerE(config, mlp_hidden_size=mlp_hidden_size)
            self.cls = nn.Linear(mlp_hidden_size, 1)
        else:
            self.n_pooler = ContextPooler(config)
            output_dim = self.n_pooler.output_dim
            self.classifier = nn.Linear(output_dim, 1)

        drop_out = getattr(config, "cls_dropout", None)
        drop_out = self.config.hidden_dropout_prob if drop_out is None else drop_out
        self.dropout = StableDropout(drop_out)

        self.init_weights()

        self.init_metric("loss", "acc")

    def get_input_embeddings(self):
        return self.deberta.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        self.deberta.set_input_embeddings(new_embeddings)

    @staticmethod
    def fold_tensor(x: Tensor):
        if x is None:
            return x
        return x.reshape(-1, x.size(-1))

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            **kwargs
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        num_choices = input_ids.shape[1]

        input_ids = self.fold_tensor(input_ids)
        attention_mask = self.fold_tensor(attention_mask)
        token_type_ids = self.fold_tensor(token_type_ids)

        outputs = self.deberta(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        encoder_layer = outputs[0]
        if self.override_pooler:
            pooled_output = self.pooler(encoder_layer)
            pooled_output = self.dropout(pooled_output)
            logits = self.cls(pooled_output).reshape(-1, num_choices)
        else:
            pooled_output = self.n_pooler(encoder_layer)
            pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output).reshape(-1, num_choices)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(logits, labels)

            if not self.training:
                acc, true_label_num = layers.get_accuracy(logits, labels)
                self.eval_metrics.update("acc", val=acc, n=true_label_num)
                self.eval_metrics.update("loss", val=loss.item(), n=true_label_num)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output
        else:
            return MultipleChoiceModelOutput(
                loss=loss,
                logits=logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )


class DebertaForSequenceClassification(DebertaPreTrainedModel, LogMixin, ABC):
    def __init__(self, config: DebertaConfig, override_pooler: bool = False, freeze_encoder: bool = False):
        super().__init__(config)

        num_labels = getattr(config, "num_labels", 2)
        self.config = config
        self.num_labels = num_labels
        self.override_pooler = override_pooler

        self.deberta = DebertaModel(config)
        for layer in self.deberta.encoder.layer:
            layer.attention.self = DisentangledSelfAttention(config)
        # self.pooler = ContextPooler(config)
        # output_dim = self.pooler.output_dim

        self.freeze_encoder = freeze_encoder
        if self.freeze_encoder:
            for param in self.deberta.encoder.parameters():
                param.requires_grad = False

        if self.override_pooler:
            mlp_hidden_size = getattr(config, "mlp_hidden_size", config.hidden_size)
            self.pooler = ContextPoolerE(config, mlp_hidden_size)
            output_dim = mlp_hidden_size
        else:
            self.n_pooler = ContextPooler(config)
            output_dim = self.n_pooler.output_dim

        self.classifier = nn.Linear(output_dim, num_labels)
        drop_out = getattr(config, "cls_dropout", None)
        drop_out = self.config.hidden_dropout_prob if drop_out is None else drop_out
        self.dropout = StableDropout(drop_out)

        # Initialize weights and apply final processing
        self.post_init()

        self.init_metric("loss", "acc")

    def get_input_embeddings(self):
        return self.deberta.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        self.deberta.set_input_embeddings(new_embeddings)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            **kwargs
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.deberta(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        encoder_layer = outputs[0]
        if self.override_pooler:
            pooled_output = self.dropout(self.pooler(encoder_layer))
        else:
            pooled_output = self.dropout(self.n_pooler(encoder_layer))
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            if not self.training:
                acc, true_label_num = layers.get_accuracy(logits, labels)
                self.eval_metrics.update("acc", val=acc, n=true_label_num)
                self.eval_metrics.update("loss", val=loss.item(), n=true_label_num)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )
