from abc import ABC
from typing import Optional

import torch
from torch import nn
from torch.nn.functional import gumbel_softmax
import warnings
import torch.nn.functional as F

try:
    from torch.overrides import has_torch_function, handle_torch_function
except ImportError as e:
    from torch._overrides import has_torch_function, handle_torch_function

from transformers.models.roberta.modeling_roberta import RobertaModel, RobertaPreTrainedModel, RobertaForMaskedLM, RobertaConfig


class Quantizer(nn.Module):
    def get_code_indices(self, hidden_states: torch.Tensor):
        raise NotImplementedError

    def quantize(self, code_indices: torch.LongTensor):
        raise NotImplementedError


class StraightEstimator(Quantizer, ABC):
    def __init__(self):
        super().__init__()
        pass


# Copied from https://github.com/THUDM/CogView/blob/main/vqvae/vqvae_zc.py
class CogViewEMAQuantizer(nn.Module):
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps

        embed = torch.randn(dim, n_embed)
        torch.nn.init.xavier_uniform_(embed, gain=torch.nn.init.calculate_gain('tanh'))
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(n_embed))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, x, continuous_relax=False, temperature=1., hard=False):
        return self.forward_(x, continuous_relax, temperature, hard)

    def forward_(self, input, continuous_relax=False, temperature=1., hard=False):
        flatten = input.reshape(-1, self.dim)
        dist = (
                flatten.pow(2).sum(1, keepdim=True)
                - 2 * flatten @ self.embed
                + self.embed.pow(2).sum(0, keepdim=True)
        )  # dist map, shape=[*, n_embed]

        if not continuous_relax:
            # argmax + lookup
            _, embed_ind = (-dist).max(1)
            embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
            embed_ind = embed_ind.view(*input.shape[:-1])
            quantize = self.embed_code(embed_ind)
        elif not hard:
            # gumbel softmax weighted sum
            embed_soft, embed_ind = gumbel_softmax(-dist, tau=temperature, hard=False)
            embed_ind = embed_ind.view(*input.shape[:-1])
            embed_soft = embed_soft.view(*input.shape[:-1], self.n_embed)
            quantize = embed_soft @ self.embed.transpose(0, 1)
        else:
            # gumbel softmax hard lookup
            embed_onehot, embed_ind = gumbel_softmax(-dist, tau=temperature, hard=True)
            embed_ind = embed_ind.view(*input.shape[:-1])
            quantize = self.embed_code(embed_ind)

        if self.training and ((continuous_relax and hard) or (not continuous_relax)):
            embed_onehot_sum = embed_onehot.sum(0)
            embed_sum = flatten.transpose(0, 1) @ embed_onehot

            # dist_fn.all_reduce(embed_onehot_sum)
            # dist_fn.all_reduce(embed_sum)

            self.cluster_size.data.mul_(self.decay).add_(
                embed_onehot_sum, alpha=1 - self.decay
            )
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = (
                    (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        if not continuous_relax:
            diff = (quantize.detach() - input).pow(2).mean()
            quantize = input + (quantize - input).detach()
        else:
            # maybe need replace a KL term here
            qy = (-dist).softmax(-1)
            diff = torch.sum(qy * torch.log(qy * self.n_embed + 1e-20), dim=-1).mean()  # KL
            # diff = (quantize - input).pow(2).mean().detach() # gumbel softmax do not need diff
            quantize = quantize.to(memory_format=torch.channels_last)
        return quantize, diff, embed_ind

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))


Tensor = torch.Tensor


def gumbel_softmax(logits, tau=1, hard=False, eps=1e-10, dim=-1):
    # type: (Tensor, float, bool, float, int) -> Tensor
    r"""
    Samples from the Gumbel-Softmax distribution (`Link 1`_  `Link 2`_) and optionally discretizes.
    Args:
      logits: `[..., num_features]` unnormalized log probabilities
      tau: non-negative scalar temperature
      hard: if ``True``, the returned samples will be discretized as one-hot vectors,
            but will be differentiated as if it is the soft sample in autograd
      eps:
      dim (int): A dimension along which softmax will be computed. Default: -1.
    Returns:
      Sampled tensor of same shape as `logits` from the Gumbel-Softmax distribution.
      If ``hard=True``, the returned samples will be one-hot, otherwise they will
      be probability distributions that sum to 1 across `dim`.
    .. note::
      This function is here for legacy reasons, may be removed from nn.Functional in the future.
    .. note::
      The main trick for `hard` is to do  `y_hard - y_soft.detach() + y_soft`
      It achieves two things:
      - makes the output value exactly one-hot
      (since we add then subtract y_soft value)
      - makes the gradient equal to y_soft gradient
      (since we strip all other gradients)
    Examples::
        >>> logits = torch.randn(20, 32)
        >>> # Sample soft categorical using reparametrization trick:
        >>> F.gumbel_softmax(logits, tau=1, hard=False)
        >>> # Sample hard categorical using "Straight-through" trick:
        >>> F.gumbel_softmax(logits, tau=1, hard=True)
    .. _Link 1:
        https://arxiv.org/abs/1611.00712
    .. _Link 2:
        https://arxiv.org/abs/1611.01144
    """
    if not torch.jit.is_scripting():
        if type(logits) is not Tensor and has_torch_function((logits,)):
            return handle_torch_function(
                gumbel_softmax, (logits,), logits, tau=tau, hard=hard, eps=eps, dim=dim)
    if eps != 1e-10:
        warnings.warn("`eps` parameter is deprecated and has no effect.")

    gumbels = -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()  # ~Gumbel(0,1)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
        return ret, index
    else:
        # Reparametrization trick.
        ret = y_soft
        index = y_soft.max(dim, keepdim=True)[1]
        return ret,
