import copy
import pickle
from abc import ABC
from dataclasses import dataclass
from typing import Union, Dict
from torch.nn import functional as F
import math

import torch
import transformers
from torch import nn, Tensor
from torch.nn import CrossEntropyLoss
from transformers.models.roberta.modeling_roberta import RobertaConfig
from transformers import AutoTokenizer

from general_util.logger import get_child_logger
from models.roberta import RobertaForMultipleChoiceForPreTrain, RelDecoderHead, MultipleChoicePreTrainModelOutput, attentive_pooling
from modules import layers

logger = get_child_logger("VEM")


class VEM(nn.Module):
    def __init__(self, temp: float = 0.05, alpha: float = 1.0, eta: int = 10, m: int = 4096):
        super().__init__()

        self.temp = temp
        self.alpha = alpha
        self.eta = eta
        self.B = None
        self.m = m

    def forward(self, q: Tensor, k: Tensor):
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        N = q.size(0)
        M = self.m  # *2 #* 2
        # M = 0
        # if not hasattr(cls, "B"):
        if self.B is None:
            self.B = torch.zeros(M, q.size(1), device=q.device, requires_grad=False)  # M*K
            # cls.B =  torch.randn(M, q.size(1), device=q.device, requires_grad=False) #M*K
            self.B = F.normalize(self.B, dim=-1)
        else:
            M = self.B.size(0)

        alpha = self.alpha
        eta = self.eta  # sampling steps

        with torch.no_grad():
            for i in range(eta):
                # for q
                L = self.B @ q.transpose(0, 1)  # /cls.model_args.temp #M*N
                # L_k = cls.B@k.transpose(0,1)#/cls.model_args.temp #M*N

                L_norm = (L / self.temp).softmax(dim=-1)  # M*N
                # L_norm = F.gumbel_softmax(L, tau = cls.model_args.temp, dim=-1, hard=False) #M*N
                # L_norm = F.gumbel_softmax(L, tau = 5.0, dim=-1, hard=False) #M*N
                # L_norm_k = (L_k/cls.model_args.temp).softmax(dim=-1) #M*N

                delta_B = L_norm @ q / N - (L_norm * L).mean(dim=1, keepdim=True) * self.B  # M*K
                # delta_B_k = L_norm_k @ k / N - (L_norm_k * L_k).mean(dim=1, keepdim=True) * cls.B #M*K

                delta_B1 = delta_B
                # delta_B1_k = delta_B_k

                delta_B = self.B @ self.B.transpose(0, 1) @ delta_B / M + self.B
                # delta_B_k = cls.B @ cls.B.transpose(0,1) @ delta_B_k / M + cls.B

                # B = cls.B + alpha / (i+1) * delta_B
                # print("delta_B:", delta_B)
                Q = torch.randn(M, q.size(1), device=q.device, requires_grad=False)
                # B = cls.B + alpha  / (i+1) * delta_B + math.sqrt(2 * alpha / (i+1)) * Q
                B = self.B + alpha * 0.5 / (i + 1) * delta_B + alpha * 0.5 / (i + 1) * delta_B1 + math.sqrt(2 * alpha / (i + 1)) * Q
                # B = cls.B + alpha *0.25  / (i+1) * delta_B + alpha * 0.25 / (i+1) * delta_B1 + math.sqrt(2 * alpha / (i+1)) * Q
                # B = B + alpha *0.25  / (i+1) * delta_B_k + alpha * 0.25 / (i+1) * delta_B1_k

                # C = alpha *0.5  / (i+1) * delta_B
                # D = alpha * 0.5 / (i+1) * delta_B1
                # E = math.sqrt(2 * alpha / (i+1)) * Q

                # G = torch.cat([C.unsqueeze(dim=-1), D.unsqueeze(dim=-1), E.unsqueeze(dim=-1)], dim=-1)

                # W = G.softmax(dim=-1)
                # G = (G * W).sum(dim=-1)
                # C = alpha *0.5  / (i+1) * delta_B
                # D = alpha * 0.5 / (i+1) * delta_B1
                # E = math.sqrt(2 * alpha / (i+1)) * Q

                # G = torch.cat([C.unsqueeze(dim=-1), D.unsqueeze(dim=-1), E.unsqueeze(dim=-1)], dim=-1)

                # W = G.softmax(dim=-1)
                # G = (G * W).sum(dim=-1)

                # B = cls.B + G

                self.B = F.normalize(B, dim=-1)
                # print("cls.B:", cls.B)

        logit_neg = (q @ self.B.transpose(0, 1) / self.temp)  # N, * M
        cos_sim = torch.einsum("bh,bh->b", q, k).unsqueeze(-1) / self.temp
        # loss_fct = nn.CrossEntropyLoss()
        # loss = loss_fct(logits, zeros)

        # cos_sim = torch.cat([cos_sim, logit_neg, logit_neg2], dim=-1) #N, M+N
        cos_sim = torch.cat([cos_sim, logit_neg], dim=-1)  # N, M+N
        # cos_sim = torch.cat([cos_sim, logit_neg], dim=-1) #N, M+N
        # loss = loss_fct(cos_sim, labels)
        # loss = loss + loss1

        return cos_sim

