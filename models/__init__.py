import torch
from torch import nn

from general_util.logger import get_child_logger

logger = get_child_logger(__name__)


class DotProductDistance(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, q, k):
        scores = torch.einsum("ah,bh->ab", q, k)
        return {"logits": scores}
