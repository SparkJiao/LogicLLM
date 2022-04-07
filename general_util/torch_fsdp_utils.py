from functools import partial

from torch.distributed.fsdp import FullyShardedDataParallel, CPUOffload
from torch.distributed.fsdp.wrap import default_auto_wrap_policy

"""
Refer to https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/,
and https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html.
"""


def torch_fsdp_initialize_default(model,
                                  device,
                                  cpu_offload=False,
                                  min_num_params: int = 1e8):
    my_auto_wrap_policy = partial(default_auto_wrap_policy,
                                  min_num_params=min_num_params)

    fsdp_model = FullyShardedDataParallel(
        model,
        fsdp_auto_wrap_policy=my_auto_wrap_policy,
        cpu_offload=CPUOffload(offload_params=cpu_offload)
    )

    if not cpu_offload:
        fsdp_model.to(device)

    return fsdp_model
