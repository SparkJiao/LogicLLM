from functools import partial

import torch
from torch.distributed.fsdp import FullyShardedDataParallel, CPUOffload, MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy, size_based_auto_wrap_policy
from transformers.models.bart.modeling_bart import BartEncoderLayer
from general_util.logger import get_child_logger


"""
Refer to https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/,
and https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html.
"""

logger = get_child_logger(__name__)



def torch_fsdp_initialize_default(model,
                                  device,
                                  fp16: bool = True,
                                  fp16_bfloat16: bool = False,
                                  cpu_offload=False):
    my_auto_wrap_policy = partial(transformer_auto_wrap_policy,
                                  transformer_layer_cls={BartEncoderLayer})

    ignored_modules = []
    for dec_layer in model.model.decoder.layers:
        ignored_modules.append(dec_layer.encoder_attn)
        ignored_modules.append(dec_layer.encoder_attn_layer_norm)

    if fp16:
        fp16_type = torch.float16 if not fp16_bfloat16 else torch.bfloat16
        mixed_precision_cfg = MixedPrecision(param_dtype=fp16_type, reduce_dtype=fp16_type, buffer_dtype=fp16_type)
    else:
        mixed_precision_cfg = None

    fsdp_model = FullyShardedDataParallel(
        model,
        mixed_precision=mixed_precision_cfg,
        auto_wrap_policy=my_auto_wrap_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        cpu_offload=CPUOffload(offload_params=cpu_offload),
        ignored_modules=ignored_modules,
        device_id=device,
    )

    return fsdp_model


def torch_fsdp_init_decoder_freeze(model,
                                   device,
                                   fp16: bool = True,
                                   fp16_bfloat16: bool = False,
                                   cpu_offload=False):
    my_auto_wrap_policy = partial(transformer_auto_wrap_policy,
                                  transformer_layer_cls={BartEncoderLayer})

    ignored_modules = [model.model.decoder]
    # for dec_layer in model.model.decoder.layers:
    #     ignored_modules.append(dec_layer.encoder_attn)
    #     ignored_modules.append(dec_layer.encoder_attn_layer_norm)

    if fp16:
        fp16_type = torch.float16 if not fp16_bfloat16 else torch.bfloat16
        mixed_precision_cfg = MixedPrecision(param_dtype=fp16_type, reduce_dtype=fp16_type, buffer_dtype=fp16_type)
    else:
        mixed_precision_cfg = None

    fsdp_model = FullyShardedDataParallel(
        model,
        mixed_precision=mixed_precision_cfg,
        auto_wrap_policy=my_auto_wrap_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        cpu_offload=CPUOffload(offload_params=cpu_offload),
        ignored_modules=ignored_modules,
        device_id=device,
    )

    return fsdp_model


def torch_fsdp_init_quantizer_ignore(model,
                                     device,
                                     fp16: bool = True,
                                     fp16_bfloat16: bool = False,
                                     cpu_offload=False):
    my_auto_wrap_policy = partial(transformer_auto_wrap_policy,
                                  transformer_layer_cls={BartEncoderLayer})

    if hasattr(model, "get_non_sharded_modules"):
        ignored_modules = model.get_non_sharded_modules()
    else:
        ignored_modules = [model.quantizer, model.dense1, model.dense2]
    logger.info(f"FSDP ignored modules: {ignored_modules}")

    if fp16:
        fp16_type = torch.float16 if not fp16_bfloat16 else torch.bfloat16
        mixed_precision_cfg = MixedPrecision(param_dtype=fp16_type, reduce_dtype=fp16_type, buffer_dtype=fp16_type)
    else:
        mixed_precision_cfg = None

    fsdp_model = FullyShardedDataParallel(
        model,
        mixed_precision=mixed_precision_cfg,
        auto_wrap_policy=my_auto_wrap_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        cpu_offload=CPUOffload(offload_params=cpu_offload),
        ignored_modules=ignored_modules,
        device_id=device,
    )

    return fsdp_model


def torch_fsdp_size_auto_wrap(model,
                              device,
                              fp16: bool = True,
                              fp16_bfloat16: bool = False,
                              cpu_offload: bool = False,
                              min_num_params: int = 1e8):
    my_auto_wrap_policy = partial(size_based_auto_wrap_policy,
                                  min_num_params=min_num_params)

    ignored_modules = []
    for dec_layer in model.model.decoder.layers:
        ignored_modules.append(dec_layer.encoder_attn)
        ignored_modules.append(dec_layer.encoder_attn_layer_norm)

    if fp16:
        fp16_type = torch.float16 if not fp16_bfloat16 else torch.bfloat16
        mixed_precision_cfg = MixedPrecision(reduce_dtype=fp16_type)
    else:
        mixed_precision_cfg = None

    fsdp_model = FullyShardedDataParallel(
        model,
        mixed_precision=mixed_precision_cfg,
        auto_wrap_policy=my_auto_wrap_policy,
        cpu_offload=CPUOffload(offload_params=cpu_offload),
        ignored_modules=ignored_modules,
        device_id=device,
    )

    return fsdp_model
