# coding=utf-8
#
# Copyright 2023 Nanyang Technological University Fangkai Jiao
#
# Part of this code is based on the source code of Transformers
# (arXiv:1910.03771)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import glob
import logging
import os
import io
from pathlib import Path
import sys

import deepspeed
import hydra
import torch
import json
from omegaconf import DictConfig, OmegaConf
from torch import distributed as dist
from transformers import (AutoTokenizer, AutoConfig, AutoModelForCausalLM, PretrainedConfig)

from general_util.evaluator import evaluate_fn as evaluate
from general_util.logger import setting_logger
from general_util.tokenization_utils import expand_special_tokenizer
from general_util.training_utils import set_seed

logger: logging.Logger

torch.backends.cuda.matmul.allow_tf32 = True


def generate_json(checkpoint_path):
    checkpoints_json = os.path.join(checkpoint_path, "checkpoints.json")

    if os.path.exists(os.path.join(checkpoint_path, "ds_inference_config.json")):  # pre-sharded model.
        checkpoints_json = os.path.join(checkpoint_path, "ds_inference_config.json")
    else:
        with io.open(checkpoints_json, "w", encoding="utf-8") as f:
            file_list = [str(entry) for entry in Path(checkpoint_path).rglob("*.[bp][it][n]") if entry.is_file()]
            data = {"type": "ds_model", "checkpoints": file_list, "version": 1.0}
            print(data)
            json.dump(data, f)

    return checkpoints_json


@hydra.main(config_path="conf", config_name="config", version_base="1.2")
def main(cfg: DictConfig):
    if "LOCAL_RANK" in os.environ and os.environ["LOCAL_RANK"] != -1:
        cfg.local_rank = int(os.environ["LOCAL_RANK"])

    if cfg.local_rank == -1 or cfg.no_cuda:
        device = str(torch.device("cuda" if torch.cuda.is_available() and not cfg.no_cuda else "cpu"))
        cfg.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of synchronizing nodes/GPUs
        torch.cuda.set_device(cfg.local_rank)
        device = str(torch.device("cuda", cfg.local_rank))
        deepspeed.init_distributed()
        cfg.n_gpu = 1
        cfg.world_size = dist.get_world_size()
    cfg.device = device
    cfg.ddp_eval = False

    global logger
    logger = setting_logger(cfg.output_dir, local_rank=cfg.local_rank)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   cfg.local_rank, cfg.device, cfg.n_gpu, bool(cfg.local_rank != -1), cfg.fp16)
    logger.warning(f"CPU cores: {os.cpu_count()}")

    # Set seed
    set_seed(cfg)

    if getattr(cfg, "enable_flash_attention", False):
        logger.info("⚡⚡⚡ enable flash attention.")
        from models.patching import replace_llama_attn_with_flash_attn
        replace_llama_attn_with_flash_attn()

    # Test
    results = {}
    if cfg.do_eval:
        checkpoints = [cfg.output_dir]
        if cfg.save_best:
            pass
        elif cfg.prediction_cfg.best_checkpoint and os.path.exists(cfg.prediction_cfg.best_checkpoint):
            checkpoints = [cfg.prediction_cfg.best_checkpoint]
        elif cfg.eval_sub_path:
            checkpoints = list(sorted(list(set(
                os.path.dirname(c) for c in
                glob.glob(cfg.output_dir + f"/{cfg.eval_sub_path}/" + "pytorch_model*.bin", recursive=True)
            ))))
            if not checkpoints:
                checkpoints = list(sorted(list(set(
                    os.path.dirname(c) for c in
                    glob.glob(cfg.output_dir + f"/{cfg.eval_sub_path}/" + "adapter_model.bin", recursive=True)
                ))))
        logger.info(" the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""
            split = "dev"

            if "model_eval" in cfg:
                model_or_config = hydra.utils.call(cfg.model_eval, checkpoint)
            else:
                model_or_config = hydra.utils.call(cfg.model, checkpoint)

            if isinstance(model_or_config, PretrainedConfig):
                with deepspeed.OnDevice(dtype=torch.float16, device="meta"):
                    model = AutoModelForCausalLM.from_config(model_or_config)
                checkpoints_json = generate_json(checkpoint)
                ds_kwargs = dict(base_dir=checkpoint, checkpoint=checkpoints_json)
            else:
                model = model_or_config
                ds_kwargs = {}

            # ds_inference_config = cfg.ds_infer_cfg
            # ds_config = OmegaConf.to_container(ds_inference_config, resolve=True)

            model = deepspeed.init_inference(
                model,
                mp_size=cfg.world_size,
                dtype=hydra.utils.instantiate(cfg.ds_inference_dtype) if "ds_inference_dtype" in cfg else torch.bfloat16,
                replace_with_kernel_inject=getattr(cfg, "replace_with_kernel_inject", True),
                injection_policy=hydra.utils.instantiate(cfg.injection_policy) if "injection_policy" in cfg else None,
                max_tokens=getattr(cfg, "ds_inference_max_tokens", 1024),
                save_mp_checkpoint_path=getattr(cfg, "ds_inference_save_mp_checkpoint_path", None),
                **ds_kwargs
            )

            tokenizer = AutoTokenizer.from_pretrained(checkpoint)
            expand_special_tokenizer(tokenizer)
            cfg.model_name_or_path = checkpoint

            if cfg.test_file:
                prefix = f'test' + (f'-{prefix}' if prefix != "" else "")
                split = "test"

            result = evaluate(cfg, model, tokenizer, prefix=prefix, _split=split)
            result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
            results.update(result)

            del model.module
            del model

    return results


if __name__ == "__main__":
    os.environ["HYDRA_FULL_ERROR"] = "1"

    hydra_formatted_args = []
    # convert the cli params added by torch.distributed.launch into Hydra format
    for arg in sys.argv:
        if arg.startswith("--"):
            hydra_formatted_args.append(arg[len("--"):])
        else:
            hydra_formatted_args.append(arg)
    sys.argv = hydra_formatted_args
    print(sys.argv)
    main()
