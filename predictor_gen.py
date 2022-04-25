# coding=utf-8
#
# Copyright 2020 Heinrich Heine University Duesseldorf
#
# Part of this code is based on the source code of BERT-DST
# (arXiv:1907.03040)
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
import sys
from typing import Union

import hydra
import torch
from omegaconf import DictConfig
from torch import distributed as dist
from torch.utils.data import (DataLoader, SequentialSampler)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import (AutoTokenizer, PreTrainedTokenizer, PreTrainedModel, GPT2Tokenizer, GPT2TokenizerFast)
from transformers.generation_utils import GenerationMixin

from general_util.logger import setting_logger
from general_util.training_utils import batch_to_device, set_seed, load_and_cache_examples

torch.backends.cuda.matmul.allow_tf32 = True

logger: logging.Logger


def evaluate(cfg, model: Union[GenerationMixin, PreTrainedModel], tokenizer: PreTrainedTokenizer, prefix="", _split="dev"):
    dataset = load_and_cache_examples(cfg, tokenizer, _split=_split)

    output_dir = os.path.join(cfg.prediction_dir, prefix) if getattr(cfg, "prediction_dir", False) else os.path.join(cfg.output_dir, prefix)
    if cfg.local_rank in [-1, 0] and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cfg.eval_batch_size = cfg.per_gpu_eval_batch_size
    if _split == 'dev' and cfg.ddp_eval and cfg.local_rank != -1:
        eval_sampler = DistributedSampler(dataset, shuffle=False)
    else:
        eval_sampler = SequentialSampler(dataset)  # Note that DistributedSampler samples randomly
    eval_collator = hydra.utils.instantiate(cfg.collator) if "collator" in cfg and cfg.collator else None
    eval_dataloader = DataLoader(dataset,
                                 sampler=eval_sampler,
                                 batch_size=cfg.eval_batch_size,
                                 collate_fn=eval_collator)

    # Eval!
    torch.cuda.empty_cache()
    logger.info("***** Running evaluation {}.{} *****".format(_split, prefix))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", cfg.eval_batch_size)
    # Seems FSDP does not need to unwrap the model for evaluating.
    model.eval()
    output_list = []
    index_list = []
    for batch in tqdm(eval_dataloader, desc="Evaluating", disable=cfg.local_rank not in [-1, 0], dynamic_ncols=True):
        if "index" in batch:
            feat_index = batch.pop("index")
            index_list.extend(feat_index.tolist())
        batch = batch_to_device(batch, cfg.device)
        if cfg.fp16:
            with torch.cuda.amp.autocast(dtype=(torch.bfloat16 if getattr(cfg, "fp16_bfloat16", False) else torch.float16)):
                with torch.no_grad():
                    outputs = model.generate(**batch, max_length=cfg.max_output_length,
                                             num_beams=cfg.num_beams, num_return_sequences=cfg.num_return_sequences)
        else:
            with torch.no_grad():
                outputs = model.generate(**batch, max_length=cfg.max_output_length,
                                         num_beams=cfg.num_beams, num_return_sequences=cfg.num_return_sequences)
        if getattr(cfg, "num_return_sequences", 1) > 1:
            outputs = [tokenizer.batch_decode(b_output, skip_special_tokens=True) for b_output in outputs]
            if isinstance(tokenizer, GPT2Tokenizer) or isinstance(tokenizer, GPT2TokenizerFast):
                outputs = [list(map(lambda x: x.split("Deduction:")[-1], output)) for output in outputs]
        else:
            outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            if isinstance(tokenizer, GPT2Tokenizer) or isinstance(tokenizer, GPT2TokenizerFast):
                outputs = list(map(lambda x: x.split("Deduction:")[-1], outputs))

        output_list.extend(outputs)

    if cfg.local_rank == -1:
        torch.save((index_list, output_list), os.path.join(output_dir, "generate-predictions.pt"))
    else:
        torch.save((index_list, output_list), os.path.join(output_dir, "generate-predictions-{}.pt".format(cfg.local_rank)))


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    if cfg.local_rank == -1 or cfg.no_cuda:
        device = str(torch.device("cuda" if torch.cuda.is_available() and not cfg.no_cuda else "cpu"))
        cfg.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of synchronizing nodes/GPUs
        torch.cuda.set_device(cfg.local_rank)
        device = str(torch.device("cuda", cfg.local_rank))
        dist.init_process_group(backend='nccl')
        cfg.n_gpu = 1
        cfg.world_size = dist.get_world_size()
    cfg.device = device

    global logger
    logger = setting_logger(cfg.output_dir, local_rank=cfg.local_rank)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   cfg.local_rank, device, cfg.n_gpu, bool(cfg.local_rank != -1), cfg.fp16)

    # Set seed
    set_seed(cfg)

    # Test
    checkpoints = [cfg.output_dir]
    if cfg.save_best:
        logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
    elif cfg.prediction_cfg.best_checkpoint and os.path.exists(cfg.prediction_cfg.best_checkpoint):
        checkpoints = [cfg.prediction_cfg.best_checkpoint]
        logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
    elif cfg.eval_sub_path:
        checkpoints = list(
            os.path.dirname(c) for c in
            sorted(glob.glob(cfg.output_dir + f"/{cfg.eval_sub_path}/" + "pytorch_model.bin", recursive=True))
        )
        logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
    logger.info(" the following checkpoints: %s", checkpoints)
    for checkpoint in checkpoints:
        global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
        prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else "best"
        split = "dev"

        model = hydra.utils.call(cfg.model, checkpoint)
        model.to(device)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)

        if cfg.test_file:
            split = "test"

        prefix = f"prediction.gen.{prefix}.{split}"

        evaluate(cfg, model, tokenizer, prefix=prefix, _split=split)


if __name__ == "__main__":
    hydra_formatted_args = []
    # convert the cli params added by torch.distributed.launch into Hydra format
    for arg in sys.argv:
        if arg.startswith("--"):
            hydra_formatted_args.append(arg[len("--"):])
        else:
            hydra_formatted_args.append(arg)
    sys.argv = hydra_formatted_args

    main()
