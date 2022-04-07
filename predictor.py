# coding=utf-8
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
import sys

import hydra
import torch
from omegaconf import DictConfig
from torch import distributed as dist
from torch.utils.data import (DataLoader, SequentialSampler)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import (AutoTokenizer, PreTrainedTokenizer)

from general_util.logger import setting_logger
from general_util.training_utils import batch_to_device, unwrap_model, set_seed

logger: logging.Logger


def evaluate(cfg, model, tokenizer: PreTrainedTokenizer, prefix="", _split="dev"):
    dataset = load_and_cache_examples(cfg, tokenizer, _split=_split)

    if cfg.local_rank in [-1, 0] and not os.path.exists(os.path.join(cfg.output_dir, prefix)):
        os.makedirs(os.path.join(cfg.output_dir, prefix))

    cfg.eval_batch_size = cfg.per_gpu_eval_batch_size
    if _split == 'dev' and cfg.ddp_eval:
        eval_sampler = DistributedSampler(dataset, shuffle=False)
    else:
        eval_sampler = SequentialSampler(dataset)  # Note that DistributedSampler samples randomly

    eval_collator = hydra.utils.instantiate(cfg.collator) if "collator" in cfg and cfg.collator else None

    eval_dataloader = DataLoader(dataset,
                                 sampler=eval_sampler,
                                 batch_size=cfg.eval_batch_size,
                                 collate_fn=eval_collator)

    single_model_gpu = unwrap_model(model)
    single_model_gpu.get_eval_log(reset=True)
    # Eval!
    torch.cuda.empty_cache()
    logger.info("***** Running evaluation {}.{} *****".format(_split, prefix))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", cfg.eval_batch_size)
    # Seems FSDP does not need to unwrap the model for evaluating.
    model.eval()
    input_ids_list = []
    logits_list = []
    pred_list = []
    for batch in tqdm(eval_dataloader, desc="Evaluating", disable=cfg.local_rank not in [-1, 0], dynamic_ncols=True):
        batch = batch_to_device(batch, cfg.device)
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                outputs = model(**batch)
                logits = outputs["logits"].detach()
                probs = outputs["logits"].softmax(dim=-1).detach()
                _, pred = probs.max(dim=-1)

                input_ids = batch["input_ids"]

                if cfg.local_rank != -1:
                    _all_logits = [torch.zeros(logits.size()).to(logits.device) for _ in range(dist.get_world_size())]
                    _all_preds = [torch.zeros(pred.size(), dtype=torch.long).to(probs.device) for _ in range(dist.get_world_size())]

                    dist.all_gather(_all_logits, logits)
                    dist.all_gather(_all_preds, pred)

                    logits_list.extend([x.cpu() for x in _all_logits])
                    pred_list.extend([x.cpu() for x in _all_preds])

                    _all_input_ids = [
                        torch.zeros(input_ids.size(), dtype=torch.long).to(input_ids.device) for _ in range(dist.get_world_size())]
                    dist.all_gather(_all_input_ids, input_ids)
                    input_ids_list.extend([x.cpu() for x in _all_input_ids])
                else:
                    logits_list.append(logits.cpu())
                    pred_list.append(pred.cpu())
                    input_ids_list.append(input_ids.cpu())

    metric_log, results = single_model_gpu.get_eval_log(reset=True, ddp=(_split == 'dev' and cfg.ddp_eval), device=cfg.device)
    logger.info("****** Evaluation Results ******")
    logger.info(f"Global Steps: {prefix}")
    logger.info(metric_log)

    if cfg.local_rank in [-1, 0]:
        torch.save({
            "logits": torch.cat(logits_list, dim=0),
            "preds": torch.cat(pred_list, dim=0),
            "input_ids": torch.cat(input_ids_list, dim=0)
        }, os.path.join(cfg.output_dir, prefix, f"{prefix}_prediction.pth"))

    return results


def load_and_cache_examples(cfg, tokenizer: PreTrainedTokenizer, _split="train"):
    if cfg.local_rank not in [-1, 0] and _split == "train":
        dist.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    if _split == "train":
        input_file = cfg.train_file
    elif _split == "dev":
        input_file = cfg.dev_file
    elif _split == "test":
        input_file = cfg.test_file
    else:
        raise RuntimeError(_split)

    dataset = hydra.utils.call(cfg.read_tensor, file_path=input_file, tokenizer=tokenizer)

    if cfg.local_rank == 0 and _split == "train":
        dist.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    return dataset


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

    # Load pre-trained model and tokenizer
    if cfg.local_rank not in [-1, 0]:
        dist.barrier()  # Make sure only the first process in distributed training will download model & vocab

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path)
    model = hydra.utils.call(cfg.model, cfg.model_name_or_path)

    if cfg.local_rank == 0:
        dist.barrier()  # Make sure only the first process in distributed training will download model & vocab

    # if cfg.local_rank == -1:  # For FullyShardedDDP, place the model on cpu first.
    #     model.to(cfg.device)

    # Test
    results = {}

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
        prefix = checkpoint.split("/")[-1]
        split = "dev"

        model = hydra.utils.call(cfg.model, checkpoint)
        model.to(device)

        if cfg.test_file:
            prefix = f'test' + (f'-{prefix}' if prefix != "" else "")
            split = "test"

        prefix = f'predict-{prefix}'

        result = evaluate(cfg, model, tokenizer, prefix=prefix, _split=split)
        result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
        results.update(result)

    return results


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
