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

import datetime
import glob
import logging
import os
import sys
from typing import Dict, Union

import deepspeed
import hydra
import torch
import wandb
from deepspeed.pipe import PipelineModule
from omegaconf import DictConfig, OmegaConf
from torch import distributed as dist
from torch.utils.data import (DataLoader, RandomSampler, DistributedSampler)
from tqdm import tqdm, trange
from transformers import (AutoTokenizer, PreTrainedTokenizer)

import models.llama_ds_mp_wrap
from general_util.evaluator import evaluate_fn as evaluate
from general_util.logger import setting_logger
from general_util.training_utils import set_seed, load_and_cache_examples, set_seed_int

logger: logging.Logger

torch.backends.cuda.matmul.allow_tf32 = True

GLOBAL_SEED = 1
GLOBAL_WORKER_ID = None


def get_zero_stage(cfg: DictConfig):
    if hasattr(cfg, "zero_optimization"):
        return int(getattr(cfg.zero_optimization, "stage", 0))
    return 0


def worker_init_fn(worker_id):
    global GLOBAL_WORKER_ID
    GLOBAL_WORKER_ID = worker_id
    set_seed_int(GLOBAL_SEED + worker_id)


def load_empty_dataset_and_collator(cfg: DictConfig):
    from data.test import TestDataset
    from data.collators.flan import FlanCollatorOverCollator

    dataset = TestDataset(None, None)
    collator = FlanCollatorOverCollator(collator=None,
                                        tokenizer=cfg.model_name_or_path,
                                        max_seq_length=128,
                                        decoder_only=True,
                                        return_standard_inputs=True,
                                        )

    # Keep consistent with `load_and_cache_examples`.
    if getattr(cfg, "dist_load_data_barrier", True):
        dist.barrier()

    if dist.is_initialized():
        dist.barrier()

    return dataset, collator


def save_model(model: Union[deepspeed.DeepSpeedEngine, deepspeed.PipelineEngine],
               cfg: DictConfig, output_dir: str, tokenizer: PreTrainedTokenizer = None, state_dict: Dict = None):
    model.save_checkpoint(output_dir)

    if cfg.local_rank not in [-1, 0]:
        dist.barrier()

    if cfg.local_rank in [-1, 0]:

        if tokenizer is not None:
            tokenizer.save_pretrained(output_dir)

        OmegaConf.save(cfg, os.path.join(output_dir, "training_config.yaml"))
        logger.info("Saving model checkpoint to %s", output_dir)

        end_dir = output_dir.split("/")[-1]

        os.system(f"./s5cmd sync {output_dir}/ {cfg.aws_output_bucket}/{end_dir}/")

        if cfg.local_rank == 0:
            dist.barrier()


def train(cfg, model, tokenizer, continue_from_global_step=0):
    """ Train the model """
    if cfg.local_rank in [-1, 0]:
        tb_helper = hydra.utils.instantiate(cfg.summary_helper) if "summary_helper" in cfg and cfg.summary_helper else None
    else:
        tb_helper = None

    cfg.train_batch_size = cfg.per_gpu_train_batch_size

    if "_target_" in cfg.train_file:
        files = hydra.utils.instantiate(cfg.train_file)
    elif cfg.train_file.startswith("hf:"):
        files = [cfg.train_file[3:]]
    elif os.path.exists(cfg.train_file):
        files = [cfg.train_file]
    else:
        files = list(glob.glob(cfg.train_file))
    logger.info(files)

    dp_degree = dist.get_world_size() // cfg.num_stages

    if getattr(cfg, "total_dataset_len", -1) > 0:
        total_dataset_len = cfg.total_dataset_len
    else:
        total_dataset_len = 0
        for _file in tqdm(files, total=len(files)):
            sub_train_dataset = load_and_cache_examples(cfg, tokenizer, _split="train", _file=_file)
            # total_dataset_len += (len(sub_train_dataset) // cfg.train_batch_size // dp_degree)
            total_dataset_len += len(sub_train_dataset)
            del sub_train_dataset

    if getattr(cfg, "do_preprocess", False):
        return

    if "extended_vocab" in cfg and cfg.extended_vocab:
        logger.info(f"Extended extra vocab size: {cfg.extended_vocab}")
        model.resize_token_embeddings(model.config.vocab_size + cfg.extended_vocab)

    _actual_train_batch_size = cfg.train_batch_size * cfg.gradient_accumulation_steps * dp_degree
    if cfg.max_steps > 0:
        t_total = cfg.max_steps
        cfg.num_train_epochs = cfg.max_steps // (total_dataset_len // _actual_train_batch_size) + 1
    else:
        t_total = total_dataset_len // _actual_train_batch_size * cfg.num_train_epochs

    num_warmup_steps = int(t_total * cfg.warmup_proportion) if cfg.warmup_proportion else cfg.warmup_steps

    ds_config = cfg.ds_cfg
    if "total_num_steps" in ds_config.scheduler.params:
        ds_config.scheduler.params.total_num_steps = t_total
    ds_config.scheduler.params.warmup_num_steps = num_warmup_steps
    ds_config = OmegaConf.to_container(ds_config, resolve=True)

    if torch.__version__ >= "2" and (getattr(os.environ, "TORCH_COMPILE", False) or getattr(cfg, "compile", False)):
        model = torch.compile(model, mode="max-autotune")
    model, optimizer, _, scheduler = deepspeed.initialize(model=model,
                                                          model_parameters=[p for p in model.parameters() if p.requires_grad],
                                                          config=ds_config)

    model.load_checkpoint(cfg.model_name_or_path, load_module_only=True, load_optimizer_states=False, load_lr_scheduler_states=False)
    logger.info(optimizer.optimizer)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", total_dataset_len)
    logger.info("  Num Epochs = %d", cfg.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", cfg.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d", _actual_train_batch_size)
    logger.info("  Gradient Accumulation steps = %d", cfg.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    logger.info("  Warmup steps = %d", num_warmup_steps)

    if continue_from_global_step > 0:
        logger.info("Fast forwarding to global step %d to resume training from latest checkpoint...", continue_from_global_step)
        model.load_checkpoint(cfg.resume)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    # model.zero_grad()
    train_iterator = trange(int(cfg.num_train_epochs), desc="Epoch", disable=cfg.local_rank not in [-1, 0])
    set_seed(cfg)  # Added here for reproducibility (even between python 2 and 3)

    if cfg.local_rank in [-1, 0]:
        os.system(f"nvidia-smi")
    
    for epoch in train_iterator:
        for _file in files:
            if model.is_first_stage() or model.is_last_stage():
                sub_train_dataset = load_and_cache_examples(cfg, tokenizer, _split="train", _file=_file)

                if dp_degree > 1:
                    dp_id = model.grid.get_data_parallel_id()
                    sub_train_sampler = DistributedSampler(sub_train_dataset, num_replicas=dp_degree, rank=dp_id)
                else:
                    sub_train_sampler = RandomSampler(sub_train_dataset)
                sub_train_collator = hydra.utils.instantiate(cfg.collator) if "collator" in cfg and cfg.collator else None

                sub_train_dataloader = DataLoader(dataset=sub_train_dataset,
                                                  sampler=sub_train_sampler,
                                                  # shuffle=True,
                                                  # shuffle=False,
                                                  batch_size=cfg.train_batch_size,
                                                  collate_fn=sub_train_collator,
                                                  num_workers=cfg.num_workers,
                                                  pin_memory=True,
                                                  prefetch_factor=cfg.prefetch_factor,
                                                  # worker_init_fn=worker_init_fn,
                                                  # generator=g,
                                                  drop_last=True,
                                                  )
            else:
                sub_train_dataset, sub_train_collator = load_empty_dataset_and_collator(cfg)
                sub_train_sampler = None

                sub_train_dataloader = DataLoader(dataset=sub_train_dataset,
                                                  batch_size=cfg.train_batch_size,
                                                  collate_fn=sub_train_collator,
                                                  drop_last=True,
                                                  shuffle=False)

            epoch_update_steps = len(sub_train_dataloader) // cfg.gradient_accumulation_steps
            sub_train_dataloader = iter(deepspeed.utils.RepeatingLoader(sub_train_dataloader))
            
            if cfg.local_rank in [-1, 0]:
                os.system("free -h")

            if sub_train_sampler is not None and isinstance(sub_train_sampler, DistributedSampler):
                sub_train_sampler.set_epoch(epoch)

            for step in tqdm(range(epoch_update_steps), desc="Iteration", disable=cfg.local_rank not in [-1, 0], dynamic_ncols=True):
                # If training is continued from a checkpoint, fast forward
                # to the state of that checkpoint.
                if global_step < continue_from_global_step:
                    if (step + 1) % cfg.gradient_accumulation_steps == 0:
                        # scheduler.step()  # Update learning rate schedule  # Done by `load_checkpoint` of DS.
                        global_step += 1
                    continue

                model.train()
                loss = model.train_batch(data_iter=sub_train_dataloader)
                global_step += 1

                tr_loss += loss.item()
                # print("Outside loss: ", loss)

                # Log metrics
                log_metrics = {}
                if cfg.local_rank in [-1, 0] and cfg.logging_steps > 0 and global_step % cfg.logging_steps == 0:
                    log_metrics['lr'] = scheduler.get_lr()[0]
                    log_metrics['loss'] = (tr_loss - logging_loss) / cfg.logging_steps
                    logging_loss = tr_loss

                # Save model checkpoint
                if cfg.save_steps > 0 and global_step % cfg.save_steps == 0:
                    output_dir = os.path.join(cfg.output_dir, 'checkpoint-{}'.format(global_step))
                    if cfg.local_rank in [-1, 0] and not os.path.exists(output_dir):
                        os.makedirs(output_dir, exist_ok=True)
                    save_model(model, cfg, output_dir, tokenizer)

                if len(log_metrics) > 0 and cfg.local_rank in [-1, 0]:
                    wandb.log(log_metrics)

                del log_metrics

            if 0 < cfg.max_steps < global_step:
                train_iterator.close()
                break

        if 0 < cfg.max_steps < global_step:
            break

    return global_step, tr_loss / global_step


@hydra.main(config_path="conf", config_name="config", version_base="1.2")
def main(cfg: DictConfig):
    if "LOCAL_RANK" in os.environ and os.environ["LOCAL_RANK"] not in [-1, "-1"]:
        cfg.local_rank = int(os.environ["LOCAL_RANK"])

    if cfg.local_rank == -1 or cfg.no_cuda:
        device = str(torch.device("cuda" if torch.cuda.is_available() and not cfg.no_cuda else "cpu"))
        cfg.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of synchronizing nodes/GPUs
        torch.cuda.set_device(cfg.local_rank)
        device = str(torch.device("cuda", cfg.local_rank))
        deepspeed.init_distributed(dist_backend="nccl", timeout=datetime.timedelta(seconds=7200))
        cfg.n_gpu = 1
        cfg.world_size = dist.get_world_size()
    cfg.device = device

    global logger
    logger = setting_logger(cfg.output_dir, local_rank=cfg.local_rank)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   cfg.local_rank, cfg.device, cfg.n_gpu, bool(cfg.local_rank != -1), cfg.fp16)
    logger.warning(f"CPU cores: {os.cpu_count()}")

    # Set seed
    set_seed(cfg)
    # deepspeed.runtime.utils.set_random_seed(cfg.seed)

    use_barrier = not os.path.exists(cfg.model_name_or_path)
    # Load pre-trained model and tokenizer
    if use_barrier and cfg.local_rank not in [-1, 0]:
        dist.barrier()  # Make sure only the first process in distributed training will download model & vocab

    if cfg.pretrain:
        pretrain_state_dict = torch.load(cfg.pretrain, map_location='cpu')
    else:
        pretrain_state_dict = None

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path)

    from general_util.tokenization_utils import expand_special_tokenizer

    expand_special_tokenizer(tokenizer)

    if getattr(cfg, "enable_flash_attention", False):
        logger.info("⚡⚡⚡ enable flash attention.")
        from models.patching import replace_llama_attn_with_flash_attn
        replace_llama_attn_with_flash_attn()

    model_or_config = hydra.utils.call(cfg.model, cfg.model_name_or_path)

    layers = hydra.utils.call(cfg.get_layers, model_or_config)

    from deepspeed.runtime.pipe.topology import PipeModelDataParallelTopology, ProcessTopology

    dp_degree = dist.get_world_size() // cfg.num_stages
    # topo = PipeModelDataParallelTopology(num_pp=cfg.num_stages, num_mp=1, num_dp=dp_degree)
    # topo = ProcessTopology(axes=['data', 'pipe'], dims=[dp_degree, cfg.num_stages])
    # print(f"Rank: {dist.get_rank()}, Topo: {topo.get_coord(dist.get_rank())}")
    model_pipe = PipelineModule(layers=layers,
                                num_stages=cfg.num_stages,
                                # topology=topo,
                                loss_fn=models.llama_ds_mp_wrap.loss_fn,
                                # partition_method="uniform",
                                activation_checkpoint_interval=getattr(cfg, "activation_checkpoint_interval", 0)
                                )
    # logger.warning(f"{dist.get_rank()}: {model_pipe}")

    if use_barrier and cfg.local_rank == 0:
        dist.barrier()  # Make sure only the first process in distributed training will download model & vocab

    # logger.info("Training/evaluation parameters %s", OmegaConf.to_yaml(cfg))
    if cfg.local_rank in [-1, 0] and cfg.do_train:
        if not os.path.exists(cfg.output_dir):
            os.makedirs(cfg.output_dir)
        OmegaConf.save(cfg, os.path.join(cfg.output_dir, "training_config.yaml"))

        wandb.init(
            project="LLaMA-BiFLAN",
            name=f"{cfg.exp_name}-{dist.get_rank()}",
            notes=cfg.exp_notes,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        wandb.define_metric(cfg.prediction_cfg.metric, summary=("max" if cfg.prediction_cfg.measure > 0 else "min"))

    # Training
    if cfg.do_train:
        continue_from_global_step = 0  # If set to 0, start training from the beginning
        if os.path.exists(cfg.output_dir) and getattr(cfg, "resume", None):
            checkpoint = cfg.resume
            logger.info("Resuming training from the latest checkpoint: %s", checkpoint)
            continue_from_global_step = int(checkpoint.split('-')[-1])

        global_step, tr_loss = train(cfg, model_pipe, tokenizer, continue_from_global_step)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Test
    results = {}
    if cfg.do_eval:
        if not cfg.ddp_eval and cfg.local_rank not in [-1, 0]:
            return results

        checkpoints = [cfg.output_dir]
        if cfg.save_best:
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        elif cfg.prediction_cfg.best_checkpoint and os.path.exists(cfg.prediction_cfg.best_checkpoint):
            checkpoints = [cfg.prediction_cfg.best_checkpoint]
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        elif cfg.eval_sub_path:
            checkpoints = list(sorted(list(set(
                os.path.dirname(c) for c in
                glob.glob(cfg.output_dir + f"/{cfg.eval_sub_path}/" + "pytorch_model*.bin", recursive=True)
            ))))
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info(" the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""
            split = "dev"

            if "model_eval" in cfg:
                model = hydra.utils.call(cfg.model_eval, checkpoint)
            else:
                model = hydra.utils.call(cfg.model, checkpoint)
            if cfg.n_gpu == 1:
                model.to(cfg.device)
            else:
                # For model parallel (of mT5)
                if getattr(cfg, "get_device_map", None):
                    model.parallelize(hydra.utils.call(cfg.get_device_map))
                else:
                    model.parallelize()

            tokenizer = AutoTokenizer.from_pretrained(checkpoint)
            cfg.model_name_or_path = checkpoint

            if cfg.test_file:
                prefix = f'test' + (f'-{prefix}' if prefix != "" else "")
                split = "test"

            result = evaluate(cfg, model, tokenizer, prefix=prefix, _split=split)
            result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
            results.update(result)

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
