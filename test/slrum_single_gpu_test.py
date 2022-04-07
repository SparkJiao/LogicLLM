import logging

import torch
import sys
import hydra
from omegaconf import DictConfig
import torch.distributed as dist
import os
import subprocess


def setup_slurm_distributed(cfg: DictConfig, backend="nccl", port=None):
    """
    Most code are copied from https://github.com/BIGBALLON/distribuuuu/blob/master/tutorial/mnmc_ddp_slurm.py.
    """
    num_gpus = torch.cuda.device_count()
    if num_gpus <= 1:
        cfg.local_rank = -1
        cfg.device = str(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        cfg.n_gpu = num_gpus
        return

    proc_id = int(os.environ["SLURM_PROCID"])
    n_tasks = int(os.environ["SLURM_NTASKS"])
    node_list = os.environ["SLURM_NODELIST"]

    torch.cuda.set_device(proc_id % num_gpus)

    addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")
    # specify master port
    if port is not None:
        os.environ["MASTER_PORT"] = str(port)
    elif "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "29500"
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = addr

    os.environ["WORLD_SIZE"] = str(n_tasks)
    os.environ["LOCAL_RANK"] = str(proc_id % num_gpus)
    os.environ["RANK"] = str(proc_id)

    cfg.local_rank = int(os.environ["LOCAL_RANK"])
    cfg.world_size = int(os.environ["WORLD_SIZE"])
    cfg.device = str(torch.device("cuda", cfg.local_rank))

    dist.init_process_group(backend=backend)


@hydra.main(config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    logger = logging.Logger("test")

    print(f"============ Before DDP initialization ===============")
    print(f"local_rank = {cfg.local_rank}")

    setup_slurm_distributed(cfg)

    print(f"============ After DDP initialization ===============")
    print(f"local_rank = {cfg.local_rank}")
    print(f"world_size = {cfg.world_size}")
    print(f"device = {cfg.device}")
    print(f"torch dist information: local_rank == {dist.get_rank()}, world_size == {dist.get_world_size()}")


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
