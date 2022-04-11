#!/bin/bash
#SBATCH -n 4
#SBATCH -p JX-GPU-IB
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --mem 240G
#SBATCH --gpus-per-task 1

srun python -u trainer_slurm_fsdp_v1.py -cp conf/roberta/wiki -cn merit_v1_pv9_v9_v3_3090
#python -m torch.distributed.launch --nproc_per_node 2 --nnodes 2  trainer_base_fsdp_v1.py -cp conf/roberta/wiki -cn merit_v1_pv9_v9_v3_3090