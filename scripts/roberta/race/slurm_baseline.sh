#!/bin/bash
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem 40G
#SBATCH --gpus-per-task 1

for seed in 42 43 44 45; do
  srun python -u trainer_slurm_fsdp_v1.py seed=$seed -cp conf/roberta/race -cn baseline_v1_0
done;