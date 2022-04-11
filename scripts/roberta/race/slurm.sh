#!/bin/bash
#SBATCH -n 2
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem 80G
#SBATCH --gpus-per-task 1

for seed in 42 43 44 45; do
  srun python -u trainer_slurm_fsdp_v1.py seed=$seed -cp conf/roberta/race -cn merit_v1_pv9_v9_v4_0
done;