#!/bin/bash
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem 40G
#SBATCH --gpus-per-task 1

#for seed in 42 43 44 45 4321; do
#  srun python -u trainer_slurm_fsdp_v1.py seed=$seed -cp conf/roberta/reclor -cn merit_v1_pv9_v9_v7_0
#done;
#
#for seed in 42 43 44 45 4321; do
#  srun python -u trainer_slurm_fsdp_v1.py seed=$seed -cp conf/roberta/reclor -cn merit_v1_pv9_v9_v7_1
#done;

#for seed in 42 43 44 45 4321; do
#  srun python -u trainer_slurm_fsdp_v1.py seed=$seed -cp conf/roberta/reclor -cn merit_v1_pv9_v9_v7_2
#done;

#for seed in 42 43 44 45 4321; do
#  srun python -u trainer_slurm_fsdp_v1.py seed=$seed -cp conf/roberta/reclor -cn merit_v1_pv9_v9_v8_0
#done;


for seed in 42 43 44 45 4321; do
  srun python -u trainer_slurm_fsdp_v1.py seed=$seed -cp conf/roberta/reclor -cn merit_v1_pv9_v9_v8_1
done;


