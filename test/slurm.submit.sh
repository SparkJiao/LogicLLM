#!/bin/bash
#SBATCH -n 2
#SBATCH -c 1
#SBATCH -p JX-GPU
#SBATCH --mem 20G
#SBATCH --gpus-per-task 1

#srun python -u slrum_single_gpu_test.py -cn base
srun python trainer_slurm_fsdp_v1.py model_name_or_path=roberta-large -cp conf/roberta/logic-nli -cn roberta_v1