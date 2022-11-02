conf=$1
conf_name=$2
gpu=$3

#for seed in 44 45 46; do
for seed in 42 43 44 45 46; do
  CUDA_VISIBLE_DEVICES=$gpu python trainer_base_fsdp_v3.py -cp $conf -cn $conf_name seed=$seed
done;

#for seed in 43 44 45 46; do
#  CUDA_VISIBLE_DEVICES=$gpu torchrun --nproc_per_node 2 trainer_base_fsdp_v3.py -cp $conf -cn $conf_name seed=$seed
#done;
