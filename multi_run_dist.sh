conf=$1
conf_name=$2
num_rank=$3
gpu=$4

#for seed in 42 43 44 45 46; do
#  CUDA_VISIBLE_DEVICES=$gpu python trainer_base_fsdp_v3.py -cp $conf -cn $conf_name seed=$seed
#done;

for seed in 42 43 44 45 46; do
  CUDA_VISIBLE_DEVICES=$gpu torchrun --nproc_per_node $num_rank --master_port 10080 trainer_base_fsdp_v3.py -cp $conf -cn $conf_name seed=$seed
done;
