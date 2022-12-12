conf=$1
conf_name=$2
num_rank=$3
gpu=$4


count=1
for arg in "$@"; do
  if [ "$count" -gt "4" ]; then
    if [ "$num_rank" -gt "1" ]; then
      echo "CUDA_VISIBLE_DEVICES=$gpu python -m torch.distributed.run --nproc_per_node $num_rank --master_port 10082 trainer_base_fsdp_v3.py -cp $conf -cn $conf_name seed=${arg}"

      CUDA_VISIBLE_DEVICES=$gpu python -m torch.distributed.run --nproc_per_node $num_rank --master_port 10081 trainer_base_fsdp_v3.py -cp $conf -cn $conf_name seed=${arg}
    else
      echo "CUDA_VISIBLE_DEVICES=$gpu python trainer_base_fsdp_v3.py -cp $conf -cn $conf_name seed=${arg}"

      CUDA_VISIBLE_DEVICES=$gpu python trainer_base_fsdp_v3.py -cp $conf -cn $conf_name seed=${arg}
    fi
  fi
  let count=count+1
done;

