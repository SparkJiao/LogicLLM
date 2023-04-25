conf=$1
conf_name=$2
num_rank=$3
gpu=$4
port=$5

count=1
for arg in "$@"; do
  if [ "$count" -gt "5" ]; then
    if [ "$num_rank" -gt "1" ]; then
      echo "deepspeed --include localhost:$gpu trainer_base_ds_v1.py --seed=${arg} -cp $conf -cn $conf_name"

      deepspeed --include localhost:$gpu --master_port $port trainer_base_ds_v1.py --seed=${arg} -cp $conf -cn $conf_name
#      CUDA_VISIBLE_DEVICES=$gpu python -m torch.distributed.run --nproc_per_node $num_rank --master_port $port trainer_base_ds_v1.py -cp $conf -cn $conf_name seed=${arg}
    else
      echo "deepspeed --include localhost:$gpu trainer_base_ds_v1.py --seed=${arg} -cp $conf -cn $conf_name"

      deepspeed --include localhost:$gpu --master_port $port trainer_base_ds_v1.py  --seed=${arg} -cp $conf -cn $conf_name
#      CUDA_VISIBLE_DEVICES=$gpu python trainer_base_ds_v1.py -cp $conf -cn $conf_name seed=${arg}
    fi
  fi
  let count=count+1
done;

