for idx in {0..9}; do
  torchrun --nproc_per_node 3 trainer_base_fsdp_v3.py -cp conf/roberta/wiki/predict -cn erica_prediction _idx=$idx
done;