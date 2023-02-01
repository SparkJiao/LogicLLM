#for idx in {0..9}; do
#  torchrun --nproc_per_node 3 trainer_base_fsdp_v3.py -cp conf/roberta/wiki/predict -cn erica_prediction _idx=$idx
#done;

for idx in {0..9}; do
  python -m torch.distributed.run --nproc_per_node 4 trainer_base_fsdp_v3.py -cp conf/roberta/wiki/predict -cn erica_prediction_j2l3_v2 _idx=$idx
done;