

##for seed in 42 43 44 45 4321; do
#for seed in 46; do
#  python -m torch.distributed.launch --nproc_per_node 2 --master_port 10000 trainer_base_fsdp_v1.py \
#    seed=$seed -cp conf/roberta/reclor -cn merit_v1_pv9_v1
#done;

for seed in 42 43 44 45 4321; do
  python -m torch.distributed.launch --nproc_per_node 2 --master_port 10000 trainer_base_fsdp_v1.py \
    seed=$seed -cp conf/roberta/reclor -cn merit_v1_pv9_v2
done;
