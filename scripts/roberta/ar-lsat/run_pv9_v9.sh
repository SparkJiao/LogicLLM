

#python trainer_base_fsdp_v1.py -cp conf/roberta/ar-lsat -cn merit_v1_pv9_v9_v4_0
python trainer_base_fsdp_v1.py -cp conf/roberta/ar-lsat -cn merit_v1_pv9_v9_v4_1

#for seed in 42 43 44 45 4321; do
#  python -m torch.distributed.launch --nproc_per_node 2 --master_port 10000 trainer_base_fsdp_v1.py \
#    seed=$seed -cp conf/roberta/reclor -cn merit_v1_pv9_v9_v5_1
#done;
