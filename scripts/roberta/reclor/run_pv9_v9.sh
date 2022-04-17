

#for seed in 42 43 44 45 4321; do
#  python -m torch.distributed.launch --nproc_per_node 4 --master_port 10000 trainer_base_fsdp_v1.py \
#    seed=$seed -cp conf/roberta/reclor -cn merit_v1_pv9_v9_v1_1
#done;


##for seed in 42 43 44 45 4321; do
#for seed in 46; do
#  python -m torch.distributed.launch --nproc_per_node 2 --master_port 10000 trainer_base_fsdp_v1.py \
#    seed=$seed -cp conf/roberta/reclor -cn merit_v1_pv9_v9_v2_0
#done;


#for seed in 42 43 44 45 4321; do
#for seed in 4321 46; do
#  python -m torch.distributed.launch --nproc_per_node 2 --master_port 10000 trainer_base_fsdp_v1.py \
#    seed=$seed -cp conf/roberta/reclor -cn merit_v1_pv9_v9_v3_0
#done;

#for seed in 42 43 44 45 4321; do
#  python -m torch.distributed.launch --nproc_per_node 2 --master_port 10000 trainer_base_fsdp_v1.py \
#    seed=$seed -cp conf/roberta/reclor -cn merit_v1_pv9_v9_v3_1
#done;


#for seed in 42 43 44 45 4321; do
#  python -m torch.distributed.launch --nproc_per_node 2 --master_port 10000 trainer_base_fsdp_v1.py \
#    seed=$seed -cp conf/roberta/reclor -cn merit_v1_pv9_v9_v4_0
#done;


#for seed in 42 43 44 45 4321; do
#  python -m torch.distributed.launch --nproc_per_node 2 --master_port 10000 trainer_base_fsdp_v1.py \
#    seed=$seed -cp conf/roberta/reclor -cn merit_v1_pv9_v9_v4_1
#done;


#for seed in 42 43 44 45 4321; do
#  python -m torch.distributed.launch --nproc_per_node 2 --master_port 10000 trainer_base_fsdp_v1.py \
#    seed=$seed -cp conf/roberta/reclor -cn merit_v1_pv9_v9_v5_0
#done;


#for seed in 42 43 44 45 4321; do
#  python -m torch.distributed.launch --nproc_per_node 2 --master_port 10000 trainer_base_fsdp_v1.py \
#    seed=$seed -cp conf/roberta/reclor -cn merit_v1_pv9_v9_v5_1
#done;


#for seed in 43 44 45 4321; do
#  python -m torch.distributed.launch --nproc_per_node 2 --master_port 10000 trainer_base_fsdp_v1.py \
#    seed=$seed -cp conf/roberta/reclor -cn merit_v1_pv9_v9_v6_0
#done;
#
#for seed in 42 43 44 45 4321; do
#  python -m torch.distributed.launch --nproc_per_node 2 --master_port 10000 trainer_base_fsdp_v1.py \
#    seed=$seed -cp conf/roberta/reclor -cn merit_v1_pv9_v9_v6_1
#done;

#seed=46
#python -m torch.distributed.launch --nproc_per_node 2 --master_port 10000 trainer_base_fsdp_v1.py \
#    seed=$seed -cp conf/roberta/reclor -cn merit_v1_pv9_v9_v6_0


#for seed in 42 43 44 45 4321; do
#  python -m torch.distributed.launch --nproc_per_node 4 --master_port 10000 trainer_base_fsdp_v1.py \
#    seed=$seed -cp conf/roberta/reclor -cn merit_v1_pv9_v9_v7_0
#done;
#
#for seed in 42 43 44 45 4321; do
#  python -m torch.distributed.launch --nproc_per_node 4 --master_port 10000 trainer_base_fsdp_v1.py \
#    seed=$seed -cp conf/roberta/reclor -cn merit_v1_pv9_v9_v7_1
#done;

for seed in 42 43 44 45 4321; do
  python -m torch.distributed.launch --nproc_per_node 4 --master_port 10000 trainer_base_fsdp_v1.py \
    seed=$seed -cp conf/roberta/reclor -cn merit_v1_pv9_v9_v7_0_titanxp
done;
