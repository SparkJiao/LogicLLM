

for seed in 42 43 44 45 4321; do
  python -m torch.distributed.launch --nproc_per_node 2 trainer_base_fsdp_v1.py seed=$seed -cp conf/deberta-v2/reclor -cn merit_v1_pv9_v9_v5_A100_512_ent_rep_test_xlarge
done;
