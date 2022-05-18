#for seed in 44 45 4321; do
#  python trainer_base_fsdp_v1.py -cp conf/deberta/reclor -cn merit_v1_pv9_v9_v5_A100_512_ent_rep_test_v1_0
#done;



for seed in 42 43 44 45 4321; do
  python trainer_base_fsdp_v1.py -cp conf/deberta/reclor -cn merit_v1_pv9_v9_v5_A100_512_ent_rep_test_v1_2
done;
