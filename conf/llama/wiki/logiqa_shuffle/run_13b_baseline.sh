CUDA_VISIBLE_DEVICES=0 PAD_TOKEN="<unk>" python trainer_base_fsdp_v4.py -cp conf/llama/wiki/logiqa_shuffle -cn llama_13b_logiqav2_eval_mc_v1_0_s0312_test

CUDA_VISIBLE_DEVICES=0 PAD_TOKEN="<unk>" python trainer_base_fsdp_v4.py -cp conf/llama/wiki/logiqa_shuffle -cn llama_13b_logiqav2_eval_mc_v1_0_s1302_test

CUDA_VISIBLE_DEVICES=0 PAD_TOKEN="<unk>" python trainer_base_fsdp_v4.py -cp conf/llama/wiki/logiqa_shuffle -cn llama_13b_logiqav2_eval_mc_v1_0_s2103_test

CUDA_VISIBLE_DEVICES=0 PAD_TOKEN="<unk>" python trainer_base_fsdp_v4.py -cp conf/llama/wiki/logiqa_shuffle -cn llama_13b_logiqav2_eval_mc_v1_0_s3012_test