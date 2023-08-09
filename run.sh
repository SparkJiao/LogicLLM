MASTER_HOST=$SM_MASTER
MASTER_ADDR=$SM_MASTER_ADDR
MASTER_PORT="23456"
NNODES="$NODE_NUMBER"
NODE_RANK="$NODE_INDEX"

MODEL_S3_BUCKET=s3://sagemaker-us-east-1-107457652907/pretrained-models/llama-7b

wandb login d9bc4cccef46949e9fdffb3df442996d803d43d2

chmod +x ./s5cmd

# git clone https://github.com/HazyResearch/flash-attention.git

# cd flash-attention
# python setup.py install
# cd ../

# ======================================

#./s5cmd sync $MODEL_S3_BUCKET/* /tmp/llama-7b/

#OUTPUT_DIR=/tmp/llama.7b.zh_instruct.10M.v1.0.seq1024.w8.adamw.NA100.0421.ds
#AWS_OUTPUT_BUCKET=s3://sagemaker-us-east-1-107457652907/experiments/llama.7b.zh_instruct.10M.v1.0.seq1024.w8.adamw.NA100.0421.ds

#./s5cmd sync $AWS_OUTPUT_BUCKET/checkpoint-1750/* /tmp/checkpoints/checkpoint-1750/

#python -m torch.distributed.run --nproc_per_node 8 --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT trainer_base_ds_mul_aws.py output_dir=$OUTPUT_DIR aws_output_bucket=$AWS_OUTPUT_BUCKET resume=/tmp/checkpoints/checkpoint-1750 -cp conf/llama/zh/ -cn llama_7b_zh_instruct_v1_0_ds



# ============ COIG SFT ============

# ./s5cmd sync s3://sagemaker-us-east-1-107457652907/experiments/llama.7b.zh_instruct.10M.v1.0.seq1024.w8.adamw.NA100.0421.ds/checkpoint-1750/* /tmp/zh_instruct_v1_0/checkpoint-1750/


# AWS_OUTPUT_BUCKET=s3://sagemaker-us-east-1-107457652907/experiments/llama.7b.zh_instruct.10M.coig.sft.v1.0.seq2048.w8.adamw.NA100.0428.ds


# python -m torch.distributed.run --nproc_per_node 8 --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT trainer_base_ds_mul_aws.py aws_output_bucket=$AWS_OUTPUT_BUCKET -cp conf/llama/zh/ -cn llama_7b_zh_instruct_coig_sft_v1_0_ds


# ============ 13B pre-train ==================

#MODEL_S3_BUCKET=s3://sagemaker-us-east-1-107457652907/pretrained-models/llama-13b
#AWS_OUTPUT_BUCKET=s3://sagemaker-us-east-1-107457652907/experiments/llama.13b.zh_instruct.10M.v1.0.seq1024.w8.adamw.NA100.0430.ds
#
#./s5cmd sync $MODEL_S3_BUCKET/* /tmp/llama-13b/
#
#python -m torch.distributed.run --nproc_per_node 8 --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT trainer_base_ds_mul_aws.py aws_output_bucket=$AWS_OUTPUT_BUCKET -cp conf/llama/zh/ -cn llama_13b_zh_instruct_v1_0_ds
#
#./s5cmd sync /tmp/log_dir/* s3://sagemaker-us-east-1-107457652907/experiments/log_dir/


# =============== 7B OpenLLaMA pre-train ===============================

# MODEL_S3_BUCKET=s3://sagemaker-us-east-1-107457652907/pretrained-models/llama-13b
# AWS_OUTPUT_BUCKET=s3://sagemaker-us-east-1-107457652907/experiments/llama.13b.merit_v91_v91.seq2seq.v5.0.3aug.gpt4all.union.w8.adamw.500steps.NA100.0516

# ./s5cmd sync $MODEL_S3_BUCKET/* /tmp/llama-13b

# python -m torch.distributed.run --nproc_per_node 8 --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT trainer_base_ds_mul_aws.py aws_output_bucket=$AWS_OUTPUT_BUCKET -cp conf/llama/wiki/ -cn llama_13b_merit_v1_pv91_v91_v5_0_gpt4all_union_v1_0

# ./s5cmd sync /tmp/log_dir/* s3://sagemaker-us-east-1-107457652907/experiments/log_dir/

# # ================== llama-65B merit pre-train =====================

# MODEL_S3_BUCKET=s3://sagemaker-us-east-1-107457652907/pretrained-models/llama-65b
# AWS_OUTPUT_BUCKET=s3://sagemaker-us-east-1-107457652907/experiments/llama.65b.q_lora.merit_v91_v91.seq2seq.v5.14.3aug.w16.adamw.500steps.NA100.0608.pad_fix.aws

# # ./s5cmd sync $AWS_OUTPUT_BUCKET/checkpoint-60/* /tmp/checkpoint-60
# ./s5cmd sync $MODEL_S3_BUCKET/* /tmp/llama-65b

# PAD_TOKEN="<unk>" python -m torch.distributed.run --nproc_per_node 8 --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT trainer_base_ds_mul_aws.py aws_output_bucket=$AWS_OUTPUT_BUCKET -cp conf/llama/wiki/ -cn llama_65b_merit_v1_pv91_v91_v5_14_aws

# ./s5cmd sync /tmp/log_dir/* s3://sagemaker-us-east-1-107457652907/experiments/log_dir/

# MODEL_S3_BUCKET=s3://sagemaker-us-east-1-107457652907/pretrained-models/llama-65b
# AWS_OUTPUT_BUCKET=s3://sagemaker-us-east-1-107457652907/experiments/llama.65b.merit_v91_v91.seq2seq.v5.0.3aug.mp2.adamw.500steps.NA100.0616.aws

# ./s5cmd sync $AWS_OUTPUT_BUCKET/checkpoint-60/* /tmp/checkpoint-60
# ./s5cmd sync $MODEL_S3_BUCKET/* /tmp/llama-65b

# PAD_TOKEN="<unk>" python -m torch.distributed.run --nproc_per_node 8 --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT trainer_base_ds_mul_aws.py aws_output_bucket=$AWS_OUTPUT_BUCKET -cp conf/llama/wiki/ -cn llama_65b_merit_v1_pv91_v91_v5_0_full_mp
# PAD_TOKEN="<unk>" python -m torch.distributed.run --nproc_per_node 1 --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT trainer_base_v3_aws.py aws_output_bucket=$AWS_OUTPUT_BUCKET -cp conf/llama/wiki/ -cn llama_65b_merit_v1_pv91_v91_v5_0_full_mp


# MODEL_S3_BUCKET=s3://sagemaker-us-east-1-107457652907/pretrained-models/llama-65b
# AWS_OUTPUT_BUCKET=s3://sagemaker-us-east-1-107457652907/experiments/llama.65b.q_lora.merit_v91_v91.seq2seq.v5.15.3aug.w16.adamw.500steps.NA100.0617.pad_fix.aws

# ./s5cmd sync $MODEL_S3_BUCKET/* /tmp/llama-65b

# PAD_TOKEN="<unk>" python -m torch.distributed.run --nproc_per_node 8 --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT trainer_base_ds_mul_aws.py aws_output_bucket=$AWS_OUTPUT_BUCKET -cp conf/llama/wiki/ -cn llama_65b_merit_v1_pv91_v91_v5_15_aws

# MODEL_S3_BUCKET=s3://sagemaker-us-east-1-107457652907/pretrained-models/llama-65b
# AWS_OUTPUT_BUCKET=s3://sagemaker-us-east-1-107457652907/experiments/llama.65b.q_lora.merit_v91_v91.seq2seq.v5.18.3aug.w16.adamw.500steps.NA100.0617.aws

# ./s5cmd sync $MODEL_S3_BUCKET/* /tmp/llama-65b

# PAD_TOKEN="<unk>" python -m torch.distributed.run --nproc_per_node 8 --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT trainer_base_ds_mul_aws.py aws_output_bucket=$AWS_OUTPUT_BUCKET -cp conf/llama/wiki/ -cn llama_65b_merit_v1_pv91_v91_v5_18_aws


# ./s5cmd sync /tmp/log_dir/* s3://sagemaker-us-east-1-107457652907/experiments/log_dir/

# ================== llama-65b mp8 dp2 merit pre-train =====================

# MODEL_S3_BUCKET=s3://sagemaker-us-east-1-107457652907/pretrained-models/llama-65b-mp8
# # MODEL_S3_BUCKET=s3://sagemaker-us-east-1-107457652907/pretrained-models/llama-65b-mp16
# AWS_OUTPUT_BUCKET=s3://sagemaker-us-east-1-107457652907/experiments/llama.65b.merit_v91_v91.seq2seq.v5.4.3aug.mp8.dp2.adamw.500steps.NA100.0630.aws

# ./s5cmd sync $MODEL_S3_BUCKET/* /tmp/llama-65b-mp8

# # PAD_TOKEN="<unk>" python -m torch.distributed.run --nproc_per_node 8 --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT trainer_base_ds_mp_aws.py aws_output_bucket=$AWS_OUTPUT_BUCKET -cp conf/llama/mp/ -cn llama_65b_merit_v1_pv91_v91_v5_0_full_aws

# # PAD_TOKEN="<unk>" python -m torch.distributed.run --nproc_per_node 8 --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT trainer_base_ds_mp_aws.py aws_output_bucket=$AWS_OUTPUT_BUCKET -cp conf/llama/mp/ -cn llama_65b_merit_v1_pv91_v91_v5_2_full_aws

# # PAD_TOKEN="<unk>" python -m torch.distributed.run --nproc_per_node 8 --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT trainer_base_ds_mp_aws.py aws_output_bucket=$AWS_OUTPUT_BUCKET -cp conf/llama/mp/ -cn llama_65b_merit_v1_pv91_v91_v5_3_full_aws

# PAD_TOKEN="<unk>" python -m torch.distributed.run --nproc_per_node 8 --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT trainer_base_ds_mp_aws.py aws_output_bucket=$AWS_OUTPUT_BUCKET -cp conf/llama/mp/ -cn llama_65b_merit_v1_pv91_v91_v5_4_full_aws

# # PAD_TOKEN="<unk>" deepspeed --num_nodes 2  --master_addr $MASTER_ADDR --master_port $MASTER_PORT trainer_base_ds_mp_aws.py aws_output_bucket=$AWS_OUTPUT_BUCKET -cp conf/llama/mp/ -cn llama_65b_merit_v1_pv91_v91_v5_0_full_aws

# # PAD_TOKEN="<unk>" python -m torch.distributed.run --nproc_per_node 8 --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT trainer_ds_mp_unify_aws.py aws_output_bucket=$AWS_OUTPUT_BUCKET -cp conf/llama/mp/ -cn llama_65b_merit_v1_pv91_v91_v5_0_full_aws


# ./s5cmd sync /tmp/log_dir/* s3://sagemaker-us-east-1-107457652907/experiments/log_dir/

# ================== falcon-40b merit pre-train =====================

# MODEL_S3_BUCKET=s3://sagemaker-us-east-1-107457652907/pretrained-models/falcon-40b
# AWS_OUTPUT_BUCKET=s3://sagemaker-us-east-1-107457652907/experiments/falcon.40b.q_lora.merit_v91_v91.seq2seq.v5.0.3aug.w16.adamw.500steps.NA100.0528.aws

# ./s5cmd sync $MODEL_S3_BUCKET/* /tmp/falcon-40b

# python -m torch.distributed.run --nproc_per_node 8 --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT trainer_base_ds_mul_aws.py aws_output_bucket=$AWS_OUTPUT_BUCKET -cp conf/rw/ -cn falcon_40b_merit_v1_pv91_v91_v5_0_aws

# ./s5cmd sync /tmp/log_dir/* s3://sagemaker-us-east-1-107457652907/experiments/log_dir/


# ========================= Llama2-70b-CoT

MODEL_S3_BUCKET=s3://sagemaker-us-east-1-107457652907/pretrained-models/Llama-2-70b-mp
AWS_OUTPUT_BUCKET=s3://sagemaker-us-east-1-107457652907/experiments/llama2.70b.act.cot.pp8.dp2.A100.v1.0.0808

./s5cmd sync $MODEL_S3_BUCKET/* /tmp/Llama-2-70b-mp
./s5cmd sync s3://sagemaker-us-east-1-107457652907/fangkai/cot-data/* /tmp/data-train

python -m torch.distributed.run --nproc_per_node 8 --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT trainer_base_ds_mp_aws.py aws_output_bucket=$AWS_OUTPUT_BUCKET -cp conf/llama2/cot_actor -cn llama2_70b_cot_tk_rank_v1_0_aws

./s5cmd sync /tmp/log_dir/* s3://sagemaker-us-east-1-107457652907/experiments/log_dir/
