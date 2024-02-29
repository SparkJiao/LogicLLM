# LogicLLM

This is the official repository for the paper: "Exploring Self-supervised Logic-enhanced Training for Large Language Models".

## Requirements

- Python: 3.9
- CUDA: 11.7/11.8

Other python packages can be installed using the following command:
```bash
pip install -r requirements.txt
```

This project relies on Hydra to manage configurations. The configuration files are located in `conf/`.

**Typical usage**:

```bash
python trainer_base_ds_mul.py -cp <config_path> -cn <config_file_name>  # torch launcher
deepspeed --include localhost:0,1,2,3 trainer_base_ds_mul.py -cp <config_path> -cn <config_file_name>  # deepspeed launcher
```

## Datasets

You can download all datasets for self-supervised training from the [huggingface repo](https://huggingface.co/datasets/chitanda/wiki_erica_path_v9.1/tree/main),
which also contains the processed datasets containing already constructed logically consistent pairs.

### Preprocessing

If you want to preprocess the datasets by yourself, you can simply run the following command:
```bash
python trainer_base_ds_mul.py -cp <config_path> -cn <config_file_name> do_preprocess=True
```
This will stops the program after the datasets have been prepared. You can also remove `do_preprocess=True` so that the program will start training immediately.
However, this is **not encouraged**, as the preprocessing step is time-consuming, and usually the training requires distributed training,
which means that the other processes will be waiting for the data to be ready.


## Training

All configs for training different models are listed as follows:
- LLaMA-7B
  - Config: `conf/llama/wiki/llama_7b_merit_v1_pv91_v91_v5_0.yaml`
  - Weights: [Huggingface Hub](https://huggingface.co/chitanda/llama.7b.merit_v91_v91.seq2seq.v5.0.3aug.w4.adamw.500steps.NA100.0504/tree/main)
- LLaMA-13B
  - Config: `conf/llama/wiki/llama_13b_merit_v1_pv91_v91_v5_0.yaml`
  - Weights: [Huggingface Hub](https://huggingface.co/chitanda/llama.13b.merit_v91_v91.seq2seq.v5.0.3aug.w8.adamw.500steps.NA100.0426/tree/main)
- LLaMA-33B (QLoRA)
  - Normal data : Counterfactual data = 1:3
    - Config: `conf/llama/wiki/llama_30b_merit_v1_pv91_v91_v5_0.yaml`
    - Weights: [Huggingface Hub](https://huggingface.co/chitanda/llama.30b.merit_v91_v91.seq2seq.v5.0.3aug.w4.adamw.400steps.NA100.0525/tree/main)
  - Normal data : Counterfactual data = 1:0
    - Config: `conf/llama/wiki/llama_30b_merit_v1_pv91_v91_v5_0_no_aug.yaml`
    - Weights: [Huggingface Hub](https://huggingface.co/chitanda/llama.30b.merit_v91_v91.seq2seq.v5.0.0aug.w4.adamw.400steps.A40.0602/tree/main)
  - Normal data : Counterfactual data = 1:1
    - Config: `conf/llama/wiki/llama_30b_merit_v1_pv91_v91_v5_0_1aug.yaml`
    - Weights: [Huggingface Hub](https://huggingface.co/chitanda/llama.30b.merit_v91_v91.seq2seq.v5.0.1aug.w4.adamw.400steps.H100.0125/tree/main)
- LLaMA-65B (QLoRA)
  - Config: `conf/llama/wiki/llama_65b_merit_v1_pv91_v91_v5_0.yaml`
- LLaMA-65B (Full parameter training w. Pipeline Parallel)
  - Config: `conf/llama/wiki/llama_65b_merit_v1_pv91_v91_v5_0_full_mp.yaml`
  - Note: For pipeline parallel training, please launch the program using `trainer_base_ds_mp.py`. 
  Also, please first convert the Huggingface weights to DeepSpeed's format via `convert2ckpt.py`.
- Falcon-40B
  - Config: `conf/rw/falcon_40b_merit_v1_pv91_v91_v5_0.yaml`
  - Weights: [Huggingface Hub](https://huggingface.co/chitanda/falcon.40b.q_lora.merit_v91_v91.seq2seq.v5.0.3aug.w16.adamw.500steps.NA100.0528/tree/main)

## Evaluation

Since there are too many configs for evaluation in this repo, we only list one example here:
```bash
python trainer_base_fsdp_v4.py -cp conf/llama/wiki/mc_eval/ -cn llama_30b_merit_v5_qlora_logiqav2_eval_mc_v1_0_test  # This is for LogiQA-v2 multiple choice evaluation.
```

## Citation

If you find the repository and the paper helpful, please kindly cite our papers:
```
@article{logicllm2023jiao,
  author       = {Fangkai Jiao and
                  Zhiyang Teng and
                  Shafiq R. Joty and
                  Bosheng Ding and
                  Aixin Sun and
                  Zhengyuan Liu and
                  Nancy F. Chen},
  title        = {LogicLLM: Exploring Self-supervised Logic-enhanced Training for Large
                  Language Models},
  journal      = {CoRR},
  volume       = {abs/2305.13718},
  year         = {2023},
}

@inproceedings{merit2022jiao,
  author       = {Fangkai Jiao and
                  Yangyang Guo and
                  Xuemeng Song and
                  Liqiang Nie},
  editor       = {Smaranda Muresan and
                  Preslav Nakov and
                  Aline Villavicencio},
  title        = {MERIt: Meta-Path Guided Contrastive Learning for Logical Reasoning},
  booktitle    = {Findings of the Association for Computational Linguistics: {ACL} 2022,
                  Dublin, Ireland, May 22-27, 2022},
  pages        = {3496--3509},
  publisher    = {Association for Computational Linguistics},
  year         = {2022},
}
```
