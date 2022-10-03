#!/bin/bash

python entity_mask_and_encode.py --input_file "../../wiki_erica_path/sampled_data_roberta/train_distant*" \
  -o "mask" --num_workers 16 --output_dir "../../wiki_erica_path/data_roberta_simsce_rank/masked/"
