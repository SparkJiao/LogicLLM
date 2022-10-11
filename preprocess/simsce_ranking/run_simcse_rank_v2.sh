#!/bin/bash

#python simcse_ranking_v2.py --input_file ../../wiki_erica_path/data_roberta_simsce_rank/masked/train_distant_0_mask.json \
#  --corpus_file "../../wiki_erica_path/sampled_data_roberta/train_distant*" \
#  --corpus_index_file ../../wiki_erica_path/data_roberta_simsce_rank/embeddings.pt \
#  --output_dir ../../wiki_erica_path/data_roberta_simsce_rank/masked/ \
#  --num_workers 8 --cpu


python simcse_ranking_v2.py --input_file ../../wiki_erica_path/data_roberta_simsce_rank/masked/train_distant_0_mask.json \
  --corpus_file "../../wiki_erica_path/sampled_data_roberta/train_distant*" \
  --corpus_index_file ../../wiki_erica_path/data_roberta_simsce_rank/embeddings.pt \
  --output_dir ../../wiki_erica_path/data_roberta_simsce_rank/masked/ \
  --num_workers 8 --query_batch_size 4 --shard
