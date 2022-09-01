#!/bin/bash

#for split_id in {0..4}; do
#  python logical_circle_to_text.py \
#    --seed 42 --mode seq2seq_simple \
#    --path ../../wikidata5m/triplet_text_align_v1.0/logical_circle_once/logic_circle_d3_4_False_s42_v2_25_${split_id}.json \
#    --id2ent ../../wikidata5m/triplet_text_align_v1.0/id2ent.json \
#    --id2rel ../../wikidata5m/triplet_text_align_v1.0/id2rel.json \
#    --triplet2sent ../../wikidata5m/triplet_text_align_v1.0/triplet2sent.json \
#    --edge2rel ../../wikidata5m/triplet_text_align_v1.0/logical_circle_once/edge2rel.json \
#    --tokenizer ../../pretrained-models/t5-large-lm-adapt \
#    --num_workers 32 --output_dir ../../wikidata5m/triplet_text_align_v1.0/logical_circle_once
#done;

python logical_circle_to_text.py \
  --seed 42 --mode seq2seq_simple \
  --path "../../wikidata5m/triplet_text_align_v1.0/logical_circle_once/logic_circle_d3_4_False_s42_v2_25_[0-5].json" \
  --id2ent ../../wikidata5m/triplet_text_align_v1.0/id2ent.json \
  --id2rel ../../wikidata5m/triplet_text_align_v1.0/id2rel.json \
  --triplet2sent ../../wikidata5m/triplet_text_align_v1.0/triplet2sent.json \
  --edge2rel ../../wikidata5m/triplet_text_align_v1.0/logical_circle_once/edge2rel.json \
  --tokenizer ../../pretrained-models/t5-large-lm-adapt \
  --num_workers 4 --output_dir ../../wikidata5m/triplet_text_align_v1.0/logical_circle_once --glob_mark "0-5" --dev_num 10000
