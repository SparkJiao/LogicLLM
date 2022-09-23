#!/bin/bash

#for split_id in {0..24}; do
#for split_id in {6..24}; do
#  python preprocess/wikidata_5m/construct_logical_circle_once.py \
#    --kg wikidata5m/wikidata5m_transductive_train.txt \
#    --id2ent wikidata5m/triplet_text_align_v1.0/id2ent.json \
#    --id2rel wikidata5m/triplet_text_align_v1.0/id2rel.json \
#    --min_depth 3 --max_depth 4 --num_workers 32 \
#    --output_dir wikidata5m/triplet_text_align_v1.0/logical_circle_once --split_chunk 25,${split_id}
#done;

for split_id in {0..24}; do
  python preprocess/wikidata_5m/construct_logical_circle_once.py \
    --kg wikidata5m/wikidata5m_transductive_train.txt \
    --id2ent wikidata5m/triplet_text_align_v1.0/id2ent.json \
    --id2rel wikidata5m/triplet_text_align_v1.0/id2rel.json \
    --min_depth 2 --max_depth 2 --num_workers 32 \
    --output_dir wikidata5m/triplet_text_align_v1.0/logical_circle_once_2hop --split_chunk 25,${split_id}
done;
