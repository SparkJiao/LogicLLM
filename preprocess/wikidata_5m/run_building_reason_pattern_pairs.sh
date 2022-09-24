#!/bin/bash

seed=42

python preprocess/wikidata_5m/build_reason_pattern_pairs.py \
  --path "wikidata5m/triplet_text_align_v1.0/logical_circle_once*/raw_data/*.json" \
  --edge2rel wikidata5m/triplet_text_align_v1.0/logical_circle_once/edge2rel.json \
  --triplet2sent wikidata5m/triplet_text_align_v1.0/triplet2sent.json\
  --id2ent wikidata5m/triplet_text_align_v1.0/id2ent.json \
  --id2rel wikidata5m/triplet_text_align_v1.0/id2rel.json \
  --pattern_pair_save_file wikidata5m/triplet_text_align_v1.0/logical_pattern_pairs/reason_pattern_pairs_v1_s${seed}.json  \
  --text_pair_save_file wikidata5m/triplet_text_align_v1.0/logical_pattern_pairs/reason_pattern_text_pairs_v1_5_5_${seed}.json \
  --num_workers 32 --max_query_per_rel 5 --max_path_per_rel 5