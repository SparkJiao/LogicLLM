#!/bin/bash

python align_triplet_text.py \
  --kg ../../wikidata5m/wikidata5m_transductive_train.txt \
  --entity_vocab ../../wikidata5m/wikidata5m_entity.txt \
  --relation_vocab ../../wikidata5m/wikidata5m_relation.txt \
  --corpus ../../wikidata5m/wikidata5m_text.txt \
  --num_workers 32 --output_dir ../../wikidata5m/triplet_text_align_v1.0