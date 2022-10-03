#!/bin/bash

#python preprocess/wiki_wiki/filter_wikiwiki.py --kg_file wikidata5m/wikidata5m_transductive_train.txt \
#  --wikiwiki ../research.data/wikiwiki-dataset/data/train.jsonl.filtered \
#  --sent_num 5 --edge_num 5 --ent_num 0 --num_workers 16 \
#  --output_dir ../research.data/wikiwiki-dataset/data/sent_edge_entity_filter

python preprocess/wiki_wiki/filter_wikiwiki.py --kg_file wikidata5m/wikidata5m_transductive_train.txt \
  --wikiwiki ../research.data/wikiwiki-dataset/data/train.jsonl.filtered \
  --sent_num 5 --edge_num 0 --ent_num 5 --num_workers 16 \
  --output_dir ../research.data/wikiwiki-dataset/data/sent_edge_entity_filter
