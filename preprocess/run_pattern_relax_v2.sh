python preprocess/unstructured_pattern_check_relax_v2.py \
  --input_file wiki_erica_path/v9.1/train_distant.path_v9.1.train.re_id.pkl \
  --output_file wiki_erica_path/v9.1/pattern_rel_id_relax_v20_3hop_limit0/rel_path_set.train.pt \
  --num_workers 64 \
  --kg ../research.data/wikidata5m/wikidata5m_transductive_train.txt \
  --path_output_file wiki_erica_path/v9.1/pattern_rel_id_relax_v20_3hop_limit0/pattern_decode_id.train.3hop.limit0.pkl \
  --rel_vocab wiki_erica_path/v9.1/pattern_rel_id_relax_v20_3hop_limit0/rel_vocab.pt \
  --hop 2 \
  --limit 0

