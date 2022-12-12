python preprocess/clustering/prepare_sent_labels.py \
  --example_file wiki_erica_path/v9.1/train_distant.path_v9.1.train.re_id.pkl \
  --edge_relation_file wiki_erica_path/v9.1/pattern_rel_id_kmeans_v21/edge_rel_cluster_1000_s42.pt \
  --output_file wiki_erica_path/v9.1/pattern_rel_id_kmeans_v21/train_distant.path_v9.1.train.re_id.labeled_c1000_s42.pkl

python preprocess/clustering/prepare_sent_labels.py \
  --example_file wiki_erica_path/v9.1/train_distant.path_v9.1.dev.shuffle.re_id.pkl \
  --edge_relation_file wiki_erica_path/v9.1/pattern_rel_id_kmeans_v21/edge_rel_cluster_1000_s42.pt \
  --output_file wiki_erica_path/v9.1/pattern_rel_id_kmeans_v21/train_distant.path_v9.1.dev.shuffle.re_id.labeled_c1000_s42.pkl
