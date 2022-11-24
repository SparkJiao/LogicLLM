#python preprocess/clustering/reading_edge_cluster_labels.py \
#  --input_file wiki_erica_path/v9.1/train_distant.path_v9.1.train.re_id.pkl \
#  --edge_relation_file "wiki_erica_path/v9.1/pattern_rel_id_kmeans_v10/edge_rel_cluster_2000_s42.pt" \
#  --path_output_file wiki_erica_path/v9.1/pattern_rel_id_kmeans_v20/pattern_decode_id.train.0.c2000.limit0.pkl \
#  --rel_vocab wiki_erica_path/v9.1/pattern_rel_id_kmeans_v20/rel_vocab_c2000.pt \
#  --output_file wiki_erica_path/v9.1/pattern_rel_id_kmeans_v20/rel_path_set.train.0.pt \
#  --edge_weights_save wiki_erica_path/v9.1/pattern_rel_id_kmeans_v20/edge_weights.pt
#
#python preprocess/clustering/reading_edge_cluster_labels.py \
#  --input_file wiki_erica_path/v9.1/train_distant.path_v9.1.dev.shuffle.re_id.pkl \
#  --edge_relation_file "wiki_erica_path/v9.1/pattern_rel_id_kmeans_v10/edge_rel_cluster_2000_s42.pt" \
#  --path_output_file wiki_erica_path/v9.1/pattern_rel_id_kmeans_v20/pattern_decode_id.dev.shuffle.c2000.limit0.pkl \
#  --output_file wiki_erica_path/v9.1/pattern_rel_id_kmeans_v20/rel_path_set.dev.shuffle.pt \

# =========================================

python preprocess/clustering/extract_necessary_path.py \
  --input_file wiki_erica_path/join2_limit3_v9.2/train_distant.path_v9.2_mm5.train.0.pkl \
  --output_file wiki_erica_path/join2_limit3_v9.2/necessary_path_edge_train_0.pt

echo " ============================================== "

python preprocess/clustering/extract_necessary_path.py \
  --input_file wiki_erica_path/join2_limit3_v9.2/train_distant.path_v9.2_mm5.dev.pkl \
  --output_file wiki_erica_path/join2_limit3_v9.2/necessary_path_edge_dev.pt

echo " ============================================== "

python preprocess/clustering/entity_pair_clustering.py \
  --ent_hidden_file experiments/roberta.base.erica.wiki_prediction_entity/ent_hidden.pt \
  --edge_relation_file wiki_erica_path/join2_limit3_v9.2/necessary_path_edge_train_0.pt \
  --output_file wiki_erica_path/join2_limit3_v9.2/edge_labels_train_0_cluster_2000_s42.pt \
  --batch_size 3000 --seed 42 --num_clusters 2000

echo " ============================================== "

python preprocess/clustering/entity_pair_clustering.py \
  --ent_hidden_file experiments/roberta.base.erica.wiki_prediction_entity/ent_hidden.pt \
  --edge_relation_file wiki_erica_path/join2_limit3_v9.2/necessary_path_edge_dev.pt \
  --processed_clusters wiki_erica_path/join2_limit3_v9.2/edge_labels_train_0_cluster_2000_s42.pt \
  --output_file wiki_erica_path/join2_limit3_v9.2/edge_labels_dev_cluster_2000_s42.pt --batch_size 3000 --seed 42 --num_clusters 2000

echo " ============================================== "

python preprocess/clustering/join_train_val_edge_labels.py \
  --train_file wiki_erica_path/join2_limit3_v9.2/edge_labels_train_0_cluster_2000_s42.pt \
  --val_file wiki_erica_path/join2_limit3_v9.2/edge_labels_dev_cluster_2000_s42.pt \
  --output_file wiki_erica_path/join2_limit3_v9.2/edge_labels_all_cluster_2000_s42.pt

echo " ============================================== "

python preprocess/clustering/reading_edge_cluster_labels.py \
  --input_file wiki_erica_path/join2_limit3_v9.2/train_distant.path_v9.2_mm5.train.0.pkl \
  --edge_relation_file wiki_erica_path/join2_limit3_v9.2/edge_labels_all_cluster_2000_s42.pt \
  --path_output_file wiki_erica_path/join2_limit3_v9.2/pattern_decode_id.train.0.c2000.limit0.pkl \
  --rel_vocab wiki_erica_path/join2_limit3_v9.2/rel_vocab_c2000.pt \
  --output_file wiki_erica_path/join2_limit3_v9.2/rel_path_set.train.0.pt \
  --edge_weights_save wiki_erica_path/join2_limit3_v9.2/edge_weights.pt

echo " ============================================== "

python preprocess/clustering/reading_edge_cluster_labels.py \
  --input_file wiki_erica_path/join2_limit3_v9.2/train_distant.path_v9.2_mm5.dev.pkl \
  --edge_relation_file wiki_erica_path/join2_limit3_v9.2/edge_labels_all_cluster_2000_s42.pt \
  --path_output_file wiki_erica_path/join2_limit3_v9.2/pattern_decode_id.dev.c2000.limit0.pkl \
  --output_file wiki_erica_path/join2_limit3_v9.2/rel_path_set.dev.pt
