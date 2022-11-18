
#python preprocess/clustering/erica_relation_clustering.py --ent_hidden_file_path "experiments/roberta.large.mlm.wiki_prediction_entity_[0-9]/test/predictions-rank[0-3].pt" \
#  --gpu_id 6 --output_file wiki_erica_path/v9.1/mlm_edge_representation/ent_hidden.pt


#python preprocess/clustering/entity_pair_clustering.py \
#--ent_hidden_file wiki_erica_path/v9.1/mlm_edge_representation/ent_hidden.pt \
#--edge_relation_file wiki_erica_path/v9.1/edge_rel_train_0.pt \
#--output_file wiki_erica_path/v9.1/mlm_edge_representation/edge_rel_train_0_cluster_1200_s42.pt \
#--batch_size 3000 --seed 42 --num_clusters 1200

python preprocess/clustering/entity_pair_clustering.py \
  --ent_hidden_file wiki_erica_path/v9.1/mlm_edge_representation/ent_hidden.pt \
  --edge_relation_file wiki_erica_path/v9.1/edge_rel_dev_shuffle.pt \
  --processed_clusters wiki_erica_path/v9.1/mlm_edge_representation/edge_rel_train_0_cluster_1200_s42.pt \
  --output_file wiki_erica_path/v9.1/mlm_edge_representation/edge_rel_dev_shuffle_cluster_1200_s42.pt --batch_size 3000 --seed 42
