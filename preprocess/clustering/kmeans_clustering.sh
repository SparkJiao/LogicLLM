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

# cluster into 1000 clusters

#python preprocess/clustering/entity_pair_clustering.py \
#  --ent_hidden_file experiments/roberta.base.erica.wiki_prediction_entity/ent_hidden.pt \
#  --edge_relation_file wiki_erica_path/v9.1/edge_rel_train_0.pt \
#  --output_file wiki_erica_path/v9.1/edge_rel_train_0_cluster_1000_s42.pt \
#  --batch_size 5000 --seed 42 --num_clusters 1000
#
#echo " ============================================== "
#
#python preprocess/clustering/entity_pair_clustering.py \
#--ent_hidden_file experiments/roberta.base.erica.wiki_prediction_entity/ent_hidden.pt \
#--edge_relation_file wiki_erica_path/v9.1/edge_rel_dev_shuffle.pt \
#--processed_clusters wiki_erica_path/v9.1/edge_rel_train_0_cluster_1000_s42.pt \
#--output_file wiki_erica_path/v9.1/edge_rel_dev_shuffle_cluster_1000_s42.pt --batch_size 3000 --seed 42 --num_clusters 1000
#
#echo " ============================================== "
#
#python preprocess/clustering/join_train_val_edge_labels.py --train_file wiki_erica_path/v9.1/edge_rel_train_0_cluster_1000_s42.pt \
#  --val_file wiki_erica_path/v9.1/edge_rel_dev_shuffle_cluster_1000_s42.pt \
#  --output_file wiki_erica_path/v9.1/pattern_rel_id_kmeans_v21/edge_rel_cluster_1000_s42.pt
#
#echo " =============================================== "
#
#python preprocess/clustering/reading_edge_cluster_labels.py \
#  --input_file wiki_erica_path/v9.1/train_distant.path_v9.1.train.re_id.pkl \
#  --edge_relation_file wiki_erica_path/v9.1/pattern_rel_id_kmeans_v21/edge_rel_cluster_1000_s42.pt \
#  --path_output_file wiki_erica_path/v9.1/pattern_rel_id_kmeans_v21/pattern_decode_id.train.0.c1000.limit0.pkl \
#  --rel_vocab wiki_erica_path/v9.1/pattern_rel_id_kmeans_v21/rel_vocab_c1000.pt \
#  --output_file wiki_erica_path/v9.1/pattern_rel_id_kmeans_v21/rel_path_set.train.0.pt
#  --edge_weights_save wiki_erica_path/v9.1/pattern_rel_id_kmeans_v21/edge_weights.pt
#
#echo " =============================================== "
#
#python preprocess/clustering/reading_edge_cluster_labels.py \
#  --input_file wiki_erica_path/v9.1/train_distant.path_v9.1.dev.shuffle.re_id.pkl \
#  --edge_relation_file wiki_erica_path/v9.1/pattern_rel_id_kmeans_v21/edge_rel_cluster_1000_s42.pt \
#  --path_output_file wiki_erica_path/v9.1/pattern_rel_id_kmeans_v21/pattern_decode_id.dev.shuffle.c1000.limit0.pkl \
#  --output_file wiki_erica_path/v9.1/pattern_rel_id_kmeans_v21/rel_path_set.dev.shuffle.pt \

# ==========================================

#python preprocess/clustering/extract_necessary_path.py \
#  --input_file wiki_erica_path/join2_limit3_v9.2/train_distant.path_v9.2_mm5.train.0.pkl \
#  --output_file wiki_erica_path/join2_limit3_v9.2/necessary_path_edge_train_0.pt
#
#echo " ============================================== "
#
#python preprocess/clustering/extract_necessary_path.py \
#  --input_file wiki_erica_path/join2_limit3_v9.2/train_distant.path_v9.2_mm5.dev.pkl \
#  --output_file wiki_erica_path/join2_limit3_v9.2/necessary_path_edge_dev.pt
#
#echo " ============================================== "
#
#python preprocess/clustering/entity_pair_clustering.py \
#  --ent_hidden_file experiments/roberta.base.erica.wiki_prediction_entity/ent_hidden.pt \
#  --edge_relation_file wiki_erica_path/join2_limit3_v9.2/necessary_path_edge_train_0.pt \
#  --output_file wiki_erica_path/join2_limit3_v9.2/edge_labels_train_0_cluster_2000_s42.pt \
#  --batch_size 6000 --seed 42 --num_clusters 2000
#

#python preprocess/clustering/entity_pair_clustering.py \
#  --ent_hidden_file experiments/roberta.base.erica.wiki_prediction_entity/ent_hidden.pt \
#  --edge_relation_file wiki_erica_path/join2_limit3_v9.2/necessary_path_edge_train_0.pt \
#  --output_file wiki_erica_path/join2_limit3_v9.2/edge_labels_train_0_cluster_1000_s42.pt \
#  --batch_size 3000 --seed 42 --num_clusters 1000

#echo " ============================================== "
#
#python preprocess/clustering/entity_pair_clustering.py \
#  --ent_hidden_file experiments/roberta.base.erica.wiki_prediction_entity/ent_hidden.pt \
#  --edge_relation_file wiki_erica_path/join2_limit3_v9.2/necessary_path_edge_dev.pt \
#  --processed_clusters wiki_erica_path/join2_limit3_v9.2/edge_labels_train_0_cluster_2000_s42.pt \
#  --output_file wiki_erica_path/join2_limit3_v9.2/edge_labels_dev_cluster_2000_s42.pt --batch_size 6000 --seed 42 --num_clusters 2000
#
#python preprocess/clustering/entity_pair_clustering.py \
#  --ent_hidden_file experiments/roberta.base.erica.wiki_prediction_entity/ent_hidden.pt \
#  --edge_relation_file wiki_erica_path/join2_limit3_v9.2/necessary_path_edge_dev.pt \
#  --processed_clusters wiki_erica_path/join2_limit3_v9.2/edge_labels_train_0_cluster_1000_s42.pt \
#  --output_file wiki_erica_path/join2_limit3_v9.2/edge_labels_dev_cluster_1000_s42.pt --batch_size 3000 --seed 42 --num_clusters 1000

#echo " ============================================== "
#
#python preprocess/clustering/join_train_val_edge_labels.py \
#  --train_file wiki_erica_path/join2_limit3_v9.2/edge_labels_train_0_cluster_2000_s42.pt \
#  --val_file wiki_erica_path/join2_limit3_v9.2/edge_labels_dev_cluster_2000_s42.pt \
#  --output_file wiki_erica_path/join2_limit3_v9.2/edge_labels_all_cluster_2000_s42.pt
#
#python preprocess/clustering/join_train_val_edge_labels.py \
#  --train_file wiki_erica_path/join2_limit3_v9.2/edge_labels_train_0_cluster_1000_s42.pt \
#  --val_file wiki_erica_path/join2_limit3_v9.2/edge_labels_dev_cluster_1000_s42.pt \
#  --output_file wiki_erica_path/join2_limit3_v9.2/edge_labels_all_cluster_1000_s42.pt

#echo " ============================================== "
#
#python preprocess/clustering/reading_edge_cluster_labels.py \
#  --input_file wiki_erica_path/join2_limit3_v9.2/train_distant.path_v9.2_mm5.train.0.pkl \
#  --edge_relation_file wiki_erica_path/join2_limit3_v9.2/edge_labels_all_cluster_2000_s42.pt \
#  --path_output_file wiki_erica_path/join2_limit3_v9.2/pattern_decode_id.train.0.c2000.limit0.pkl \
#  --rel_vocab wiki_erica_path/join2_limit3_v9.2/rel_vocab_c2000.pt \
#  --output_file wiki_erica_path/join2_limit3_v9.2/rel_path_set.train.0.pt \
#  --edge_weights_save wiki_erica_path/join2_limit3_v9.2/edge_weights.pt
#
#python preprocess/clustering/reading_edge_cluster_labels.py \
#  --input_file wiki_erica_path/join2_limit3_v9.2/train_distant.path_v9.2_mm5.train.0.pkl \
#  --edge_relation_file wiki_erica_path/join2_limit3_v9.2/edge_labels_all_cluster_1000_s42.pt \
#  --path_output_file wiki_erica_path/join2_limit3_v9.2/pattern_decode_id.train.0.c1000.limit0.pkl \
#  --rel_vocab wiki_erica_path/join2_limit3_v9.2/rel_vocab_c1000.pt \
#  --output_file wiki_erica_path/join2_limit3_v9.2/rel_path_set.c1000.train.0.pt
#  --edge_weights_save wiki_erica_path/join2_limit3_v9.2/edge_weights_c1000.pt

#echo " ============================================== "
#
#python preprocess/clustering/reading_edge_cluster_labels.py \
#  --input_file wiki_erica_path/join2_limit3_v9.2/train_distant.path_v9.2_mm5.dev.pkl \
#  --edge_relation_file wiki_erica_path/join2_limit3_v9.2/edge_labels_all_cluster_2000_s42.pt \
#  --path_output_file wiki_erica_path/join2_limit3_v9.2/pattern_decode_id.dev.c2000.limit0.pkl \
#  --output_file wiki_erica_path/join2_limit3_v9.2/rel_path_set.dev.pt
#
#python preprocess/clustering/reading_edge_cluster_labels.py \
#  --input_file wiki_erica_path/join2_limit3_v9.2/train_distant.path_v9.2_mm5.dev.pkl \
#  --edge_relation_file wiki_erica_path/join2_limit3_v9.2/edge_labels_all_cluster_1000_s42.pt \
#  --path_output_file wiki_erica_path/join2_limit3_v9.2/pattern_decode_id.dev.c1000.limit0.pkl \
#  --output_file wiki_erica_path/join2_limit3_v9.2/rel_path_set.c1000.dev.pt

#3.3493408262023356
#1395701
#1610376
# ==============================================
#3.356515630488233
#51729
#17063
# ==============================================
#1235331
#1395701
#Start
#4968
#1390733
#100%|██████████| 232/232 [3:50:44<00:00, 59.67s/it]
#100%|██████████| 232/232 [01:47<00:00,  2.15it/s]
#torch.Size([1390733])
#tensor([[-0.1034, -0.0505, -0.1454,  ...,  0.7422,  0.0742,  0.3086],
#        [ 0.5215, -0.0510, -0.5312,  ..., -0.4209, -0.3199, -0.0121],
#        [ 0.0160,  0.0715,  0.4600,  ...,  0.5493, -0.5605,  0.3799],
#        ...,
#        [ 0.0342, -0.1909, -0.1343,  ...,  0.8184, -0.5244,  0.4668],
#        [-0.4883, -0.2188,  0.3672,  ...,  0.3379, -0.2354, -0.2188],
#        [ 0.0488, -0.3633, -0.2539,  ..., -0.6484, -0.3379, -0.2773]])
#2000
# ==============================================
#1235331
#51729
#Loading defined clusters from wiki_erica_path/join2_limit3_v9.2/edge_labels_train_0_cluster_2000_s42.pt
#torch.Size([2000, 1536])
#100%|██████████| 9/9 [00:14<00:00,  1.66s/it]
#torch.Size([51601])
#tensor([[-0.1034, -0.0505, -0.1454,  ...,  0.7422,  0.0742,  0.3086],
#        [ 0.5215, -0.0510, -0.5312,  ..., -0.4209, -0.3199, -0.0121],
#        [ 0.0160,  0.0715,  0.4600,  ...,  0.5493, -0.5605,  0.3799],
#        ...,
#        [ 0.0342, -0.1909, -0.1343,  ...,  0.8184, -0.5244,  0.4668],
#        [-0.4883, -0.2188,  0.3672,  ...,  0.3379, -0.2354, -0.2188],
#        [ 0.0488, -0.3633, -0.2539,  ..., -0.6484, -0.3379, -0.2773]])
#1697
# ==============================================
#100%|██████████| 51601/51601 [00:00<00:00, 950418.63it/s]
# ==============================================
#1691208
#1691208
#100%|██████████| 1394987/1394987 [00:01<00:00, 1326355.45it/s]
#100%|██████████| 1691208/1691208 [00:09<00:00, 173844.36it/s]
#542045
#100%|██████████| 1691208/1691208 [00:03<00:00, 521797.04it/s]
#0 / 1685229 = 0.0
# ==============================================
#17082
#17082
#100%|██████████| 1394987/1394987 [00:01<00:00, 1383158.46it/s]
#100%|██████████| 17082/17082 [00:00<00:00, 178491.69it/s]
#13976
#100%|██████████| 17082/17082 [00:00<00:00, 649878.01it/s]
#0 / 17018 = 0.0


# ====================================== ####

num_clusters=2000


#python preprocess/clustering/extract_necessary_path.py \
#  --input_file wiki_erica_path/join2_limit3_v9.2_w_v9.1/train_distant.path.v9.2_mm5.train.pkl \
#  --output_file wiki_erica_path/join2_limit3_v9.2_w_v9.1/necessary_path_edge_train.pt
#
#echo " ============================================== "
#
#python preprocess/clustering/extract_necessary_path.py \
#  --input_file wiki_erica_path/join2_limit3_v9.2_w_v9.1/train_distant.path.v9.2_mm5.dev.pkl \
#  --output_file wiki_erica_path/join2_limit3_v9.2_w_v9.1/necessary_path_edge_dev.pt
#
#echo " ============================================== "

#python preprocess/clustering/entity_pair_clustering.py \
#  --ent_hidden_file experiments/roberta.base.erica.wiki_prediction_entity/ent_hidden.pt \
#  --edge_relation_file wiki_erica_path/join2_limit3_v9.2_w_v9.1/necessary_path_edge_train.pt \
#  --output_file wiki_erica_path/join2_limit3_v9.2_w_v9.1/edge_labels_train_cluster_${num_clusters}_s42.pt \
#  --batch_size 3000 --seed 42 --num_clusters ${num_clusters}
#
#echo " ============================================== "
#
#python preprocess/clustering/entity_pair_clustering.py \
#  --ent_hidden_file experiments/roberta.base.erica.wiki_prediction_entity/ent_hidden.pt \
#  --edge_relation_file wiki_erica_path/join2_limit3_v9.2_w_v9.1/necessary_path_edge_dev.pt \
#  --processed_clusters wiki_erica_path/join2_limit3_v9.2_w_v9.1/edge_labels_train_cluster_${num_clusters}_s42.pt \
#  --output_file wiki_erica_path/join2_limit3_v9.2_w_v9.1/edge_labels_dev_cluster_${num_clusters}_s42.pt \
#  --batch_size 3000 --seed 42 --num_clusters ${num_clusters}
#
#echo " ============================================== "
#
#python preprocess/clustering/join_train_val_edge_labels.py \
#  --train_file wiki_erica_path/join2_limit3_v9.2_w_v9.1/edge_labels_train_cluster_${num_clusters}_s42.pt \
#  --val_file wiki_erica_path/join2_limit3_v9.2_w_v9.1/edge_labels_dev_cluster_${num_clusters}_s42.pt \
#  --output_file wiki_erica_path/join2_limit3_v9.2_w_v9.1/edge_labels_all_cluster_${num_clusters}_s42.pt
#
#echo " ============================================== "
#
#python preprocess/clustering/reading_edge_cluster_labels.py \
#  --input_file wiki_erica_path/join2_limit3_v9.2_w_v9.1/train_distant.path.v9.2_mm5.train.pkl \
#  --edge_relation_file wiki_erica_path/join2_limit3_v9.2_w_v9.1/edge_labels_all_cluster_${num_clusters}_s42.pt \
#  --path_output_file wiki_erica_path/join2_limit3_v9.2_w_v9.1/pattern_decode_id.train.c${num_clusters}.limit0.pkl \
#  --rel_vocab wiki_erica_path/join2_limit3_v9.2_w_v9.1/rel_vocab_c${num_clusters}.pt \
#  --output_file wiki_erica_path/join2_limit3_v9.2_w_v9.1/rel_path_set.c${num_clusters}.train.pt \
#
#echo " ============================================== "
#
#python preprocess/clustering/reading_edge_cluster_labels.py \
#  --input_file wiki_erica_path/join2_limit3_v9.2_w_v9.1/train_distant.path.v9.2_mm5.dev.pkl \
#  --edge_relation_file wiki_erica_path/join2_limit3_v9.2_w_v9.1/edge_labels_all_cluster_${num_clusters}_s42.pt \
#  --path_output_file wiki_erica_path/join2_limit3_v9.2_w_v9.1/pattern_decode_id.dev.c${num_clusters}.limit0.pkl \
#  --output_file wiki_erica_path/join2_limit3_v9.2_w_v9.1/rel_path_set.c${num_clusters}.dev.pt



# cluster into 500 clusters
num_clusters=500

python preprocess/clustering/entity_pair_clustering.py \
  --ent_hidden_file experiments/roberta.base.erica.wiki_prediction_entity/ent_hidden.pt \
  --edge_relation_file wiki_erica_path/v9.1/edge_rel_train_0.pt \
  --output_file wiki_erica_path/v9.1/pattern_rel_id_kmeans_v22/edge_rel_train_cluster_${num_clusters}_s42.pt \
  --batch_size 5000 --seed 42 --num_clusters ${num_clusters}

echo " ============================================== "

python preprocess/clustering/entity_pair_clustering.py \
  --ent_hidden_file experiments/roberta.base.erica.wiki_prediction_entity/ent_hidden.pt \
  --edge_relation_file wiki_erica_path/v9.1/edge_rel_dev_shuffle.pt \
  --processed_clusters wiki_erica_path/v9.1/pattern_rel_id_kmeans_v22/edge_rel_train_cluster_${num_clusters}_s42.pt \
  --output_file wiki_erica_path/v9.1/pattern_rel_id_kmeans_v22/edge_rel_dev_shuffle_cluster_${num_clusters}_s42.pt \
  --batch_size 5000 --seed 42 --num_clusters ${num_clusters}

echo " ============================================== "

python preprocess/clustering/join_train_val_edge_labels.py \
  --train_file wiki_erica_path/v9.1/pattern_rel_id_kmeans_v22/edge_rel_train_cluster_${num_clusters}_s42.pt \
  --val_file wiki_erica_path/v9.1/pattern_rel_id_kmeans_v22/edge_rel_dev_shuffle_cluster_${num_clusters}_s42.pt \
  --output_file wiki_erica_path/v9.1/pattern_rel_id_kmeans_v22/edge_rel_${num_clusters}_s42.pt

echo " =============================================== "

python preprocess/clustering/reading_edge_cluster_labels.py \
  --input_file wiki_erica_path/v9.1/train_distant.path_v9.1.train.re_id.pkl \
  --edge_relation_file wiki_erica_path/v9.1/pattern_rel_id_kmeans_v22/edge_rel_${num_clusters}_s42.pt \
  --path_output_file wiki_erica_path/v9.1/pattern_rel_id_kmeans_v22/pattern_decode_id.train.0.c${num_clusters}.limit0.pkl \
  --rel_vocab wiki_erica_path/v9.1/pattern_rel_id_kmeans_v22/rel_vocab_c${num_clusters}.pt \
  --output_file wiki_erica_path/v9.1/pattern_rel_id_kmeans_v22/rel_path_set_c${num_clusters}.train.pt


echo " =============================================== "

python preprocess/clustering/reading_edge_cluster_labels.py \
  --input_file wiki_erica_path/v9.1/train_distant.path_v9.1.dev.shuffle.re_id.pkl \
  --edge_relation_file wiki_erica_path/v9.1/pattern_rel_id_kmeans_v22/edge_rel_${num_clusters}_s42.pt \
  --path_output_file wiki_erica_path/v9.1/pattern_rel_id_kmeans_v22/pattern_decode_id.dev.shuffle.c${num_clusters}.limit0.pkl \
  --output_file wiki_erica_path/v9.1/pattern_rel_id_kmeans_v22/rel_path_set_c${num_clusters}.dev.shuffle.pt

# ==========================================

#python preprocess/clustering/extract_necessary_path.py \
#  --input_file wiki_erica_path/join2_limit3_v9.2/train_distant.path_v9.2_mm5.train.0.pkl \
#  --output_file wiki_erica_path/join2_limit3_v9.2/necessary_path_edge_train_0.pt
#
#echo " ============================================== "
#
#python preprocess/clustering/extract_necessary_path.py \
#  --input_file wiki_erica_path/join2_limit3_v9.2/train_distant.path_v9.2_mm5.dev.pkl \
#  --output_file wiki_erica_path/join2_limit3_v9.2/necessary_path_edge_dev.pt
#
#echo " ============================================== "
#
#python preprocess/clustering/entity_pair_clustering.py \
#  --ent_hidden_file experiments/roberta.base.erica.wiki_prediction_entity/ent_hidden.pt \
#  --edge_relation_file wiki_erica_path/join2_limit3_v9.2/necessary_path_edge_train_0.pt \
#  --output_file wiki_erica_path/join2_limit3_v9.2/edge_labels_train_0_cluster_2000_s42.pt \
#  --batch_size 6000 --seed 42 --num_clusters 2000
#

#python preprocess/clustering/entity_pair_clustering.py \
#  --ent_hidden_file experiments/roberta.base.erica.wiki_prediction_entity/ent_hidden.pt \
#  --edge_relation_file wiki_erica_path/join2_limit3_v9.2/necessary_path_edge_train_0.pt \
#  --output_file wiki_erica_path/join2_limit3_v9.2/edge_labels_train_0_cluster_1000_s42.pt \
#  --batch_size 3000 --seed 42 --num_clusters 1000

#echo " ============================================== "
#
#python preprocess/clustering/entity_pair_clustering.py \
#  --ent_hidden_file experiments/roberta.base.erica.wiki_prediction_entity/ent_hidden.pt \
#  --edge_relation_file wiki_erica_path/join2_limit3_v9.2/necessary_path_edge_dev.pt \
#  --processed_clusters wiki_erica_path/join2_limit3_v9.2/edge_labels_train_0_cluster_2000_s42.pt \
#  --output_file wiki_erica_path/join2_limit3_v9.2/edge_labels_dev_cluster_2000_s42.pt --batch_size 6000 --seed 42 --num_clusters 2000
#
#python preprocess/clustering/entity_pair_clustering.py \
#  --ent_hidden_file experiments/roberta.base.erica.wiki_prediction_entity/ent_hidden.pt \
#  --edge_relation_file wiki_erica_path/join2_limit3_v9.2/necessary_path_edge_dev.pt \
#  --processed_clusters wiki_erica_path/join2_limit3_v9.2/edge_labels_train_0_cluster_1000_s42.pt \
#  --output_file wiki_erica_path/join2_limit3_v9.2/edge_labels_dev_cluster_1000_s42.pt --batch_size 3000 --seed 42 --num_clusters 1000

#echo " ============================================== "
#
#python preprocess/clustering/join_train_val_edge_labels.py \
#  --train_file wiki_erica_path/join2_limit3_v9.2/edge_labels_train_0_cluster_2000_s42.pt \
#  --val_file wiki_erica_path/join2_limit3_v9.2/edge_labels_dev_cluster_2000_s42.pt \
#  --output_file wiki_erica_path/join2_limit3_v9.2/edge_labels_all_cluster_2000_s42.pt
#
#python preprocess/clustering/join_train_val_edge_labels.py \
#  --train_file wiki_erica_path/join2_limit3_v9.2/edge_labels_train_0_cluster_1000_s42.pt \
#  --val_file wiki_erica_path/join2_limit3_v9.2/edge_labels_dev_cluster_1000_s42.pt \
#  --output_file wiki_erica_path/join2_limit3_v9.2/edge_labels_all_cluster_1000_s42.pt

#echo " ============================================== "
#
#python preprocess/clustering/reading_edge_cluster_labels.py \
#  --input_file wiki_erica_path/join2_limit3_v9.2/train_distant.path_v9.2_mm5.train.0.pkl \
#  --edge_relation_file wiki_erica_path/join2_limit3_v9.2/edge_labels_all_cluster_2000_s42.pt \
#  --path_output_file wiki_erica_path/join2_limit3_v9.2/pattern_decode_id.train.0.c2000.limit0.pkl \
#  --rel_vocab wiki_erica_path/join2_limit3_v9.2/rel_vocab_c2000.pt \
#  --output_file wiki_erica_path/join2_limit3_v9.2/rel_path_set.train.0.pt \
#  --edge_weights_save wiki_erica_path/join2_limit3_v9.2/edge_weights.pt
#
#python preprocess/clustering/reading_edge_cluster_labels.py \
#  --input_file wiki_erica_path/join2_limit3_v9.2/train_distant.path_v9.2_mm5.train.0.pkl \
#  --edge_relation_file wiki_erica_path/join2_limit3_v9.2/edge_labels_all_cluster_1000_s42.pt \
#  --path_output_file wiki_erica_path/join2_limit3_v9.2/pattern_decode_id.train.0.c1000.limit0.pkl \
#  --rel_vocab wiki_erica_path/join2_limit3_v9.2/rel_vocab_c1000.pt \
#  --output_file wiki_erica_path/join2_limit3_v9.2/rel_path_set.c1000.train.0.pt
#  --edge_weights_save wiki_erica_path/join2_limit3_v9.2/edge_weights_c1000.pt

#echo " ============================================== "
#
#python preprocess/clustering/reading_edge_cluster_labels.py \
#  --input_file wiki_erica_path/join2_limit3_v9.2/train_distant.path_v9.2_mm5.dev.pkl \
#  --edge_relation_file wiki_erica_path/join2_limit3_v9.2/edge_labels_all_cluster_2000_s42.pt \
#  --path_output_file wiki_erica_path/join2_limit3_v9.2/pattern_decode_id.dev.c2000.limit0.pkl \
#  --output_file wiki_erica_path/join2_limit3_v9.2/rel_path_set.dev.pt
#
#python preprocess/clustering/reading_edge_cluster_labels.py \
#  --input_file wiki_erica_path/join2_limit3_v9.2/train_distant.path_v9.2_mm5.dev.pkl \
#  --edge_relation_file wiki_erica_path/join2_limit3_v9.2/edge_labels_all_cluster_1000_s42.pt \
#  --path_output_file wiki_erica_path/join2_limit3_v9.2/pattern_decode_id.dev.c1000.limit0.pkl \
#  --output_file wiki_erica_path/join2_limit3_v9.2/rel_path_set.c1000.dev.pt
