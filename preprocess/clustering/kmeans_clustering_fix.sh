num_clusters=1024
seed=42
train_file=wiki_erica_path/v9.1_fixed/distant_path_v9.1_fix_no_shuffle.train.0.pkl
dev_file=wiki_erica_path/v9.1_fixed/distant_path_v9.1_fix_no_shuffle.dev.pkl
data_dir=wiki_erica_path/v9.1_fixed
edge_relation_file_train=${data_dir}/necessary_path_edge_train.pt
edge_relation_file_dev=${data_dir}/necessary_path_edge_dev.pt
cluster_file_train=${data_dir}/edge_labels_train_cluster_c${num_clusters}_s${seed}.pt
cluster_file_dev=${data_dir}/edge_labels_dev_cluster_c${num_clusters}_s${seed}.pt
cluster_file=${data_dir}/edge_labels_all_cluster_c${num_clusters}_s${seed}.pt
pattern_decode_id_train=${data_dir}/pattern_decode_id.train.c${num_clusters}.limit0.pkl
pattern_decode_id_dev=${data_dir}/pattern_decode_id.dev.c${num_clusters}.limit0.pkl

python preprocess/clustering/extract_necessary_path.py \
  --input_file ${train_file} \
  --output_file ${edge_relation_file_train}

python preprocess/clustering/extract_necessary_path.py \
  --input_file ${dev_file} \
  --output_file ${edge_relation_file_dev}

python preprocess/clustering/entity_pair_clustering.py \
  --ent_hidden_file experiments/roberta.base.erica.wiki_prediction_entity/ent_hidden.pt \
  --edge_relation_file ${edge_relation_file_train} \
  --output_file ${cluster_file_train} \
  --batch_size 3000 --seed ${seed} --num_clusters ${num_clusters}

python preprocess/clustering/entity_pair_clustering.py \
  --ent_hidden_file experiments/roberta.base.erica.wiki_prediction_entity/ent_hidden.pt \
  --edge_relation_file ${edge_relation_file_dev} \
  --processed_clusters ${cluster_file_train} \
  --output_file ${cluster_file_dev} \
  --batch_size 3000 --seed ${seed} --num_clusters ${num_clusters}

python preprocess/clustering/join_train_val_edge_labels.py \
  --train_file ${cluster_file_train} \
  --val_file ${cluster_file_dev} \
  --output_file ${cluster_file}

python preprocess/clustering/reading_edge_cluster_labels.py \
  --input_file ${train_file} \
  --edge_relation_file ${cluster_file} \
  --path_output_file ${pattern_decode_id_train} \
  --rel_vocab ${data_dir}/rel_vocab_c${num_clusters}.pt \
  --output_file ${data_dir}/rel_path_set.c${num_clusters}.train.pt

python preprocess/clustering/reading_edge_cluster_labels.py \
  --input_file ${dev_file} \
  --edge_relation_file ${cluster_file} \
  --path_output_file ${pattern_decode_id_dev} \
  --output_file ${data_dir}/rel_path_set.c${num_clusters}.dev.pt
