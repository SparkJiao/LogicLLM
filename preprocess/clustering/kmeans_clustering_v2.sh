seed=42
#num_clusters=1024
num_clusters=512
save_dir=wiki_erica_path/v9.1_fixed/kmeans_v2
example_file='wiki_erica_path/v9.1_fixed/train_distant_*.path_v9.1.pkl'
entity_hidden_file=experiments/roberta.base.erica.entity_no_none_cp2500/test-checkpoint-2500/predictions.pt
exp_id2edge_hidden=${save_dir}/exp_id2edge_hidden.pt
edge_cluster_file=${save_dir}/edge_cluster_c${num_clusters}_s${seed}.pt
train_file=wiki_erica_path/v9.1_fixed/distant_path_v9.1_fix_no_shuffle.train.0.pkl
dev_file=wiki_erica_path/v9.1_fixed/distant_path_v9.1_fix_no_shuffle.dev.pkl

rel_path_set_train=${save_dir}/rel_path_set.train.c${num_clusters}.pkl
rel_path_set_dev=${save_dir}/rel_path_set.dev.c${num_clusters}.pkl

pattern_decode_id_train=${save_dir}/pattern_decode_id.train.c${num_clusters}.limit0.pkl
pattern_decode_id_dev=${save_dir}/pattern_decode_id.dev.c${num_clusters}.limit0.pkl

rel_vocab=${save_dir}/rel_vocab_c${num_clusters}.pkl

#python preprocess/clustering/extract_path_edge_hidden.py \
#  --example_file "wiki_erica_path/v9.1_fixed/train_distant_*.path_v9.1.pkl" \
#  --entity_hidden_file ${entity_hidden_file} \
#  --output_file ${exp_id2edge_hidden}

python preprocess/clustering/entity_pair_clustering_v2.py \
  --exp_id2edge_hidden_file ${exp_id2edge_hidden} \
  --output_file ${edge_cluster_file} \
  --batch_size 3000 --seed ${seed} --num_clusters ${num_clusters}

python preprocess/clustering/reading_edge_cluster_labels_v2.py \
  --input_file ${train_file} \
  --edge_cluster_file ${edge_cluster_file} \
  --path_output_file ${pattern_decode_id_train} \
  --num_workers 64 --output_file ${rel_path_set_train} --rel_vocab ${rel_vocab}

python preprocess/clustering/reading_edge_cluster_labels_v2.py \
  --input_file ${dev_file} \
  --edge_cluster_file ${edge_cluster_file} \
  --path_output_file ${pattern_decode_id_dev} \
  --num_workers 64 --output_file ${rel_path_set_dev}
