```bash

HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=1,2,3 run1 bash scripts/roberta/wiki/predict/run_erica_prediction.sh

run1 python preprocess/clustering/erica_relation_clustering.py --ent_hidden_file_path "experiments/roberta.base.erica.wiki_prediction_entity_[0-9]/test/predictions-rank[0-3].pt" \
  --gpu_id 1 --output_file experiments/roberta.base.erica.wiki_prediction_entity/ent_hidden.pt

python preprocess/clustering/extract_necessary_path.py \
  --input_file wiki_erica_path/v9.1/pattern_rel_id_v10/train_distant.path_v9.1.train.0.pkl \
  --output_file wiki_erica_path/v9.1/edge_rel_train_0.pt
  
3.1400256593739284  
551819  
453685

CUDA_VISIBLE_DEVICES=1 run1 python preprocess/clustering/entity_pair_clustering.py \
  --ent_hidden_file experiments/roberta.base.erica.wiki_prediction_entity/ent_hidden.pt \
  --edge_relation_file wiki_erica_path/v9.1/edge_rel_train_0.pt \
  --output_file wiki_erica_path/v9.1/edge_rel_train_0_cluster_s42.pt --batch_size 100000 --seed 42
  
1235331  
551819  
Start  
2895   
548924
100%|██████████| 6/6 [00:24<00:00,  4.15s/it]  
100%|██████████| 6/6 [00:01<00:00,  5.11it/s]                                 
torch.Size([548924])                                                       
tensor([[ 0.3683, -0.2355, -0.4253,  ...,  0.5834,  0.1306, -0.0376],
        [ 0.1877, -0.1157, -0.2962,  ...,  0.9242, -0.2606,  0.2632],
        [-0.2011, -0.0018,  0.0555,  ...,  0.8334, -0.3041,  0.2671],
        ...,
        [-0.1557,  0.0314,  0.1167,  ..., -0.1082,  0.0948,  0.2012],
        [-0.0057, -0.0557, -0.1869,  ...,  0.5964,  0.1385, -0.0177],
        [-0.2659, -0.2413,  0.0458,  ...,  0.7152, -0.3275,  0.2594]])
50
  
python preprocess/clustering/extract_necessary_path.py \
  --input_file wiki_erica_path/v9.1/pattern_rel_id_v10/train_distant.path_v9.1.dev.shuffle.pkl \
  --output_file wiki_erica_path/v9.1/edge_rel_dev_shuffle.pt
  
3.1414961320071644   
132083
52175

CUDA_VISIBLE_DEVICES=1 run1 python preprocess/clustering/entity_pair_clustering.py \
  --ent_hidden_file experiments/roberta.base.erica.wiki_prediction_entity/ent_hidden.pt \
  --edge_relation_file wiki_erica_path/v9.1/edge_rel_dev_shuffle.pt \
  --processed_clusters wiki_erica_path/v9.1/edge_rel_train_0_cluster_s42.pt \
  --output_file wiki_erica_path/v9.1/edge_rel_dev_shuffle_cluster_s42.pt --batch_size 100000 --seed 42

1235331
132083
Loading defined clusters from wiki_erica_path/v9.1/edge_rel_train_0_cluster_s42.pt
torch.Size([50, 1536])
100%|██████████| 2/2 [00:01<00:00,  1.81it/s]
torch.Size([131449])
tensor([[ 0.3683, -0.2355, -0.4253,  ...,  0.5834,  0.1306, -0.0376],
        [ 0.1877, -0.1157, -0.2962,  ...,  0.9242, -0.2606,  0.2632],
        [-0.2011, -0.0018,  0.0555,  ...,  0.8334, -0.3041,  0.2671],
        ...,
        [-0.1557,  0.0314,  0.1167,  ..., -0.1082,  0.0948,  0.2012],
        [-0.0057, -0.0557, -0.1869,  ...,  0.5964,  0.1385, -0.0177],
        [-0.2659, -0.2413,  0.0458,  ...,  0.7152, -0.3275,  0.2594]])
50

run1 python preprocess/clustering/reading_edge_cluster_labels.py \          
   --input_file wiki_erica_path/v9.1/train_distant.path_v9.1.train.0.pkl \ 
   --edge_relation_file "wiki_erica_path/v9.1/pattern_rel_id_kmeans_v10/edge_rel_cluster_s4?.pt" \  
   --path_output_file wiki_erica_path/v9.1/pattern_rel_id_kmeans_v10/pattern_decode_id.train.0.limit0.pkl \ 
   --rel_vocab wiki_erica_path/v9.1/pattern_rel_id_kmeans_v10/rel_vocab.pt
  
472342
472342
['wiki_erica_path/v9.1/pattern_rel_id_kmeans_v10/edge_rel_cluster_s43.pt', 'wiki_erica_path/v9.1/pattern_rel_id_kmeans_v10/edge_rel_cluster_s42.pt']
Combined edge labels: 884
100%|██████████| 472342/472342 [00:01<00:00, 242552.67it/s]
35979 
100%|██████████| 472342/472342 [00:00<00:00, 518329.06it/s]
0 / 469392 = 0.0

run1 python preprocess/clustering/reading_edge_cluster_labels.py \
  --input_file wiki_erica_path/v9.1/pattern_rel_id_v10/train_distant.path_v9.1.dev.shuffle.pkl \
  --edge_relation_file "wiki_erica_path/v9.1/pattern_rel_id_kmeans_v10/edge_rel_cluster_s4?.pt" \
  --path_output_file wiki_erica_path/v9.1/pattern_rel_id_kmeans_v10/pattern_decode_id.dev.shuffle.limit0.pkl
  
52482 
52482 
['wiki_erica_path/v9.1/pattern_rel_id_kmeans_v10/edge_rel_cluster_s43.pt', 'wiki_erica_path/v9.1/pattern_rel_id_kmeans_v10/edge_rel_cluster_s42.pt'] 
Combined edge labels: 884 
100%|██████████| 52482/52482 [00:00<00:00, 169609.09it/s] 
9280 
100%|██████████| 52482/52482 [00:00<00:00, 360971.48it/s] 
0 / 52128 = 0.0

```