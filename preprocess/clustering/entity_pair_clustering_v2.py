import argparse
import torch
from typing import Dict, List
from kmeans_pytorch import pairwise_cosine, pairwise_distance, initialize, kmeans, kmeans_predict
import sys
from tqdm import tqdm, trange
import time


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_id2edge_hidden_file", type=str)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--processed_clusters", type=str, default=None)
    parser.add_argument("--num_clusters", type=int, default=50)
    args = parser.parse_args()

    return args


def extract_edge_hidden(exp_id2edge_hidden):
    exp_ids = []
    edges = []
    hidden = []
    for exp_id, exp_edge_hidden in tqdm(exp_id2edge_hidden.items(), total=len(exp_id2edge_hidden), desc="Extracting edge hidden..."):
        for edge, edge_hidden in exp_edge_hidden.items():
            exp_ids.append(exp_id)
            edges.append(edge)
            hidden.append(edge_hidden)
    return exp_ids, edges, hidden


def main():
    args = parse_args()

    # ent_hidden = torch.load(args.ent_hidden_file, map_location="cpu")
    # ent_pairs = torch.load(args.edge_relation_file)
    # print(len(ent_hidden))
    # print(len(ent_pairs))
    exp_id2edge_hidden = torch.load(args.exp_id2edge_hidden_file, map_location="cpu")

    # edge_hidden, ignored_edge = process_edge_hidden(ent_hidden, ent_pairs)
    exp_ids, edges, hidden = extract_edge_hidden(exp_id2edge_hidden)
    edge_hidden_tensor = torch.stack(hidden, dim=0)

    if args.processed_clusters is None:
        print("Start")

        cluster_centers = []
        for i in trange(0, edge_hidden_tensor.size(0), args.batch_size):
            batch = edge_hidden_tensor[i: (i + args.batch_size)]
            _, cluster_centers = kmeans(
                X=batch, num_clusters=args.num_clusters, distance="cosine", cluster_centers=cluster_centers, device=torch.device("cuda:0"),
                seed=args.seed, tqdm_flag=False
            )
    else:
        print(f"Loading defined clusters from {args.processed_clusters}")
        _, _, cluster_centers = torch.load(args.processed_clusters, map_location="cpu")
        print(cluster_centers.size())

    cluster_ids_x = []
    for i in trange(0, edge_hidden_tensor.size(0), args.batch_size):
        batch = edge_hidden_tensor[i: (i + args.batch_size)]
        batch_cluster_ids = kmeans_predict(
            X=batch, cluster_centers=cluster_centers, distance="cosine", device=torch.device("cuda:0"), tqdm_flag=False
        )
        cluster_ids_x.append(batch_cluster_ids)

    cluster_ids_x = torch.cat(cluster_ids_x, dim=0)
    print(cluster_ids_x.size())
    print(cluster_centers)
    print(len(set(cluster_ids_x.tolist())))
    torch.save(((exp_ids, edges), cluster_ids_x, cluster_centers), args.output_file)


if __name__ == '__main__':
    main()
