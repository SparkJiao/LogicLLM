import argparse
import torch
from typing import Dict, List
from kmeans_pytorch import pairwise_cosine, pairwise_distance, initialize, kmeans, kmeans_predict
import sys
from tqdm import tqdm, trange
import time


# def kmeans(
#         X,
#         num_clusters,
#         distance='euclidean',
#         tol=1e-4,
#         device=torch.device('cpu')
# ):
#     """
#     perform kmeans
#     :param X: (torch.tensor) matrix
#     :param num_clusters: (int) number of clusters
#     :param distance: (str) distance [options: 'euclidean', 'cosine'] [default: 'euclidean']
#     :param tol: (float) threshold [default: 0.0001]
#     :param device: (torch.device) device [default: cpu]
#     :return: (torch.tensor, torch.tensor) cluster ids, cluster centers
#     """
#     print(f'running k-means on {device}..')
#
#     if distance == 'euclidean':
#         pairwise_distance_function = pairwise_distance
#     elif distance == 'cosine':
#         pairwise_distance_function = pairwise_cosine
#     else:
#         raise NotImplementedError
#
#     # convert to float
#     X = X.float()
#
#     # transfer to device
#     X = X.to(device)
#
#     # initialize
#     initial_state = initialize(X, num_clusters)
#
#     iteration = 0
#     tqdm_meter = tqdm(desc='[running kmeans]')
#     while True:
#         dis = pairwise_distance_function(X, initial_state, device)
#
#         choice_cluster = torch.argmin(dis, dim=1)
#
#         initial_state_pre = initial_state.clone()
#
#         for index in range(num_clusters):
#             selected = torch.nonzero(choice_cluster == index).squeeze().to(device)
#
#             selected = torch.index_select(X, 0, selected)
#             initial_state[index] = selected.mean(dim=0)
#
#         center_shift = torch.sum(
#             torch.sqrt(
#                 torch.sum((initial_state - initial_state_pre) ** 2, dim=1)
#             ))
#
#         # increment iteration
#         iteration = iteration + 1
#
#         # update tqdm meter
#         tqdm_meter.set_postfix(
#             iteration=f'{iteration}',
#             center_shift=f'{center_shift ** 2:0.6f}',
#             tol=f'{tol:0.6f}'
#         )
#         tqdm_meter.update()
#         if center_shift ** 2 < tol:
#             break
#
#     return choice_cluster.cpu(), initial_state.cpu()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--edge_relation_file", type=str)
    parser.add_argument("--ent_hidden_file", type=str)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--processed_clusters", type=str, default=None)
    args = parser.parse_args()

    return args


def process_edge_hidden(ent_hidden: Dict[str, torch.Tensor], ent_pairs: List[str]):
    edge_hidden = {}
    ignored_edge = 0
    for edge in ent_pairs:
        s, e = edge.split("\t")
        if s not in ent_hidden or e not in ent_hidden:
            ignored_edge += 1
            continue
        s_h = ent_hidden[s]
        e_h = ent_hidden[e]
        edge_hidden[edge] = torch.cat([s_h, e_h], dim=-1)
    return edge_hidden, ignored_edge


def main():
    args = parse_args()

    ent_hidden = torch.load(args.ent_hidden_file, map_location="cpu")
    ent_pairs = torch.load(args.edge_relation_file)
    print(len(ent_hidden))
    print(len(ent_pairs))

    edge_hidden, ignored_edge = process_edge_hidden(ent_hidden, ent_pairs)
    edge_hidden_tensor = torch.stack(list(edge_hidden.values()), dim=0)

    if args.processed_clusters is None:
        print("Start")
        print(ignored_edge)
        print(len(edge_hidden))

        cluster_centers = []
        for i in trange(0, edge_hidden_tensor.size(0), args.batch_size):
            batch = edge_hidden_tensor[i: (i + args.batch_size)]
            _, cluster_centers = kmeans(
                X=batch, num_clusters=50, distance="cosine", cluster_centers=cluster_centers, device=torch.device("cuda:0"),
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
    torch.save((edge_hidden, cluster_ids_x, cluster_centers), args.output_file)


if __name__ == '__main__':
    main()
