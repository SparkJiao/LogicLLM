import argparse
import torch
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str)
    parser.add_argument("--val_file", type=str)
    parser.add_argument("--output_file", type=str)
    args = parser.parse_args()

    train_edge_hidden, train_cluster_ids_x, cluster_centers = torch.load(args.train_file, map_location="cpu")
    val_edge_hidden, val_cluster_ids_x, _ = torch.load(args.val_file, map_location="cpu")

    train_cluster_ids_x = train_cluster_ids_x.tolist()
    for (val_edge, edge_hidden), val_edge_label in tqdm(zip(val_edge_hidden.items(), val_cluster_ids_x), total=len(val_cluster_ids_x)):
        if val_edge not in train_edge_hidden:
            _lens = len(train_edge_hidden)

            train_edge_hidden[val_edge] = edge_hidden
            train_cluster_ids_x.append(val_edge_label)
            # train_cluster_ids_x = torch.cat([train_cluster_ids_x, val_edge_label], dim=0)

            # assert (train_edge_hidden[list(train_edge_hidden.keys())[_lens]] == edge_hidden).sum() == 1536

    train_cluster_ids_x = torch.tensor(train_cluster_ids_x, dtype=torch.int)
    torch.save((train_edge_hidden, train_cluster_ids_x, cluster_centers), args.output_file)


if __name__ == '__main__':
    main()
