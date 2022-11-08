import argparse
import os
from collections import defaultdict
from glob import glob

import torch
from tqdm import tqdm


def read_entity_hidden(file_path, device: torch.device):
    if os.path.exists(file_path):
        files = [file_path]
    else:
        files = list(glob(file_path))
        files = sorted(files)
        print(files)

    ent2hidden_list = defaultdict(list)
    for file in files:
        print(file)
        data = torch.load(file, map_location="cpu")
        mentions = data["mentions"]
        hidden = data["hidden"]
        index = data["index"]

        for item in hidden:
            for ent_id, ent_hidden in item.items():
                ent2hidden_list[ent_id].append(ent_hidden)
    print(len(ent2hidden_list))

    ent2hidden = {}
    for ent_id, ent_hidden_ls in tqdm(ent2hidden_list.items(), total=len(ent2hidden_list)):
        ent_hidden = torch.stack(ent_hidden_ls, dim=0)
        ent_hidden = ent_hidden.to(device)
        ent_hidden = torch.mean(ent_hidden, dim=0).cpu()
        ent2hidden[ent_id] = ent_hidden

    return ent2hidden


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ent_hidden_file_path", type=str)
    parser.add_argument("--gpu_id", type=int)
    parser.add_argument("--output_file", type=str)
    args = parser.parse_args()

    ent2hidden = read_entity_hidden(args.ent_hidden_file_path, torch.device(f"cuda:{args.gpu_id}"))
    torch.save(ent2hidden, args.output_file)


if __name__ == '__main__':
    main()
