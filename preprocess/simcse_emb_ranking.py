import glob
import json
import os
import sys
from argparse import ArgumentParser

import torch
from tqdm import tqdm
import numpy as np

sys.path.append("../")

from modules.sim_cse import ModifiedSimCSE


def main():
    parser = ArgumentParser()
    parser.add_argument("--q_index_path", type=str)
    parser.add_argument("--k_index_path", type=str)
    parser.add_argument("--num_workers", type=int)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--top_k", type=int)
    args = parser.parse_args()

    model = ModifiedSimCSE("princeton-nlp/sup-simcse-roberta-large")

    q_memory = torch.load(args.q_index_path)
    k_memory = torch.load(args.k_index_path)
    q_index = q_memory["hidden_states"]
    k_index = k_memory["hidden_states"]
    q_ids = q_memory["indices"]
    k_ids = k_memory["indices"]

    print("Building index...")
    model.build_index(sentences_or_file_path=list(map(str, k_ids)), use_faiss=True, device="cuda", embeddings=k_index.float().numpy())
    print("Index built")

    results = {}
    for q_id, q_vec in tqdm(zip(q_ids, q_index), total=len(q_ids)):
        res = model.search("", q_vec.unsqueeze(0).float().numpy(), device="cuda", top_k=args.top_k)
        results[q_id] = res

    torch.save(results, args.output_file)


if __name__ == '__main__':
    main()
