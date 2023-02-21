import glob
import json
import os
import sys
from argparse import ArgumentParser
from transformers import AutoTokenizer, PreTrainedTokenizer

import torch
from tqdm import tqdm
import numpy as np

sys.path.append("../")

from modules.sim_cse import ModifiedSimCSE


def load_memory(memory_path: str, tokenizer: str):
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(tokenizer)
    all_examples, raw_texts = torch.load(memory_path, map_location="cpu")
    print(f"Loading raw texts from memory.")
    memory = []
    for exp in tqdm(all_examples, total=len(all_examples)):
        token_ids = tokenizer.convert_tokens_to_ids(exp["tokens"][0])
        text = tokenizer.decode(token_ids, skip_special_tokens=True)
        memory.append(text)
    return memory


def read_reclor_data(file_path):
    data = json.load(open(file_path, 'r'))

    all_context = []
    all_question = []
    all_option_list = []
    all_label = []
    for sample in data:
        all_context.append(sample["context"])
        all_question.append(sample["question"])
        if "label" not in sample:
            all_label.append(-1)
        else:
            all_label.append(sample["label"])
        all_option_list.append(sample["answers"])

    outputs = []
    for ctx, q in zip(all_context, all_question):
        outputs.append(ctx + ' ' + q)
    return outputs


def main():
    parser = ArgumentParser()
    parser.add_argument("--q_input_file", type=str)
    parser.add_argument("--k_input_file", type=str)
    parser.add_argument("--k_memory_path", type=str)
    parser.add_argument("--k_tokenizer", type=str)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--top_k", type=int)
    args = parser.parse_args()

    model = ModifiedSimCSE("princeton-nlp/sup-simcse-roberta-large")

    if os.path.exists(args.k_memory_path):
        embedding = torch.load(args.k_memory_path)
    else:
        embedding = None

    memory = load_memory(args.k_input_file, args.k_tokenizer)
    queries = read_reclor_data(args.q_input_file)

    if embedding is None:
        print("Encoding")
        embedding = model.encode(memory, batch_size=128, normalize_to_unit=True, device="cuda", return_numpy=True,
                                 save_path=args.k_memory_path)

    print("Building index...")
    model.build_index(memory, embeddings=embedding.numpy(), use_faiss=True, device="cuda")
    print("Index built.")

    results = {}
    no_sample = 0
    for q_id, query in tqdm(enumerate(queries), total=len(queries)):
        res = model.search(query, top_k=args.top_k, device="cuda", threshold=0.0)
        results[q_id] = res
        if len(res) == 0:
            no_sample += 1

    print(no_sample)

    torch.save(results, args.output_file)


if __name__ == '__main__':
    main()
