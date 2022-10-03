import glob
import json
import os
import sys
from argparse import ArgumentParser

import torch
from tqdm import tqdm

sys.path.append("../")

from modules.sim_cse import ModifiedSimCSE


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--num_workers", type=int)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # Load corpus
    if os.path.exists(args.input_file):
        input_files = [args.input_file]
    else:
        input_files = list(glob.glob(args.input_file))
        input_files = sorted(input_files)

    all_sentences = []
    sent2full_id = {}
    for _file_id, _file in enumerate(input_files):
        samples = json.load(open(_file))
        for sample_id, sample in enumerate(tqdm(samples, desc=f"Extracting sentences from {_file}", total=len(samples))):
            # all_sentences.extend([
            #     (sent, _file_id, sample_id, sent_id) for sent_id, sent in enumerate(sample["sents"])
            # ])
            for sent_id, sent in enumerate(sample["sents"]):
                sent = " ".join(sent)
                all_sentences.append(sent)
                sent2full_id[sent] = (_file_id, sample_id, sent_id)

    # all_sentences = all_sentences[:500000]

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Load SimCSE
    model = ModifiedSimCSE("princeton-nlp/sup-simcse-roberta-large")

    embedding_path = os.path.join(args.output_dir, "embeddings.pt")
    if os.path.exists(embedding_path):
        embeddings = torch.load(embedding_path, map_location="cpu").numpy()
    else:
        print("Encoding...")
        embeddings = model.encode(all_sentences, batch_size=128, normalize_to_unit=True, return_numpy=True, save_path=embedding_path)

    print("Building index...")
    model.build_index(all_sentences, embeddings=embeddings)
    print("Index built.")

    for _file_id, _file in enumerate(input_files):
        samples = json.load(open(_file))
        for sample_id, sample in enumerate(tqdm(samples, desc="Searching over corpus", total=len(samples))):
            sent_sim_results = []
            for sent_id, sent in enumerate(sample["sents"]):
                sent = " ".join(sent)
                tmp_results = model.search(sent, top_k=10)
                results = []
                for res in tmp_results:
                    full_id = sent2full_id[res[0]]
                    if full_id[1] != sample_id:
                        results.append((res[0], float(res[1]), full_id))
                        break
                if len(results):
                    sent_sim_results.append(results[0])
                else:
                    sent_sim_results.append(None)
            sample["sent_sim_results"] = sent_sim_results
        output_file = os.path.join(args.output_dir, _file.split("/")[-1])
        json.dump(samples, open(output_file, "w"))


if __name__ == '__main__':
    main()
