import glob
import json
import os
import sys
from argparse import ArgumentParser
import numpy as np

import torch
from tqdm import tqdm

sys.path.append("../../")

from modules.sim_cse import ModifiedSimCSE


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--corpus_file", type=str)
    parser.add_argument("--corpus_index_file", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--query_batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int)
    parser.add_argument("--cpu", default=False, action="store_true")
    parser.add_argument("--shard", default=False, action="store_true")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # Load SimCSE
    model = ModifiedSimCSE("princeton-nlp/sup-simcse-roberta-large", device="cpu" if args.cpu else None, shard=args.shard)

    # Load corpus file
    if os.path.exists(args.corpus_file):
        corpus_files = [args.corpus_file]
    else:
        corpus_files = list(glob.glob(args.corpus_file))
        corpus_files = sorted(corpus_files)

    corpus_sentences = []
    sent2full_id = {}
    for _file_id, _file in enumerate(corpus_files):
        samples = json.load(open(_file))
        print(f"Loading corpus from {_file}")
        for sample_id, sample in enumerate(samples):
            for sent_id, sent in enumerate(sample["sents"]):
                sent = " ".join(sent)
                corpus_sentences.append(sent)
                sent2full_id[sent] = (_file_id, sample_id, sent_id)

    corpus_index = torch.load(args.corpus_index_file, map_location="cpu").numpy()
    model.build_index(corpus_sentences, embeddings=corpus_index)

    # Load query file
    if os.path.exists(args.input_file):
        input_files = [args.input_file]
    else:
        input_files = list(glob.glob(args.input_file))
        input_files = sorted(input_files)

    for _file_id, _file in enumerate(input_files):
        sentences = json.load(open(_file))
        all_sentences = []
        all_full_ids = []
        for doc_id, doc in enumerate(sentences):
            all_sentences.extend([" ".join(sent) for sent in doc])
            all_full_ids.extend([(doc_id, sent_id) for sent_id in range(len(doc))])

        sent_emb_file = _file[:-5] + "_emb.pt"
        sent_query_vecs = torch.load(sent_emb_file, map_location="cpu").numpy()
        assert len(all_sentences) == sent_query_vecs.shape[0]

        # tmp_results = model.search(all_sentences, sent_query_vecs, top_k=10)
        tmp_results = []
        bsz = args.query_batch_size
        total_batch = len(all_sentences) // bsz + (1 if len(sentences) % bsz > 0 else 0)
        for batch_id in tqdm(range(total_batch), dynamic_ncols=True):
            _input_sentences = all_sentences[(batch_id * bsz): ((batch_id + 1) * bsz)]
            _input_vecs = sent_query_vecs[(batch_id * bsz): ((batch_id + 1) * bsz)]
            tmp_results.extend(model.search(_input_sentences, _input_vecs, top_k=10))

        search_results = []
        for full_id, sample_res_ls in zip(all_full_ids, tmp_results):
            tmp = []
            for res in sample_res_ls:
                query_full_id = sent2full_id[res[0]]
                if query_full_id[0] == _file_id and query_full_id[1] == full_id[0] and query_full_id[2] == full_id[1]:
                    continue
                tmp.append((res[0], float(res[1]), query_full_id))
                break
            if len(tmp):
                search_results.append(tmp[0])
            else:
                search_results.append(None)

        outputs = list(zip(all_sentences, search_results))
        output_file = _file[:-5] + "_simcse_rank_v2.json"
        json.dump(outputs, open(output_file, "w"))


if __name__ == '__main__':
    main()
