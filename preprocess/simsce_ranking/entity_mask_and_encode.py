import glob
import json
import os
import sys
from argparse import ArgumentParser
from functools import partial
from multiprocessing import Pool

from tqdm import tqdm

sys.path.append("../../")

from modules.sim_cse import ModifiedSimCSE


def sentence_entity_mask(sample, mask_token="<mask>"):
    sentences = sample["sents"]
    entities = sample["vertexSet"]

    sent2ent = [[] for _ in range(len(sentences))]
    for ent in entities:
        for ent_pos in ent:
            sent2ent[ent_pos["sent_id"]].append(ent_pos)

    masked_sent_tokens = []
    for sent_id, sent_ent_ls in enumerate(sent2ent):
        sent_ents = sorted(sent_ent_ls, key=lambda x: x["pos"][0])

        # Non-overlapping check
        for _tmp_id, _tmp in enumerate(sent_ents):
            if _tmp_id == 0:
                continue
            assert _tmp["pos"][0] >= sent_ents[_tmp_id - 1]["pos"][1]

        tokens = []
        last_e = 0
        for ent_pos in sent_ents:
            s, e = ent_pos["pos"]
            tokens.extend(sentences[sent_id][last_e: s])
            tokens.append(mask_token)
            last_e = e

        tokens.extend(sentences[sent_id][last_e:])
        masked_sent_tokens.append(tokens)

    return masked_sent_tokens


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--option", "-o", type=str, default="mask")
    parser.add_argument("--num_workers", type=int)
    parser.add_argument("--output_dir", type=str)

    return parser.parse_args()


def main():
    args = parse_args()

    # Load corpus
    if os.path.exists(args.input_file):
        input_files = [args.input_file]
    else:
        input_files = list(glob.glob(args.input_file))
        input_files = sorted(input_files)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    model = ModifiedSimCSE("princeton-nlp/sup-simcse-roberta-large")

    for _file_id, _file in enumerate(input_files):
        print(f"Processing input file {_file}")
        samples = json.load(open(_file))
        file_name = _file.split("/")[-1]
        with Pool(args.num_workers) as p:
            _annotate = partial(sentence_entity_mask, mask_token="<mask>")
            masked_sents = list(tqdm(
                p.imap(_annotate, samples, chunksize=32),
                total=len(samples),
                dynamic_ncols=True,
                desc="masking sentences"
            ))

        sent_output_file = os.path.join(args.output_dir, file_name[:-5] + "_mask.json")
        json.dump(masked_sents, open(sent_output_file, "w"))

        query_embedding_path = os.path.join(args.output_dir, file_name[:-5] + "_mask_emb.pt")
        all_sentences = []
        for doc in masked_sents:
            all_sentences.extend([" ".join(sent) for sent in doc])
        model.encode(all_sentences, batch_size=128, normalize_to_unit=True, return_numpy=True, save_path=query_embedding_path)

        print("Done.")


if __name__ == '__main__':
    main()
