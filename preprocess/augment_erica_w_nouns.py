import argparse
import collections
import copy
import glob
import json
import os
from multiprocessing import Pool
from typing import Dict, Any

import spacy
from tqdm import tqdm

nlp = None


def initializer():
    global nlp
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'textcat'])


def process_item(item: Dict[str, Any]):
    noun_dict = collections.defaultdict(list)

    # Some types that can be considered:
    #  noun, phrase, verb, ...
    collected_pos_list = ['NOUN']

    sents = item["sents"]
    for sent_id, sent in enumerate(sents):
        sent_str = " ".join(sent)

        collected_spans = []

        doc = nlp(sent_str)
        for token in doc:
            if token.pos_ in collected_pos_list:
                res = {
                    "pos": [],
                    "type": "noun",
                    "sent_id": sent_id,
                    "name": token.lemma_.lower(),
                    "id": f"N#{token.lemma_.lower()}",
                }
                collected_spans.append((token.text, res))

        word_vis = [True] * len(sent)
        for ent in item["vertexSet"]:
            for mention in ent:
                if mention["sent_id"] == sent_id:
                    for p in range(mention["pos"][0], mention["pos"][1]):
                        word_vis[p] = False

        span_vis = [True] * len(collected_spans)
        for word_id, word in enumerate(sent):
            for span_id, span in enumerate(collected_spans):
                if span[0] == word and span_vis[span_id] and word_vis[word_id]:
                    span[1]["pos"] = [word_id, word_id + 1]
                    noun_dict[span[1]["id"]].append(copy.deepcopy(span[1]))
                    span_vis[span_id] = False
                    word_vis[word_id] = False
                    break

    item["vertexSet"].extend(list(noun_dict.values()))
    return item


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--num_workers", type=int, default=48)
    args = parser.parse_args()

    if os.path.exists(args.input_path):
        input_files = [args.input_path]
    else:
        input_files = glob.glob(args.input_path)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    for file in input_files:
        file_name = file.split("/")[-1]
        output_file_name = os.path.join(args.output_dir, file_name)

        samples = json.load(open(file))

        with Pool(args.num_workers, initializer=initializer) as p:
            processed_samples = list(tqdm(
                p.imap(process_item, samples, chunksize=32),
                total=len(samples),
                desc='processing samples'
            ))

        json.dump(processed_samples, open(output_file_name, "w"))


if __name__ == '__main__':
    main()
