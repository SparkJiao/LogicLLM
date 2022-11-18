import glob
import json
import os
import pickle
import random
from argparse import ArgumentParser
from functools import partial
from multiprocessing import Pool

from tqdm import tqdm


def annotate_entity(sent, mention1, mention2, h_s_sep: str, h_e_sep: str, t_s_sep: str, t_e_sep: str):
    if mention1["pos"][0] > mention2["pos"][0]:
        mention2, mention1 = mention1, mention2
    assert mention1["pos"][1] <= mention2["pos"][0]

    text = ""
    prefix = sent[:mention1["pos"][0]]
    if len(prefix):
        text = text + " ".join(prefix)

    text = text + " " + " ".join([h_s_sep] + sent[mention1["pos"][0]: mention1["pos"][1]] + [h_e_sep])

    inter = sent[mention1["pos"][1]: mention2["pos"][0]]
    if len(inter):
        text = text + " " + " ".join(inter)

    text = text + " " + " ".join([t_s_sep] + sent[mention2["pos"][0]: mention2["pos"][1]] + [t_e_sep])

    text = text + " " + " ".join(sent[mention2["pos"][1]:])
    return text


def process_single_para(sample, h_s_sep: str, h_e_sep: str, t_s_sep: str, t_e_sep: str):
    entities = sample["vertexSet"]
    sentences = sample["sents"]

    results = []
    for idx, ent1 in enumerate(entities):
        for ent2 in entities[(idx + 1):]:
            if ent1[0]["id"] == ent2[0]["id"]:
                continue
            for mention1 in ent1:
                for mention2 in ent2:
                    if mention1["sent_id"] == mention2["sent_id"]:
                        results.append(annotate_entity(sentences[mention1["sent_id"]], mention1, mention2,
                                                       h_s_sep, h_e_sep, t_s_sep, t_e_sep))
    return results


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='')
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--dev_ratio', type=float, default=0.01)

    args = parser.parse_args()

    if os.path.exists(args.input_file):
        input_files = [args.input_file]
    else:
        input_files = list(glob.glob(args.input_file))

    file_suffix = f'.sent_entity_annotate_v1.0.pkl'
    if args.output_dir and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    examples = []
    for _file in input_files:
        samples = json.load(open(_file))

        with Pool(args.num_workers) as p:
            _annotate = partial(process_single_para)
            _results = list(tqdm(
                p.imap(_annotate, samples, chunksize=32),
                total=len(samples),
                desc='processing samples'
            ))
        for _res in _results:
            examples.extend(_res)

        print(len(examples))

    indices = list(range(len(examples)))
    num_dev = int(len(indices) * args.dev_ratio)
    dev_ids = set(random.sample(indices, num_dev))
    train = []
    dev = []
    for idx, exp in enumerate(examples):
        if idx in dev_ids:
            dev.append(exp)
        else:
            train.append(exp)
    print(len(train))
    print(len(dev))

    pickle.dump(train, open(os.path.join(args.output_dir, "wiki_sent_ent_ann.train" + file_suffix), "wb"))
    pickle.dump(dev, open(os.path.join(args.output_dir, "wiki_sent_ent_ann.dev" + file_suffix), "wb"))
