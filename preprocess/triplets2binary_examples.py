import argparse
import json
import random
from collections import defaultdict
from typing import List, Dict, Tuple


def replace_entity(sentence: str, entity: str, replacement: str):
    return sentence.replace(entity, replacement)


def triplet2binary_example(triplet: Tuple[int, int, int, str, str, str],
                           sentences: List[str],
                           positions: List[str]):
    sent_id_1, sent_id_2, sent_id_3, part_1, part_2, pos = triplet

    sent_id_ls = [sent_id_1, sent_id_2, sent_id_3]
    ent_ls = [part_1, part_2, pos]
    sent2ent = defaultdict(list)
    for ent in ent_ls:
        for sent_id in sent_id_ls:
            if ent in sentences[sent_id]:
                sent2ent[sent_id].append(ent)

    # assert all(len(sent2ent[sent_id]) == 2 for sent_id in sent_id_ls)
    assert all(len(sent2ent[sent_id]) >= 2 for sent_id in sent_id_ls)

    examples = []
    for sent_id in sent_id_ls:
        if pos not in sentences[sent_id]:
            continue

        rest_sentences = sentences[:sent_id] + sentences[sent_id + 1:]

        neg_pos = random.choice(positions)
        while neg_pos == pos:
            neg_pos = random.choice(positions)

        neg = replace_entity(sentences[sent_id], pos, neg_pos)

        examples.append({
            "ori_sent": sentences[sent_id],
            "neg_sent": neg,
            "ori_sent_id": sent_id,
            "ori_pos": pos,
            "neg_pos": neg_pos,
            "rest_sentences": rest_sentences
        })
    return examples


def triplets2binary_examples(item: Dict):
    if "triplets" not in item:
        return item

    sentences = item["sentences"]
    triplets = item["triplets"]
    participants = item["rows"]
    positions = item["columns"]

    examples = []
    for triplet in triplets:
        examples.extend(triplet2binary_example(triplet, sentences, positions))

    if len(examples):
        item["examples"] = examples
    return item


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, default=None)
    args = parser.parse_args()

    with open(args.input_file, "r") as f:
        data = json.load(f)

    outputs = list(map(triplets2binary_examples, data))

    if args.output_file is not None:
        with open(args.output_file, "w") as f:
            json.dump(outputs, f, indent=2)
    else:
        output_file = args.input_file.replace(".json", "_examples.json")
        with open(output_file, "w") as f:
            json.dump(outputs, f, indent=2)


if __name__ == '__main__':
    main()
