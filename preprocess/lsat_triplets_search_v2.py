import argparse
import json
from collections import defaultdict
from typing import Dict

from nltk import sent_tokenize


def find_triplets_ctx(item: Dict):
    context = item["context"]
    # sentences = sent_tokenize(context) + [item["question"], item["answers"][item["label"]]]
    sentences = sent_tokenize(context)

    participants = item["rows"]
    positions = item["columns"]

    if not participants or not positions:
        return item

    part2sent = defaultdict(list)
    pos2sent = defaultdict(list)
    for sent_id, sent in enumerate(sentences):
        if sent_id == 0:
            continue
        if sent_id <= 2 and sum([1 for p in participants if p in sent]) > (len(participants) // 2):
            continue
        for part in participants:
            if part in sent:
                part2sent[part].append(sent_id)
        for pos in positions:
            if pos in sent:
                pos2sent[pos].append(sent_id)

    triplets = []
    for part_id, part_1 in enumerate(participants):
        for pos in positions:
            common_s = set(part2sent[part_1]) & set(pos2sent[pos])
            if len(common_s) == 0:
                continue
            for part_2 in participants[part_id + 1:]:
                part2part_commons_s = set(part2sent[part_2]) & set(part2sent[part_1])
                if len(part2part_commons_s) == 0:
                    continue
                part2pos_common_s = set(part2sent[part_2]) & set(pos2sent[pos])
                if len(part2pos_common_s) == 0:
                    continue
                for sent_1 in common_s:
                    for sent_2 in part2part_commons_s:
                        if sent_2 == sent_1:
                            continue
                        for sent_3 in part2pos_common_s:
                            if sent_3 == sent_1 or sent_3 == sent_2:
                                continue
                            triplets.append((sent_1, sent_2, sent_3, part_1, part_2, pos))

    item["triplets"] = triplets
    item["sentences"] = sentences
    return item


def find_triplets_ctx_q(item: Dict):
    context = item["context"]
    # sentences = sent_tokenize(context) + [item["question"], item["answers"][item["label"]]]
    # sentences = sent_tokenize(context)
    sentences = sent_tokenize(context) + [item["question"]]

    participants = item["rows"]
    positions = item["columns"]

    if not participants or not positions:
        return item

    part2sent = defaultdict(list)
    pos2sent = defaultdict(list)
    for sent_id, sent in enumerate(sentences):
        if sent_id == 0:
            continue
        if sent_id <= 2 and sum([1 for p in participants if p in sent]) > (len(participants) // 2):
            continue
        for part in participants:
            if part in sent:
                part2sent[part].append(sent_id)
        for pos in positions:
            if pos in sent:
                pos2sent[pos].append(sent_id)

    triples = []
    for part_id, part_1 in enumerate(participants):
        for pos in positions:
            common_s = set(part2sent[part_1]) & set(pos2sent[pos])
            if len(common_s) == 0:
                continue
            for part_2 in participants[part_id + 1:]:
                part2part_commons_s = set(part2sent[part_2]) & set(part2sent[part_1])
                if len(part2part_commons_s) == 0:
                    continue
                part2pos_common_s = set(part2sent[part_2]) & set(pos2sent[pos])
                if len(part2pos_common_s) == 0:
                    continue
                for sent_1 in common_s:
                    for sent_2 in part2part_commons_s:
                        if sent_2 == sent_1:
                            continue
                        for sent_3 in part2pos_common_s:
                            if sent_3 == sent_1 or sent_3 == sent_2:
                                continue
                            triples.append((sent_1, sent_2, sent_3, part_1, part_2, pos))

    item["triplets"] = triples
    item["sentences"] = sentences
    return item


def find_triplets_ctx_q_a(item: Dict):
    context = item["context"]
    sentences = sent_tokenize(context) + [item["question"], item["answers"][item["label"]]]
    # sentences = sent_tokenize(context)
    # sentences = sent_tokenize(context) + [item["question"]]

    participants = item["rows"]
    positions = item["columns"]

    if not participants or not positions:
        return item

    part2sent = defaultdict(list)
    pos2sent = defaultdict(list)
    for sent_id, sent in enumerate(sentences):
        if sent_id == 0:
            continue
        if sent_id <= 2 and sum([1 for p in participants if p in sent]) > (len(participants) // 2):
            continue
        for part in participants:
            if part in sent:
                part2sent[part].append(sent_id)
        for pos in positions:
            if pos in sent:
                pos2sent[pos].append(sent_id)

    triples = []
    for part_id, part_1 in enumerate(participants):
        for pos in positions:
            common_s = set(part2sent[part_1]) & set(pos2sent[pos])
            if len(common_s) == 0:
                continue
            for part_2 in participants[part_id + 1:]:
                part2part_commons_s = set(part2sent[part_2]) & set(part2sent[part_1])
                if len(part2part_commons_s) == 0:
                    continue
                part2pos_common_s = set(part2sent[part_2]) & set(pos2sent[pos])
                if len(part2pos_common_s) == 0:
                    continue
                for sent_1 in common_s:
                    for sent_2 in part2part_commons_s:
                        if sent_2 == sent_1:
                            continue
                        for sent_3 in part2pos_common_s:
                            if sent_3 == sent_1 or sent_3 == sent_2:
                                continue
                            triples.append((sent_1, sent_2, sent_3, part_1, part_2, pos))

    item["triplets"] = triples
    item["sentences"] = sentences
    return item


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--action", type=int, default=0)

    args = parser.parse_args()

    data = json.load(open(args.input_file, "r"))
    if args.action == 0:
        outputs = list(map(find_triplets_ctx, data))
    elif args.action == 1:
        outputs = list(map(find_triplets_ctx_q, data))
    elif args.action == 2:
        outputs = list(map(find_triplets_ctx_q_a, data))
    else:
        raise NotImplementedError

    cnt = 0
    tot = 0
    for item in outputs:
        if "triplets" in item:
            if not item["triplets"]:
                item.pop("triplets")
                continue
            cnt += 1
            tot += len(item["triplets"])
    print(f"{cnt}/{len(outputs)}")
    print(f"Total triplets: {tot}")

    if args.output_file:
        json.dump(outputs, open(args.output_file, "w"), indent=2)
    else:
        output_file = args.input_file.replace(".json", f"_triplets_v2_type_{args.action}.json")
        json.dump(outputs, open(output_file, "w"), indent=2)


if __name__ == '__main__':
    main()
