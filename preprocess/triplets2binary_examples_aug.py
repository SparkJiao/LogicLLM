import argparse
import copy
import json
import random
from collections import defaultdict
from typing import List, Dict, Tuple
from nltk import word_tokenize

entity_pool: List[List[str]]


def is_alphabet(char):
    res = ord('a') <= ord(char) <= ord('z') or ord('A') <= ord(char) <= ord('Z')
    return res


def replace_entity(sentence: str, entity: str, replacement: str):
    if len(entity) == 1:  # Process single character entity uniquely.
        s = sentence.find(entity)
        while s != -1:
            a = not is_alphabet(sentence[s - 1]) if s > 0 else 'jump'
            b = not is_alphabet(sentence[s + 1]) if s + 1 < len(sentence) else 'jump'
            # print(f"DEBUG: {s} [{sentence[s]}] {a} {b}")
            # if a and a != 'jump':
            #     print(f"DEBUG: {sentence[s - 1]}")
            # if b and b != 'jump':
            #     print(f"DEBUG: {sentence[s + 1]}")
            if a and b:
                # print(a, b)
                sentence = sentence[:s] + replacement + sentence[s + 1:]
                s = sentence.find(entity, s + len(replacement))
            else:
                s = sentence.find(entity, s + 1)
        return sentence

    assert replacement.strip() != ""
    return sentence.replace(entity, replacement)


def sample_entities(participants: List[str]):
    num = len(participants)
    entities = random.choice(entity_pool)
    cnt = 0
    while len(entities) < num or set(entities) & set(participants):
        entities = random.choice(entity_pool)
        cnt += 1
        if cnt > 30:
            break
    return entities


def replace_entity_set(sentences: List[str], entity_set: List[str], replacements: List[str]):
    new_sentences = []

    replacements_copy = copy.deepcopy(replacements)
    random.shuffle(replacements_copy)
    assert len(entity_set) <= len(replacements_copy)
    replacements_copy = replacements_copy[:len(entity_set)]

    ent_mapping = [(ent, rep) for ent, rep in zip(entity_set, replacements_copy)]
    # print(ent_mapping)

    sentences = copy.deepcopy(sentences)

    for sent in sentences:
        # print(f"+++++++++++ {sent}")
        src_sent = copy.deepcopy(sent)
        for src_ent, dst_ent in ent_mapping:
            # print(f"====== {src_ent} to {dst_ent} =========")
            src_sent = replace_entity(src_sent, src_ent, dst_ent)
            # print(f"|||||||||||| {src_sent}")
        new_sentences.append(src_sent)

    return new_sentences, {ent: rep for ent, rep in ent_mapping}


def triplet2binary_example(triplet: Tuple[int, int, int, str, str, str],
                           sentences: List[str],
                           participants: List[str],
                           positions: List[str],
                           neg_num: int = 1,
                           aug_num: int = 0,
                           shuffle: bool = False) -> List[Dict]:
    assert neg_num >= 1

    sent_id_1, sent_id_2, sent_id_3, part_1, part_2, pos = triplet

    sent_id_ls = [sent_id_1, sent_id_2, sent_id_3]
    ent_ls = [part_1, part_2, pos]
    sent2ent = defaultdict(list)
    for ent in ent_ls:
        for sent_id in sent_id_ls:
            if ent in sentences[sent_id]:
                sent2ent[sent_id].append(ent)

    # print(sentences)
    # print(triplet)
    # print(participants)
    # print(positions)
    # print(sent2ent)
    # print("+++++++++++++++++")

    # assert all(len(sent2ent[sent_id]) == 2 for sent_id in sent_id_ls)
    # assert all(len(sent2ent[sent_id]) >= 2 for sent_id in sent_id_ls)
    for sent_id in sent_id_ls:
        assert len(sent2ent[sent_id]) >= 2, (sentences[sent_id], sent2ent[sent_id])

    examples = []
    for sent_id in sent_id_ls:
        assert sent_id > 1, (sent_id, sentences[sent_id], sentences[sent_id - 1], participants, positions)

        if pos not in sentences[sent_id]:
            ent_set = participants
        else:
            ent_set = positions

        rest_sentences = sentences[:sent_id] + sentences[sent_id + 1:]

        neg_sent_ls = []
        neg_pos_ls = []
        for _ in range(neg_num):
            neg_pos = random.choice(ent_set)
            while neg_pos == pos:
                neg_pos = random.choice(ent_set)

            neg = replace_entity(sentences[sent_id], pos, neg_pos)

            neg_sent_ls.append(neg)
            neg_pos_ls.append(neg_pos)

        examples.append({
            "ori_sent": sentences[sent_id],
            # "neg_sent": neg,
            "neg_sent": neg_sent_ls,
            "ori_sent_id": sent_id,
            "ori_pos": pos,
            # "neg_pos": neg_pos,
            "neg_pos": neg_pos_ls,
            "rest_sentences": rest_sentences
        })

    for _ in range(aug_num):
        rep_part_set = sample_entities(participants)
        if len(rep_part_set) < len(participants):
            break

        new_sentences, ent_mapping = replace_entity_set(sentences, participants, rep_part_set)
        mapped_triplet = (sent_id_1, sent_id_2, sent_id_3, ent_mapping[part_1], ent_mapping[part_2], pos)
        aug_examples = triplet2binary_example(mapped_triplet, new_sentences, list(ent_mapping.values()), positions, neg_num, 0)

        if shuffle:
            for aug_ex in aug_examples:
                aug_sentences = aug_ex["rest_sentences"]
                suffix = copy.deepcopy(aug_sentences[2:])
                random.shuffle(suffix)
                aug_ex["rest_sentences"] = aug_sentences[:2] + suffix
                aug_ex["ent_mapping"] = ent_mapping
                examples.append(aug_ex)
        else:
            examples.extend(aug_examples)

    return examples


def triplets2binary_examples(item: Dict, neg_num: int = 1, aug_num: int = 0, shuffle: bool = False) -> Dict:
    if "triplets" not in item:
        return item

    sentences = item["sentences"]
    triplets = item["triplets"]
    participants = item["rows"]
    positions = item["columns"]

    if len(participants) > 20 or len(positions) > 20:
        return item
    if set(participants) & set(positions):
        return item
    for tmp in participants + positions:
        if len(tmp.split()) > 1:
            print("************** participants or positions contain multi-words:", tmp)
            return item

    sentences = [' '.join(word_tokenize(sent)) for sent in sentences]
    item["sentences"] = sentences

    examples = []
    for triplet in triplets:
        examples.extend(triplet2binary_example(triplet, sentences, participants, positions, neg_num, aug_num, shuffle))

    if len(examples):
        item["examples"] = examples
    return item


def initializer(data):
    global entity_pool
    entity_pool = [item["rows"] for item in data]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--neg_num", type=int, default=1)
    parser.add_argument("--aug_num", type=int, default=0)
    parser.add_argument("--shuffle", default=False, action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    with open(args.input_file, "r") as f:
        data = json.load(f)

    initializer(data)

    outputs = []
    for item in data:
        outputs.append(triplets2binary_examples(item, args.neg_num, args.aug_num, args.shuffle))

    suffix = f"_neg{args.neg_num}_aug{args.aug_num}{'_shuffle' if args.shuffle else ''}_examples_{args.seed}.json"

    output_file = args.input_file.replace(".json", suffix)
    with open(output_file, "w") as f:
        json.dump(outputs, f, indent=2)

    print("Done!")


if __name__ == '__main__':
    main()
