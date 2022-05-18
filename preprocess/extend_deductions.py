import collections
import json
from typing import Dict
from nltk import sent_tokenize
from collections import defaultdict
import argparse
from tqdm import tqdm


def is_negation(sent):
    return any(word in sent for word in ["no", "not", "n't", "neither", "nor"])


def dfs(part_1, part_2, assigned_table, constraints):
    ...


def extend_item(item: Dict):
    context = item["context"]
    sentences = sent_tokenize(context) + [item["question"]]

    participants = item["rows"]
    positions = item["columns"]

    assigned_table = {}
    for participant in participants:
        for position in positions:
            for sent_id, sent in enumerate(sentences):
                if sent_id == 0:
                    continue
                if participant in sent and position in sent:
                    if participant not in assigned_table:
                        assigned_table[participant] = defaultdict(list)
                    assigned_table[participant][position].append((sent_id, is_negation(sent)))
                    # if is_negation(sent):
                    #     assigned_table[participant] = (position, sent_id, False)
                    # else:
                    #     assigned_table[participant] = (position, sent_id, True)

    constraints = {}
    for part_id, part_1 in enumerate(participants):
        for part_2 in participants[part_id + 1:]:
            for sent_id, sent in enumerate(sentences):
                if sent_id == 0:
                    continue
                if part_1 in sent and part_2 in sent:
                    if part_1 not in constraints:
                        constraints[part_1] = {}
                    if part_2 not in constraints[part_1]:
                        constraints[part_1][part_2] = []
                    constraints[part_1][part_2].append(sent_id)

                    if part_2 not in constraints:
                        constraints[part_2] = {}
                    if part_1 not in constraints[part_2]:
                        constraints[part_2][part_1] = []
                    constraints[part_2][part_1].append(sent_id)

    extended_assignments = []
    queue = []
    vis = set()
    for part_1 in participants:
        for position in positions:
            if part_1 in assigned_table and position in assigned_table[part_1]:
                for sent_id, flag in assigned_table[part_1][position]:
                    if not flag:
                        queue.append((part_1, position, sent_id))
                        vis.add(f"{part_1}${position}")
    while queue:
        part_1, pos, sent_id_1 = queue.pop()
        if part_1 not in constraints:
            continue
        for part_2 in constraints[part_1]:
            if f"{part_2}${pos}" in vis:
                continue
            for sent_id_2 in constraints[part_1][part_2]:
                if part_2 not in assigned_table or pos not in assigned_table[part_2] and sent_id_2 != sent_id_1:
                    extended_assignments.append({
                        "edge": (part_2, pos),
                        "bridge": [
                            [part_1, pos, sent_id_1], [part_1, part_2, sent_id_2]
                        ]
                    })
                    queue.append((part_2, pos, sent_id_2))
                    vis.add(f"{part_2}${pos}")

    item["initial_assignments"] = assigned_table
    item["constraints"] = constraints
    item["extended_assignments"] = extended_assignments
    item["sentences"] = sentences
    return item


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, default=None)

    args = parser.parse_args()

    data = json.load(open(args.input_file))

    outputs = list(map(extend_item, data))

    if args.output_file:
        json.dump(outputs, open(args.output_file, "w"), indent=2)
    else:
        output_file = args.input_file.replace(".json", "_extended.json")
        json.dump(outputs, open(output_file, "w"), indent=2)


if __name__ == '__main__':
    main()
