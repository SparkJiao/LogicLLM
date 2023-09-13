import json
import os
import random
from argparse import ArgumentParser
from copy import deepcopy
from typing import List, Dict

from data.readers import LogiQAReaderV2

_templates = [
    "Because {}, {}.",
    "Since {}, {}.",
    "As {}, {}.",
    "For {}, {}.",
    "Given {}, {}.",
    "{}, then {}.",
    "{}, therefore {}.",
    "{}, so {}.",
    "Considering that {}, hence {}.",
]


def compose_proofs(proofs: List[str], triples: Dict[str, str], inter_conclusions: Dict[str, str], hypothesis: str):
    full_text_proof = []
    for p in proofs:
        segments_a = p.split("->")[0]
        segments_b = p.split("->")[1]
        assert segments_a, p
        assert segments_b, p

        for sent_k, sent_v in triples.items():
            if sent_k in segments_a:
                segments_a = segments_a.replace(sent_k, sent_v)

            if sent_k in segments_b:
                segments_b = segments_b.replace(sent_k, sent_v)

        for int_k, int_v in inter_conclusions.items():
            if f"{int_k}:" in segments_a:
                segments_a = segments_a.replace(f"{int_k}:", f"")
            elif int_k in segments_a:
                segments_a = segments_a.replace(int_k, int_v)

            if f"{int_k}:" in segments_b:
                segments_b = segments_b.replace(f"{int_k}:", f"")
            elif int_k in segments_b:
                segments_b = segments_b.replace(int_k, int_v)

        segments_a = segments_a.replace("&", "and")
        segments_b = segments_b.replace("&", "and")

        segments_a = segments_a.replace("hypothesis", hypothesis)
        segments_b = segments_b.replace("hypothesis", hypothesis)

        full_text_proof.append(random.choice(_templates).format(segments_a, segments_b))

    full_text_proof = "\n".join(full_text_proof)
    return full_text_proof


def entailment_bank_reader_v1(file: str, negative_sample_num: int = 5, split_input_output: bool = False):
    data = open(file).readlines()

    items = []
    for line in data:
        item = json.loads(line)
        proof = item["proof"]
        triples = item["meta"]["triples"]
        inters = item["meta"]["intermediate_conclusions"]
        hypothesis = item["hypothesis"]

        delete_list = {x["uuid"]: x["fact"] for x in item["meta"]["delete_list"]}

        prefix = item["question"]
        suffix = f"So the answer is: {item['answer']}."

        proofs = proof.split(";")
        proofs = [p.strip() for p in proofs if p.strip()]

        full_text_proof = compose_proofs(proofs, triples, inters, hypothesis)

        neg_sample_num = min(negative_sample_num, len(triples) + len(inters), len(delete_list))

        negative_proofs = []
        distractors = {}
        distractors.update(triples)
        distractors.update(inters)
        for _ in range(neg_sample_num):
            src = random.choice(list(delete_list.keys()))
            src_fact = delete_list.pop(src)

            tgt = random.choice(list(distractors.keys()))
            distractors.pop(tgt)
            if tgt.startswith("int"):
                inter_copy = deepcopy(inters)
                inter_copy[tgt] = src_fact
                negative_proofs.append(prefix + "\n\n" + compose_proofs(proofs, triples, inter_copy, hypothesis) + "\n\n" + suffix)
            elif tgt.startswith("sent"):
                triple_copy = deepcopy(triples)
                triple_copy[tgt] = src_fact
                negative_proofs.append(prefix + "\n\n" + compose_proofs(proofs, triple_copy, inters, hypothesis) + "\n\n" + suffix)
            else:
                raise ValueError(f"Unknown target type: {tgt}")

        if split_input_output:
            items.append({
                "pos_input": prefix + "\n\n",
                "pos_output": full_text_proof + "\n\n" + suffix,
                "neg_input": [prefix + "\n\n"] * len(negative_proofs),
                "neg_output": [p + "\n\n" + suffix for p in negative_proofs],
                "meta_data": {
                    "source": "entailment_bank",
                    "version": "v1",
                    "id": item["id"],
                }
            })
        else:
            items.append({
                "pos": prefix + "\n\n" + full_text_proof + "\n\n" + suffix,
                "neg": negative_proofs,
                "meta_data": {
                    "source": "entailment_bank",
                    "version": "v1",
                    "id": item["id"],
                }
            })

    return items


def logiqa_v2_reader_v1(file: str):
    reader = LogiQAReaderV2()
    all_context, all_question, all_option_list, all_label = reader(file)

    items = []
    for index in range(len(all_context)):
        context = all_context[index]
        question = all_question[index]
        option_list = all_option_list[index]
        label = all_label[index]

        template = "{}\n\n{}\n\n{}"

        negs = []
        pos = ""
        for op_id, option in enumerate(option_list):
            if op_id == label:
                pos = template.format(context, question, option)
            else:
                negs.append(template.format(context, question, option))

        items.append({
            "pos": pos,
            "neg": negs,
            "meta_data": {
                "source": "logiqa_v2",
                "version": "v1",
                "id": index,
            }
        })

    return items


def main():
    parser = ArgumentParser()
    parser.add_argument("--entailment_bank", type=str, default=None)
    parser.add_argument("--logiqa_v2", type=str, default=None)
    parser.add_argument("--output_file", type=str, default=None, required=True)
    parser.add_argument("--negative_sample_num", type=int, default=5)
    parser.add_argument("--split_input_output", action="store_true")
    args = parser.parse_args()

    assert any([args.entailment_bank, args.logiqa_v2]), "Please specify at least one dataset."

    file_suffix = "ccb_v1_0"
    items = []
    if args.entailment_bank:
        items.extend(entailment_bank_reader_v1(args.entailment_bank, args.negative_sample_num, args.split_input_output))
        file_suffix += "_enb" + str(args.negative_sample_num) + "neg" + ("_split" if args.split_input_output else "")

    if args.logiqa_v2:
        items.extend(logiqa_v2_reader_v1(args.logiqa_v2))
        file_suffix += "_lqv2"

    # make directory is needed.
    if not os.path.exists(os.path.dirname(args.output_file)):
        os.makedirs(os.path.dirname(args.output_file))
    json.dump(items, open(args.output_file + file_suffix + ".json", "w"), indent=2)


if __name__ == '__main__':
    main()
