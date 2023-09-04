from typing import Dict


def extract_ending_entity(example: Dict):
    direct = example["positive"] if "positive" in example else example["condition"]
    h = direct["h"]
    t = direct["t"]
    h_span = ""
    t_span = ""
    # print(direct)
    for r in direct["entity_replacement"]:
        if r["id"] == h:
            h_span = r["tgt"]
        elif r["id"] == t:
            t_span = r["tgt"]

    if not h_span or not t_span:
        return False, None, None

    return True, h_span, t_span


def extract_ending_entity_and_anonymization(example: Dict):
    direct = example["positive"] if "positive" in example else example["condition"]
    context = example["context"]
    h = direct["h"]
    t = direct["t"]

    for r in direct["entity_replacement"]:
        if r["id"] == h:
            direct["spans"][r["span_index"]] = "[X]"
        elif r["id"] == t:
            direct["spans"][r["span_index"]] = "[Y]"

    for sent in context:
        for r in sent["entity_replacement"]:
            if r["id"] == h:
                sent["spans"][r["span_index"]] = "[X]"
            elif r["id"] == t:
                sent["spans"][r["span_index"]] = "[Y]"

    return example, "[X]", "[Y]"


def extract_ending_entity_and_replace(example: Dict, anchor_ent_a, anchor_ent_b):
    direct = example["positive"] if "positive" in example else example["condition"]
    context = example["context"]
    h = direct["h"]
    t = direct["t"]

    for r in direct["entity_replacement"]:
        if r["id"] == h:
            direct["spans"][r["span_index"]] = anchor_ent_a
        elif r["id"] == t:
            direct["spans"][r["span_index"]] = anchor_ent_b

    for sent in context:
        for r in sent["entity_replacement"]:
            if r["id"] == h:
                sent["spans"][r["span_index"]] = anchor_ent_a
            elif r["id"] == t:
                sent["spans"][r["span_index"]] = anchor_ent_b

    return example, anchor_ent_a, anchor_ent_b
