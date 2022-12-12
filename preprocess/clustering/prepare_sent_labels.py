import argparse
import pickle
import torch
from typing import Dict
from tqdm import tqdm


def load_edge_relation_file(edge_relation_file):
    edge_hidden, edge_relation_labels, _ = torch.load(edge_relation_file, map_location="cpu")
    edge_labels = {}
    for edge, edge_label in zip(edge_hidden.keys(), edge_relation_labels):
        edge_labels[edge] = edge_label.item()

    return edge_labels


def assign_sent_labels_w_rel_and_parse(example, edge_labels: Dict[str, int]):
    for s_id, sent in example["selected_sentences"].items():
        assert sent["h"] != -1 and sent["t"] != -1
        edge = f"{sent['h']}\t{sent['t']}"
        if edge in edge_labels:
            sent["edge_label"] = edge_labels[edge]
        else:
            sent["edge_label"] = -1

    for pos in example["pos"]:
        edge = f"{pos['h']}\t{pos['t']}"
        if edge in edge_labels:
            pos["edge_label"] = edge_labels[edge]
        else:
            pos["edge_label"] = -1

    return example


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--example_file", type=str)
    parser.add_argument("--edge_relation_file", type=str)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--num_workers", type=int)

    args = parser.parse_args()

    data = pickle.load(open(args.example_file, "rb"))
    examples = data["examples"]

    edge_labels = load_edge_relation_file(args.edge_relation_file)
    new_examples = []
    for exp in tqdm(examples, total=len(examples)):
        new_examples.append(assign_sent_labels_w_rel_and_parse(exp, edge_labels))
        assert "edge_label" in new_examples[-1]["pos"][0]

    data["examples"] = new_examples
    pickle.dump(data, open(args.output_file, "wb"))


if __name__ == '__main__':
    main()
