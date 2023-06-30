import argparse
import json

import torch
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",
                        default="wiki_erica_path/v9.1_fixed/"
                                "distant_path_v9.1_fix_no_shuffle.dev.pkl_llama_True_3_3_512_0.4_5_1.0_1.0_0.0_8_path_v9.1.2_seq2seq")
    parser.add_argument("--prediction_file", type=str, required=True)
    args = parser.parse_args()

    data = torch.load(args.data)[0]
    predictions = json.load(open(args.prediction_file, 'r'))

    id2pred = {pred["index"]: pred for pred in predictions}

    metrics = {"counterfactual": 0, "normal": 0}
    counterfactual_llh = {"positive": 0, "negative": 0}
    num_counterfactual = 0
    num_normal = 0
    for item_id, item in tqdm(enumerate(data), total=len(data)):
        if "h" in item:  # counterfactual data
            num_counterfactual += 1
            metrics["counterfactual"] += (3 * id2pred[item_id]["logits"][0] - sum(id2pred[item_id]["logits"][1:])) / 3.0
            counterfactual_llh["positive"] += id2pred[item_id]["logits"][0]
            counterfactual_llh["negative"] += sum(id2pred[item_id]["logits"][1:])
        else:  # normal data
            num_normal += 1
            metrics["normal"] += (3 * id2pred[item_id]["logits"][0] - sum(id2pred[item_id]["logits"][1:])) / 3.0

    metrics = {
        "counterfactual": metrics["counterfactual"] / num_counterfactual,
        "normal": metrics["normal"] / num_normal
    }
    print(metrics)
    counterfactual_llh = {
        "positive": counterfactual_llh["positive"] / num_counterfactual,
        "negative": counterfactual_llh["negative"] / 3 / num_counterfactual,
    }
    print(counterfactual_llh)
