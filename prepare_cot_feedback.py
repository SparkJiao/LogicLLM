import json
from argparse import ArgumentParser
from collections import defaultdict
import os
import re


def prepare_actor_feedback(rw_file, cot_file, output_dir, answer_trigger: str = "Let's think step by step:"):
    rw_predictions = json.load(open(rw_file))

    cot_data = json.load(open(cot_file))

    id2rw = {item["index"]: item for item in rw_predictions}

    assert len(id2rw) % len(cot_data) == 0, (len(id2rw), len(cot_data))

    total = len(id2rw)

    if len(id2rw) != len(cot_data):
        mul = len(id2rw) // len(cot_data)
        id2rw_group = defaultdict(list)
        id2rw = sorted(id2rw.items(), key=lambda x: x[0])
        for item in id2rw:
            id2rw_group[item[0] // mul].append(item[1])
        id2rw = id2rw_group

    regrex = r"A|B|C|D|E"
    mapping = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, }

    cnt = 0
    for item in cot_data:
        rw = id2rw[item["index"]]
        if isinstance(rw, list):
            item["reward"] = [rw_item["pred"] for rw_item in rw]
        elif isinstance(rw, dict):
            item["reward"] = rw["pred"]
        else:
            raise ValueError(rw)

        if isinstance(item["output"], list):
            for o_id, o in enumerate(item["output"]):
                if o.find(item["input"]) != -1:
                    item["output"][o_id] = o.replace(item["input"], "")

                x = re.findall(regrex, item["output"][o_id])
                if x:
                    item["cleaned_output"][o_id] = x[-1]

                if item["cleaned_output"][o_id] == item["label"]:
                    cnt += 1
        else:
            if item["output"].find(item["input"]) != -1:
                item["output"] = item["output"].replace(item["input"], "")

            # if answer_trigger in item["output"]:
            #     response = item["output"].split(answer_trigger)[1]
            #     x = re.findall(regrex, response)
            #     if x:
            #         item["cleaned_output"] = x[-1]
            #     else:
            #         x = re.findall(regrex, item["output"])
            #         if x:
            #             item["cleaned_output"] = x[-1]
            # else:
            x = re.findall(regrex, item["output"])
            # x = mapping[x[-1]]
            # assert x, item["output"]
            if x:
                item["cleaned_output"] = x[-1]

            if item["cleaned_output"] == item["label"]:
                cnt += 1

    print(cnt / total)

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    json.dump(cot_data, open(os.path.join(output_dir, "cot_feedback.json"), "w"), indent=2)

    meta_data = {
        "reward_prediction": rw_file,
        "cot_prediction": cot_file,
    }

    json.dump(meta_data, open(os.path.join(output_dir, "cot_feedback_meta.json"), "w"), indent=2)

    return cot_data


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--rw_file", type=str, required=True)
    parser.add_argument("--cot_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    prepare_actor_feedback(args.rw_file, args.cot_file, args.output_dir)
