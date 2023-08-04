import json


def read_model_cot_prediction(file: str, clean_input: bool = True, answer_trigger: str = "", split_index: int = -1):
    predictions = json.load(open(file))

    items = []
    for pred in predictions:
        output = pred["output"]

        if clean_input:
            output = pred["output"].replace(pred["input"], "").strip()

        if answer_trigger:
            output = output.split(answer_trigger)[split_index].strip()

        items.append({
            "input": pred["input"],
            "index": pred["index"],
            "label": pred["label"],
            "output": output,
        })

    return sorted(items, key=lambda x: x["index"])


def read_model_cot_prediction_sentence_split(file: str, clean_input: bool = True, answer_trigger: str = "", split_index: int = -1):
    predictions = json.load(open(file))

    items = []
    for pred in predictions:
        output = pred["output"]

        if clean_input:
            output = pred["output"].replace(pred["input"], "").strip()

        if answer_trigger:
            output = output.split(answer_trigger)[split_index].strip()

        steps = output.split("\n\n")

        for step_id in range(len(steps)):
            items.append({
                "input": pred["input"],
                "index": f"{pred['index']}Step{step_id}",
                "label": pred["label"],
                "output": "\n\n".join(steps[:step_id + 1]),
            })

    return sorted(items, key=lambda x: x["index"])


def entailment_bank_reader_v1(file: str):
    data = open(file).readlines()

    items = []
    for line in data:
        item = json.loads(line)
        full_text_proof = item["full_text_proof"]
        full_text_proof = full_text_proof.replace("[BECAUSE]", "\nBecause")
        full_text_proof = full_text_proof.replace("[AND]", "and")
        full_text_proof = full_text_proof.replace("[INFER]", ", then")

        for idx, v in item["meta"]["intermediate_conclusions"].items():
            full_text_proof = full_text_proof.replace(f"{idx}:", "")
            full_text_proof = full_text_proof.replace(idx, v)

        full_text_proof = full_text_proof.strip()
        full_text_proof = " ".join(full_text_proof.split())

        items.append({
            "input": f"Question: {item['question']}\n\nHypothesis: {item['hypothesis']}\n\nProof:",
            "index": item["id"],
            "output": full_text_proof,
            "label": item["label"],
        })
