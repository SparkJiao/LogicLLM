import json
from glob import glob

from transformers import PreTrainedTokenizer

from general_util.logger import get_child_logger

logger = get_child_logger(__name__)


class BBHReader:
    def __init__(self, file_path: str, tokenizer: PreTrainedTokenizer):
        files = list(glob(file_path + "/*.json"))
        self.data = []
        for file in files:
            task_data = json.load(open(file, "r"))["examples"]
            task = file.split("/")[-1][: -len(".json")]
            for exp_id, example in enumerate(task_data):
                example["task"] = task
                if task == "sports_understanding":
                    example["input"] += "\nOptions: - yes - no"
                elif task == "web_of_lies":
                    example["input"] += "\nOptions: - Yes - No"
                elif task == "dyck_languages":
                    example["input"] += "\nOutput: "
                elif task == "word_sorting":
                    example["input"] += "\nThe answer is: List: "
                elif task == "object_counting":
                    example["input"] += "\nThe answer is: "
                elif task == "boolean_expressions":
                    example["input"] += "\nOptions: - True - False"

                example["index"] = f"{task}-{exp_id}"
                if "Options" in example["input"]:
                    example["input"] += "\nThe answer is: "

                # if example["target"] lies in between (A) and (Z):
                if len(example["target"]) == 3 and "Z" >= example["target"][1] >= "A":
                    example["target"] = example["target"][1]
                    example["input"] += "("
            self.data.extend(task_data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def load_prompt(file):
    with open(file, "r") as f:
        lines = f.readlines()
    return "".join(lines[2:])


def load_prompt_direct(file):
    with open(file, "r") as f:
        text = f.read()

    exemplars = text.split("\n\n")
    trigger = "A: Let's think step by step."
    direct_prompts = []
    prefix = ""
    if "snarks" in file:
        prefix = exemplars[1]
        exemplars = exemplars[1:]
    for example in exemplars[1:]:
        if trigger in example:
            groups = example.split(trigger)
            assert len(groups) == 2, (example, groups)
            q = groups[0]
            a = groups[1]
            direct_a = a.split("So the answer is")[-1]
            if prefix:
                direct_prompts.append(prefix + "\n\n" + q + "\nA: The answer is " + direct_a)
            else:
                direct_prompts.append(q + "A: The answer is " + direct_a)
        else:
            raise ValueError(f"Trigger not found in example: {file}\n{example}\n{exemplars}")
    assert len(direct_prompts) == 3
    return "\n\n".join(direct_prompts)


class BBHReaderCoT:
    def __init__(self, file_path: str, tokenizer: PreTrainedTokenizer, cot_prompts_path: str):
        files = list(glob(file_path + "/*.json"))
        cot_files = list(glob(cot_prompts_path + "/*.txt"))

        task2cot = {}
        for file in cot_files:
            task = file.split("/")[-1][: -len(".txt")]
            task2cot[task] = load_prompt(file)

        self.data = []
        for file in files:
            task_data = json.load(open(file, "r"))["examples"]
            task = file.split("/")[-1][: -len(".json")]
            for exp_id, example in enumerate(task_data):
                example["task"] = task
                example["input"] = task2cot[task] + "\n\n" + "Q: " + example["input"] + "\nA: Let's think step by step."

                example["index"] = f"{task}-{exp_id}"

                if len(example["target"]) == 3 and "Z" >= example["target"][1] >= "A":
                    example["target"] = example["target"][1]

            self.data.extend(task_data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


class BBHReaderDirectFewShot:
    def __init__(self, file_path: str, tokenizer: PreTrainedTokenizer, cot_prompts_path: str):
        files = list(glob(file_path + "/*.json"))
        cot_files = list(glob(cot_prompts_path + "/*.txt"))

        task2prompt = {}
        for file in cot_files:
            task = file.split("/")[-1][: -len(".txt")]
            task2prompt[task] = load_prompt_direct(file)

        self.data = []
        for file in files:
            task_data = json.load(open(file, "r"))["examples"]
            task = file.split("/")[-1][: -len(".json")]
            for exp_id, example in enumerate(task_data):
                example["task"] = task
                example["input"] = task2prompt[task] + "\n\n" + "Q: " + example["input"] + "\nA: The answer is"

                example["index"] = f"{task}-{exp_id}"

                if len(example["target"]) == 3 and "Z" >= example["target"][1] >= "A":
                    example["target"] = example["target"][1]

            self.data.extend(task_data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

#
# "Answer following question: {Input} \n\n The answer is".format():
#
# Zero-shot:
# Few-shot Direct: The answer is
# # Few-shot CoT: "A: Let's think step by step.: "
#
# Input: Context: xxx Question: xxx Optiosn: (A): (B): (C): xxx
# Output:
