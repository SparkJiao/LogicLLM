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
                    example["input"] += " Options: - yes - no"
                example["index"] = f"{task}-{exp_id}"
            self.data.extend(task_data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
