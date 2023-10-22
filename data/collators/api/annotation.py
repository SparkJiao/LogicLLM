import os.path
from typing import Optional

from torch.utils.data import Dataset
from omegaconf import ListConfig
import json

from general_util.logger import get_child_logger

logger = get_child_logger("Annotation")


class AnnotationDataset(Dataset):
    def __init__(self, file_path: str, file_names: ListConfig, output_file: Optional[str] = None, **kwargs):
        super().__init__()
        file_names = list(file_names)
        all_data = []
        for file in file_names:
            data = json.load(open(os.path.join(file_path, file)))
            all_data.extend(data)

        self.data = all_data
        if os.path.exists(output_file):
            existing_data = set()
            with open(output_file, "r") as f:
                for line in f.readlines():
                    existing_data.add(json.loads(line)["text"])
            self.data = [tmp for tmp in self.data if tmp["meta_data"]["text"] not in existing_data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
