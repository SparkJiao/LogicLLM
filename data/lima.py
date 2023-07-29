from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
import json


class LIMADataset(Dataset):
    def __init__(self, file_path: str, tokenizer: PreTrainedTokenizer, eot_token: str = "<eot>"):
        super().__init__()
        self.data = []
        with open(file_path, "r") as f:
            for line in f.readlines():
                self.data.append(json.loads(line)["conversations"])
        self.eot_token = eot_token

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.eot_token.join(self.data[index])
