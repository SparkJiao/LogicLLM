from torch.utils.data import Dataset


class TestDataset(Dataset):
    def __init__(self, file_path, tokenizer):
        super().__init__()
        self.data = ["My name is Jiao Fangkai."]

    def __len__(self):
        return 100000000

    def __getitem__(self, index):
        return {
            "flan": {
                "inputs": self.data[0],
                "targets": self.data[0],
            },
            "index": index,
        }
