import torch
from torch.utils.data import Dataset
import os
import math
import random
import gzip


def create_datasets(directory, test_split=0.1, suffix=None):
    if suffix is None:
        suffix = ".pt"

    files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(suffix)]
    random.shuffle(files)
    split_index = math.floor(len(files) * test_split)

    train_files = files[split_index:]
    test_files = files[:split_index]
    return DatasetFolder(train_files), DatasetFolder(test_files)


class DatasetFolder(Dataset):
    def __init__(self, files):
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        try:
            with gzip.open(self.files[index], 'rb') as f:
                data = torch.load(f)
        except:
            print("Failed to deserialize file, moving to next one")
            return self[index + 1]

        return data
