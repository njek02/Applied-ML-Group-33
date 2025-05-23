import os
import torch
from torch.utils.data import Dataset, DataLoader
from preprocessing.preprocess_CNN_sample import preprocess_sample
import numpy as np
import pandas as pd


class AudioDataset(Dataset):
    def __init__(self, folder_path, csv_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        df = pd.read_csv(csv_path)
        self.file_label_dict = dict(zip(df['clip_name'], df['label']))
        self.file_list = [f for f in os.listdir(folder_path) if f.endswith('.aiff')]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        file_path = os.path.join(self.folder_path, file_name)

        spec = preprocess_sample(file_path)
        spec = torch.tensor(spec, dtype=torch.float32)

        if self.transform:
            spec = self.transform(spec)

        label = self.file_label_dict[file_name]
        label = torch.tensor(label, dtype=torch.long)

        return spec, label
