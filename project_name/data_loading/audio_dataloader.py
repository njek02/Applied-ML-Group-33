import os
import torch
from torch.utils.data import Dataset, DataLoader
from preprocessing.preprocess_CNN_sample import preprocess_sample
import numpy as np
import pandas as pd


class AudioDataset(Dataset):
    """
    Custom Dataset for loading audio files and their labels.
    """    
    def __init__(self, folder_path: str, csv_path: str, transform=None) -> None:
        """
        Dataset containing audio files and their corresponding labels.

        Args:
            folder_path (str): Path to the folder containing audio files.
            csv_path (str): Path to the CSV file containing file names and labels.
            transform (callable, optional): A function/transform to apply to the audio samples defaulted to None.
        """
        self.folder_path = folder_path
        df = pd.read_csv(csv_path)
        self.file_label_dict = dict(zip(df['clip_name'], df['label']))
        self.file_list = [f for f in os.listdir(folder_path) if f.endswith('.aiff')]
        self.transform = transform

    def __len__(self) -> int:
        """
        Returns the total number of audio files in the dataset.

        Returns:
            int: The number of audio files.
        """        
        return len(self.file_list)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            idx (int): Index of the audio file to retrieve.

        Returns:
            spec, label (tuple[torch.Tensor, torch.Tensor]): A tuple containing the spectrogram and the corresponding label.
        """
        file_name = self.file_list[idx]
        file_path = os.path.join(self.folder_path, file_name)

        spec = preprocess_sample(file_path)
        spec = torch.tensor(spec, dtype=torch.float32)

        if self.transform:
            spec = self.transform(spec)

        label = self.file_label_dict[file_name]
        label = torch.tensor(label, dtype=torch.long)

        return spec, label
