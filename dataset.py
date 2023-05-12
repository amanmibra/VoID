import os

from torch.utils.data import Dataset
import pandas as pd
import torchaudio

class VoiceDataset(Dataset):

    def __init__(self, data_directory):
        self._data_path = os.path.join(data_directory)
        self._labels = os.listdir(self._data_path)

        self.audio_files, self.audio_labels = self._join_audio_files()

    def __len__(self):
        total_audio_files = 0
        for label in self._labels:
            label_path = os.path.join(self._data_path, label)
            total_audio_files += len(os.listdir(label_path))
        return total_audio_files

    def __getitem__(self, index):
        return self.audio_files[index], self.audio_labels[index]

    def _join_audio_files(self):
        audio_files = []
        audio_labels = []

        for label in self._labels:
            label_path = os.path.join(self._data_path, label)
            for f in os.listdir(label_path):
                audio_files.append(f)
                audio_labels.append(label)

        return audio_files, audio_labels