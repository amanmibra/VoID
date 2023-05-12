import os

import torch
from torch.utils.data import Dataset
import pandas as pd
import torchaudio

class VoiceDataset(Dataset):

    def __init__(self, data_directory, transformation, target_sample_rate):
        # file processing
        self._data_path = os.path.join(data_directory)
        self._labels = os.listdir(self._data_path)

        self.audio_files_labels = self._join_audio_files()

        # audio processing
        self.transformation = transformation
        self.target_sample_rate = target_sample_rate

    def __len__(self):
        total_audio_files = 0
        for label in self._labels:
            label_path = os.path.join(self._data_path, label)
            total_audio_files += len(os.listdir(label_path))
        return total_audio_files

    def __getitem__(self, index):
        file, label = self.audio_files_labels[index]
        filepath = os.path.join(self._data_path, label, file)

        wav, sr = torchaudio.load(filepath, normalize=True)
        wav = self._resample(wav, sr)
        wav = self._mix_down(wav)
        wav = self.transformation(wav)

        return wav, label


    def _join_audio_files(self):
        """Join all the audio file names and labels into one single dimenional array"""
        audio_files_labels = []

        for label in self._labels:
            label_path = os.path.join(self._data_path, label)
            for f in os.listdir(label_path):
                audio_files_labels.append((f, label))

        return audio_files_labels

    def _resample(self, wav, current_sample_rate):
        """Resample audio to the target sample rate, if necessary"""
        if current_sample_rate != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(current_sample_rate, self.target_sample_rate)
            wav = resampler(wav)
        
        return wav

    def _mix_down(self, wav):
        """Mix down audio to a single channel, if necessary"""
        if wav.shape[0] > 1:
            wav = torch.mean(wav, dim=0, keepdim=True)

        return wav