import os

import torch
from torch.utils.data import Dataset
import pandas as pd
import torchaudio

class VoiceDataset(Dataset):

    def __init__(
            self,
            data_directory,
            transformation,
            target_sample_rate,
            device,
            time_limit_in_secs=5,
        ):
        # file processing
        self._data_path = os.path.join(data_directory)
        self._labels = os.listdir(self._data_path)
        self.audio_files_labels = self._join_audio_files()

        self.device = device

        # audio processing
        self.transformation = transformation
        self.target_sample_rate = target_sample_rate
        self.num_samples = time_limit_in_secs * self.target_sample_rate

    def __len__(self):
        total_audio_files = 0
        for label in self._labels:
            label_path = os.path.join(self._data_path, label)
            total_audio_files += len(os.listdir(label_path))
        return total_audio_files

    def __getitem__(self, index):
        # get file
        file, label = self.audio_files_labels[index]
        filepath = os.path.join(self._data_path, label, file)

        # load wav
        wav, sr = torchaudio.load(filepath, normalize=True)

        # modify wav file, if necessary
        wav = wav.to(self.device)
        wav = self._resample(wav, sr)
        wav = self._mix_down(wav)
        wav = self._cut_or_pad(wav)
        
        # apply transformation
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

    def _cut_or_pad(self, wav):
        """Modify audio if number of samples != target number of samples of the dataset.

        If there are too many samples, cut the audio.
        If there are not enough samples, pad the audio with zeros.
        """

        length_signal =  wav.shape[1]
        if length_signal > self.num_samples:
            wav = wav[:, :self.num_samples]
        elif length_signal < self.num_samples:
            num_of_missing_samples = self.num_samples - length_signal
            pad = (0, num_of_missing_samples)
            wav = torch.nn.functional.pad(wav, pad)

        return wav
