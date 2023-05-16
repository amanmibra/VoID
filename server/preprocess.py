"""
Util functions to process any incoming audio data to be processable by the model 
"""
import torch
import torchaudio

DEFAULT_SAMPLE_RATE=48000

def process_from_filename(filename, target_sample_rate=DEFAULT_SAMPLE_RATE, wav_length=5):
    wav, sample_rate = torchaudio.load(filename)

    wav = process_raw_wav(wav, sample_rate, target_sample_rate, wav_length)

    spec = _wav_to_spec(wav, target_sample_rate)

    return spec

def process_raw_wav(wav, sample_rate=DEFAULT_SAMPLE_RATE, target_sample_rate=DEFAULT_SAMPLE_RATE, wav_length=5):
    num_samples = wav_length * target_sample_rate

    wav = _resample(wav, sample_rate, target_sample_rate)
    wav = _mix_down(wav)
    wav = _cut(wav, num_samples)
    wav = _pad(wav, num_samples)

    return wav

def _wav_to_spec(wav, target_sample_rate):
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=target_sample_rate,
        n_fft=2048,
        hop_length=512,
        n_mels=128,
    )

    return mel_spectrogram(wav)

def _resample(wav, sample_rate, target_sample_rate):
    if sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(sample_rate, target_sample_rate)
        wav = resampler(wav)
    
    return wav

def _mix_down(wav):
    if wav.shape[0] > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)
    
    return wav

def _cut(wav, num_samples):
    if wav.shape[1] > num_samples:
        wav = wav[:, :num_samples]
    
    return wav

def _pad(wav, num_samples):
    if wav.shape[1] < num_samples:
        missing_samples = num_samples - wav.shape[1]
        pad = (0, missing_samples)
        wav = torch.nn.function.pad(wav, pad)
    
    return wav