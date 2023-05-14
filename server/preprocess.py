"""
Util functions to process any incoming audio data to be processable by the model 
"""
import torch
import torchaudio

def process_raw_wav(wav, sample_rate, target_sample_rate=4800, wav_length=5):
    num_samples = wav_length * target_sample_rate

    wav = _resample(wav, sample_rate, target_sample_rate)
    wav = _mix_down(wav)
    wav = _cut(wav, num_samples)
    wav = _pad(wav, num_samples)

    return wav

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