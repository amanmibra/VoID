import sys
sys.path.append('..')

import copy
import time

# torch
import torch
import torchaudio
from torch import nn
from torch.utils.data import DataLoader

# huggingface
from transformers import AutoModelForAudioXVector, Wav2Vec2FeatureExtractor

# modal
from modal import Mount, Stub, gpu, create_package_mounts

# internal
from pipelines.images import training_image_pip

# model
from dataset import VoiceDataset
from cnn import CNNetwork

# script defaults
BATCH_SIZE = 128
EPOCHS = 100
LEARNING_RATE = 0.001

TRAIN_FILE="data/aisf/augmented/train"
TEST_FILE="data/aisf/augmented/test"
SAMPLE_RATE=48000

@stub.function(
    gpu=gpu.A100(memory=20),
    mounts=[
        Mount.from_local_file(local_path='dataset.py'),
        Mount.from_local_file(local_path='cnn.py'),
    ],
)
def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False, device="cuda"):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history

def get_device():
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    return device

@stub.local_entrypoint()
def main():
    print("Initiating model fine-tuning...")
    device = get_device()

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained('microsoft/unispeech-sat-base-plus-sv')
    model = AutoModelForAudioXVector.from_pretrained("microsoft/unispeech-sat-base-plus-sv")

    # instantiating our dataset object and create data loader
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=2048,
        hop_length=512,
        n_mels=128
    )

    # dataset/dataloader
    train_dataset = VoiceDataset(TRAIN_FILE, mel_spectrogram, device, target_sample_rate=16000, time_limit_in_secs=5)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    train_wavs = [wav[0] for wav in train_dataset.wavs]

    test_dataset = VoiceDataset(TEST_FILE, mel_spectrogram, device, target_sample_rate=16000, time_limit_in_secs=5)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_wavs = [wav[0] for wav in test_dataset.wavs]