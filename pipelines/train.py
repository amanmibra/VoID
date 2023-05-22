import sys
sys.path.append('..')
import time

# torch
import torch
import torchaudio
from torch import nn
from torch.utils.data import DataLoader

# modal
from modal import Mount, Secret, Stub, gpu, create_package_mounts

# internal
from pipelines.images import training_image_pip

# model
from dataset import VoiceDataset
from cnn import CNNetwork

# script defaults
BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.001

TRAIN_FILE="data/train"
TEST_FILE="data/test"
SAMPLE_RATE=48000

stub = Stub(
    "void-training",
    image=training_image_pip,
)

@stub.function(
    gpu="any",
    mounts=[
        Mount.from_local_file(local_path='dataset.py'),
        Mount.from_local_file(local_path='cnn.py'),
    ],
    timeout=EPOCHS * 200,
    secret=Secret.from_name("wandb"),
)
def train(
        model,
        train_dataloader,
        loss_fn,
        optimizer,
        origin_device="cuda",
        epochs=10,
    ):
    import os

    import time
    import torch
    import wandb

    print("Begin model training...")
    begin = time.time()

    modal_device = origin_device

    # set model to cuda
    if torch.cuda.is_available() and modal_device != "cuda":
        modal_device = "cuda"
        model = model.to(modal_device)

    # metrics
    training_acc = []
    training_loss = []

    wandb.init(project="void-training")

    for i in range(epochs):
        print(f"Epoch {i + 1}/{epochs}")
        then = time.time()

        # train model
        model, train_epoch_loss, train_epoch_acc = train_epoch.call(model, train_dataloader, loss_fn, optimizer, modal_device)

        # training metrics
        training_loss.append(train_epoch_loss/len(train_dataloader))
        training_acc.append(train_epoch_acc/len(train_dataloader))
        wandb.log({'training_loss': training_loss[i], 'training_acc': training_acc[i]})        

        now = time.time()
        print("Training Loss: {:.2f}, Training Accuracy: {:.4f}, Time: {:.2f}s".format(training_loss[i], training_acc[i], now - then))

        print ("-------------------------------------------------------- \n")
    
    end = time.time()
    wandb.finish()
    print("-------- Finished Training --------")
    print("-------- Total Time -- {:.2f}s --------".format(end - begin))

    return model.to(origin_device)

@stub.function(
    gpu="any",
    mounts=[
        Mount.from_local_file(local_path='dataset.py'),
        Mount.from_local_file(local_path='cnn.py'),
    ]
)
def train_epoch(model, train_dataloader, loss_fn, optimizer, device):
    import torch
    from tqdm import tqdm

    train_loss = 0.0
    train_acc = 0.0
    total = 0.0

    model.train()

    for wav, target in tqdm(train_dataloader):
        wav, target = wav.to(device), target.to(device)

        # calculate loss
        output = model(wav)
        loss = loss_fn(output, target)

        # backprop and update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # metrics
        train_loss += loss.item()
        prediction = torch.argmax(output, 1)
        train_acc += (prediction == target).sum().item()/len(prediction)
        total += 1
       
    return model, train_loss, train_acc

def save_model(model):
    now = time.strftime("%Y%m%d_%H%M%S")
    model_filename = f"models/void_{now}.pth"
    torch.save(model.state_dict(), model_filename)
    print(f"Trained void model saved at {model_filename}")

def get_device():
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    return device

@stub.local_entrypoint()
def main():
    print("Initiating model training...")
    device = get_device()

    # instantiating our dataset object and create data loader
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=2048,
        hop_length=512,
        n_mels=128
    )

    # dataset/dataloader
    train_dataset = VoiceDataset(TRAIN_FILE, mel_spectrogram, device, time_limit_in_secs=3)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # construct model
    model = CNNetwork()

    # init loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # train model
    model = train.call(model, train_dataloader, loss_fn, optimizer, device, 3)

    # save model
    save_model(model)
    