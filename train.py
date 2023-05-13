from datetime import datetime
from tqdm import tqdm

# torch
import torch
import torchaudio
from torch import nn
from torch.utils.data import DataLoader

# internal
from dataset import VoiceDataset
from cnn import CNNetwork

BATCH_SIZE = 128
EPOCHS = 100
LEARNING_RATE = 0.001

TRAIN_FILE="data/train"
SAMPLE_RATE=16000

def train(model, dataloader, loss_fn, optimizer, device, epochs):
  for i in tqdm(range(epochs), "Training model..."):
    print(f"Epoch {i + 1}")

    train_epoch(model, dataloader, loss_fn, optimizer, device)

    print (f"----------------------------------- \n")
  
  print("---- Finished Training ----")
  

def train_epoch(model, dataloader, loss_fn, optimizer, device):
  for x, y in dataloader:
    x, y = x.to(device), y.to(device)

    # calculate loss
    pred = model(x)
    loss = loss_fn(pred, y)

    # backprop and update weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  
  print(f"Loss: {loss.item()}") 

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using {device} device.")

    # instantiating our dataset object and create data loader
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

    train_dataset = VoiceDataset(TRAIN_FILE, mel_spectrogram, SAMPLE_RATE, device)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # construct model
    model = CNNetwork().to(device)
    print(model)

    # init loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


    # train model
    train(model, train_dataloader, loss_fn, optimizer, device, EPOCHS)

    # save model
    now = datetime.now()
    now = now.strftime("%Y%m%d_%H%M%S")
    model_filename = f"models/void_{now}.pth"
    torch.save(model.state_dict(), model_filename)
    print(f"Trained feed forward net saved at {model_filename}")