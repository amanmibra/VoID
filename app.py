import torch
import gradio as gr

from cnn import CNNetwork
from server.preprocess import process_raw_wav, _wav_to_spec

model = CNNetwork()
state_dict = torch.load('models/void_20230522_223553.pth')
model.load_state_dict(state_dict)

LABELS = ["shafqat", "aman", "jake"]


def greet(input):
    sr, wav = input

    wav = torch.tensor([wav]).float()
    wav = process_raw_wav(wav, sr, 48000, 3)
    wav = _wav_to_spec(wav, 48000)

    model_input = wav.unsqueeze(0)
    output = model(model_input)
    print(output)

    prediction_index = torch.argmax(output, 1).item()
    return LABELS[prediction_index]

demo = gr.Interface(fn=greet, inputs="mic", outputs="text")

demo.launch() 