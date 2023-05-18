# 
import sys
sys.path.append('..')

import os
from fastapi import FastAPI

# torch
import torch

# utils
from preprocess import process_from_filename, process_from_url, process_raw_wav
from cnn import CNNetwork

# load model
model = CNNetwork()
state_dict = torch.load("../models/aisf/void_20230517_113634.pth")
model.load_state_dict(state_dict)

# TODO: update to grabbing labels stored on model
LABELS = ["shafqat", "aman", "jake"]

print(f"Model loaded! \n {model}")

app = FastAPI()

@app.get("/")
async def root():
    return { "message": "Hello World" }

@app.get("/urlpredict")
def url_predict(url: str):
    wav = process_from_url(url)
    
    model_prediction = model_predict(wav)
    return {
        "message": "Voice Identified!",
        "data": model_prediction,
    }

@app.put("/predict")
def predict(wav):
    print(f"wav {wav}")
    # return wav
    wav = process_raw_wav(wav)
    model_prediction = model_predict(wav)

    return {
        "message": "Voice Identified!",
        "data": model_prediction,
    }

def model_predict(wav):
    model_input = wav.unsqueeze(0)
    output = model(model_input)
    prediction_index = torch.argmax(output, 1).item()
    output = output.detach().cpu().numpy()[0]

    return {
        "prediction_index": prediction_index,
        "labels": LABELS,
        "prediction_label": LABELS[prediction_index],
        "prediction_output": output.tolist(),
    }