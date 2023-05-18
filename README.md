# VoID: Voice Identifier

Classifier to recognize the identity of a speaker

Project was done as a part of the AISF hackathon (www.aisf.co), VoID was a finalist :D



## Details

### Dataset

Custom dataset with a set of 5-7 pure, unaugmented voices from labeled speakers saying the same phrase (similar to a passcode). In our initial version, we had three voices saying the same phrase ("The quick brown fox jumps over the lazy dog"). In our demo, we used three voices saying their own individual name.

For test and training data, we then created hundreds of augmented copies of each pure voice using augmentation layers (as seen in the augmentation notebook under the `/notebooks/` folder). The types of distortions were added in randomly and were even compounded randomly, augmentations such as clipping, noise, reverb, and etc.

If you would like access to the dataset used in this project, feel free to contact me.

### Model Structure

```
CNNetwork(
  (conv1): Sequential(
    (0): Conv2d(1, 16, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2))
    (1): ReLU()
    (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (conv2): Sequential(
    (0): Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2))
    (1): ReLU()
    (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (conv3): Sequential(
    (0): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2))
    (1): ReLU()
    (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (conv4): Sequential(
    (0): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2))
    (1): ReLU()
    (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (flatten): Flatten(start_dim=1, end_dim=-1)
  (linear): Linear(in_features=21888, out_features=3, bias=True)
  (softmax): Softmax(dim=1)
)
```

## Demo

Watch the demo of our model [here](https://www.loom.com/share/a8cb126af7b64ddaaa67c6f00e23f4e9)!
