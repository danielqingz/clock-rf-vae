import torch
from model import RF_VAE2  # or your own model


model = RF_VAE2()  # e.g. define your custom encoder-decoder
dummy_input = torch.randn(1, 3, 224, 224)  # batch=1, 3-channel, 224x224 image
output = model(dummy_input)
for i, out_i in enumerate(output):
    print(f"Item {i} shape: ", out_i.shape)
