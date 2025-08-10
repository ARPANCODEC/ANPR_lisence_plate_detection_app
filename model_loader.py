import torch
import torch.nn as nn
from model import CRNN  # assuming your model class is here

def load_model(model_path):
    model = CRNN(img_height=32, num_channels=1, num_classes=37)

    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model
