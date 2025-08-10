import torch
from torch.utils.data import DataLoader
from dataset_loader import PlateDataset
from model_loader import load_model
from decoder import decode_predictions
import numpy as np
from torch.nn.functional import log_softmax

# Constants
CSV_PATH = r"D:\1. ANPR_CRNN(IMAGES)\data\labels.csv"
MODEL_PATH = r"D:\1. ANPR_CRNN(IMAGES)\checkpoints\crnn_plate.pth"
BATCH_SIZE = 1

# Load data
dataset = PlateDataset(CSV_PATH)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# Load model
model = load_model(MODEL_PATH)

correct = 0
total = 0

for images, labels in dataloader:
    with torch.no_grad():
        output = model(images)
        output = log_softmax(output, 2)
        output = output.permute(1, 0, 2)
        _, preds = output.max(2)
        preds = preds.transpose(1, 0).contiguous().numpy()
        
        decoded_preds = decode_predictions(preds)
        
        for pred, label in zip(decoded_preds, labels):
            total += 1
            if pred.strip() == label.strip():
                correct += 1

print(f"\n✅ Accuracy: {(correct / total) * 100:.2f}% ({correct}/{total})")
