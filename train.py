# ===================================================
# train.py - Train a CRNN model on number plate images
# ===================================================
# Description : Uses image filenames as ground truth
# Model       : Convolutional Recurrent Neural Network (CRNN)
# Dataset     : Images in `data/images/` named after plate numbers (e.g. WB12BJ2370.jpg)
# Author      : [Your Name or Team]
# ===================================================

# ==== STANDARD LIBRARIES ====
import os

# ==== THIRD-PARTY LIBRARIES ====
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# ==== LOCAL MODULES ====
from model import CRNN
from utils import PlateDataset, LabelEncoder

# ==== CONFIGURATION ====
IMG_HEIGHT = 32
NUM_CHANNELS = 1
BATCH_SIZE = 4
EPOCHS = 1000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== DATA PREPARATION ====
IMG_DIR = "data/images"
encoder = LabelEncoder()
NUM_CLASSES = len(encoder.alphabet) + 1  # +1 for CTC blank

dataset = PlateDataset(image_dir=IMG_DIR, encoder=encoder)

def collate_fn(batch):
    images, labels, label_text = zip(*batch)
    images = torch.stack(images)
    label_lens = torch.tensor([len(l) for l in labels], dtype=torch.long)
    labels = torch.cat([torch.tensor(label, dtype=torch.long) for label in labels])
    return images, labels, label_lens

dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

# ==== MODEL INITIALIZATION ====
model = CRNN(img_height=IMG_HEIGHT, num_channels=NUM_CHANNELS, num_classes=NUM_CLASSES).to(DEVICE)
criterion = nn.CTCLoss(blank=0, zero_infinity=True)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ==== TRAINING LOOP ====
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for images, labels, label_lens in dataloader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        logits = model(images)  # (T, N, C)
        T, N, _ = logits.size()
        input_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.long).to(DEVICE)

        loss = criterion(logits, labels, input_lengths, label_lens.to(DEVICE))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {total_loss:.4f}")

# ==== SAVE TRAINED MODEL ====
os.makedirs("checkpoints", exist_ok=True)
torch.save(model.state_dict(), "checkpoints/crnn_plate.pth")
print("✅ Model trained and saved.")
