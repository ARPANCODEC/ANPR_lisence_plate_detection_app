# inference.py - Predict number plate from image using trained CRNN

import torch
from torchvision import transforms
from PIL import Image
from model import CRNN
from utils import LabelEncoder
import sys
import os

# ==== CONFIG ====
IMG_HEIGHT = 32
IMG_WIDTH = 128
MODEL_PATH = "checkpoints/crnn_plate.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== LOAD MODEL ====
NUM_CLASSES = len("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ") + 1  # +1 for CTC blank
model = CRNN(img_height=IMG_HEIGHT, num_channels=1, num_classes=NUM_CLASSES).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ==== ENCODER ====
encoder = LabelEncoder()

# ==== TRANSFORM ====
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ==== PREDICT FUNCTION ====
def predict_plate(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(image)
        preds = torch.argmax(logits, dim=2).permute(1, 0)  # [batch, time]

        pred_indices = preds[0].detach().cpu().tolist()

        # CTC decoding: remove duplicates and blank index (0)
        decoded = []
        prev = -1
        for idx in pred_indices:
            if idx != prev and idx != 0:
                decoded.append(idx)
            prev = idx

        pred_text = encoder.decode(decoded)

    return pred_text

# ==== RUN FROM CLI ====
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python inference.py <image_path>")
        sys.exit(1)

    img_path = sys.argv[1]
    if not os.path.exists(img_path):
        print("Error: Image file not found.")
        sys.exit(1)

    prediction = predict_plate(img_path)
    print(f"Predicted Plate: {prediction}")
