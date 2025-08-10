import os
import torch
from PIL import Image
from model import CRNN
from utils import LabelEncoder, transform
from inference import predict_plate

# Paths
IMAGE_DIR = "data/images"
MODEL_PATH = "checkpoints/crnn_plate.pth"
OUTPUT_CSV = "batch_results.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and encoder
encoder = LabelEncoder()
model = CRNN(img_height=32, num_classes=len(encoder.alphabet) + 1).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# Batch process all images
results = []

for filename in os.listdir(IMAGE_DIR):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        img_path = os.path.join(IMAGE_DIR, filename)
        expected = os.path.splitext(filename)[0].upper()
        predicted = predict_plate(img_path)

        match = "✅" if predicted == expected else "❌"
        results.append((filename, expected, predicted, match))
        print(f"{filename}: Expected={expected}, Predicted={predicted} {match}")

# Save to CSV
import csv
with open(OUTPUT_CSV, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Filename", "Expected", "Predicted", "Match"])
    writer.writerows(results)

print(f"\n📄 Results saved to {OUTPUT_CSV}")
