import torch
from torch.utils.data import Dataset
import pandas as pd
import os
import cv2
from PIL import Image

class PlateDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.data.iloc[idx, 0]
        label = self.data.iloc[idx, 1]

        # Make sure image path is absolute or fix as needed
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise ValueError(f"cv2 failed to load image: {image_path}")

        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)

        return image, label
