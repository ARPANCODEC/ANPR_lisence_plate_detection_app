import string
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import os

# Define the character set (A-Z, 0-9)
ALPHABET = string.ascii_uppercase + string.digits

# Label Encoder
class LabelEncoder:
    def __init__(self, alphabet=ALPHABET):
        self.alphabet = alphabet
        self.char2idx = {char: idx + 1 for idx, char in enumerate(alphabet)}  # 0 is reserved for blank (CTC)
        self.idx2char = {idx + 1: char for idx, char in enumerate(alphabet)}
        self.blank_idx = 0

    def encode(self, text):
        return [self.char2idx[char] for char in text if char in self.char2idx]

    def decode(self, indices):
        return ''.join([self.idx2char[idx] for idx in indices if idx in self.idx2char])

    def decode_sequence(self, seq):
        result = []
        prev = -1
        for idx in seq:
            if idx != self.blank_idx and idx != prev:
                result.append(self.idx2char.get(idx, ''))
            prev = idx
        return ''.join(result)

# Image preprocessing
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((32, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Custom Dataset for training
class PlateDataset(Dataset):
    def __init__(self, image_dir, encoder):
        self.image_dir = image_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
        self.encoder = encoder

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        label_text = os.path.splitext(img_name)[0].upper().replace(" ", "")  # Remove spaces
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("L")
        image = transform(image)
        label = self.encoder.encode(label_text)
        return image, label, label_text
