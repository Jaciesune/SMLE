import os
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np

class YOLODataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None, input_size=(416, 416)):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform
        self.input_size = input_size
        self.image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Wczytaj obraz
        img_path = os.path.join(self.images_dir, self.image_files[idx])
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Nie można wczytać obrazu: {img_path}")
        
        # Przeskaluj obraz
        img = cv2.resize(img, self.input_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Wczytaj etykiety
        label_path = os.path.join(self.labels_dir, self.image_files[idx].replace('.jpg', '.txt').replace('.png', '.txt'))
        boxes = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    class_id, x, y, w, h = map(float, line.strip().split())
                    boxes.append([class_id, x, y, w, h])
        
        # Konwersja obrazu na tensor
        img = img.astype(np.float32) / 255.0  # Normalizacja do [0, 1]
        
        # Zastosuj transformacje, jeśli istnieją
        if self.transform:
            img = self.transform(img)
        
        boxes = torch.tensor(boxes, dtype=torch.float32)
        return img, boxes