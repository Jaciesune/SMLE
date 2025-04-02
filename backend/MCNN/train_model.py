import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import transforms
from PIL import Image
import os
import time
from model import MCNN
import torch.nn.functional as F
from torch.amp import autocast, GradScaler

def clear_memory():
    torch.cuda.empty_cache()

def generate_density_map(image_size, annotation_path):
    density_map = np.zeros(image_size, dtype=np.float32)
    try:
        with open(annotation_path, 'r') as f:
            points = [list(map(float, line.strip().split()[1:3])) for line in f.readlines()]
        for x, y in points:
            x, y = int(x * image_size[1]), int(y * image_size[0])
            if 0 <= x < image_size[1] and 0 <= y < image_size[0]:  
                density_map[y, x] += 1
        kernel_size = max(5, min(2 * (len(points) // 10 + 1) + 1, 35))
        sigma = kernel_size / 6  
        density_map = np.clip(density_map, 0, None)
        density_map /= (density_map.max() + 1e-6)
        density_map = cv2.GaussianBlur(density_map, (kernel_size, kernel_size), sigma)
        density_map /= (density_map.max() + 1e-6)  
    except Exception as e:
        print(f"Błąd wczytywania anotacji {annotation_path}: {e}")
    return density_map

class ImageDataset(Dataset):
    def __init__(self, image_folder, annotation_folder, transform=None, density_size=(1024, 1024)):
        self.image_folder = image_folder
        self.annotation_folder = annotation_folder
        self.transform = transform
        self.density_size = density_size
        self.images = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png'))]
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        img_name = os.path.join(self.image_folder, self.images[idx])
        annotation_name = os.path.join(self.annotation_folder, self.images[idx].replace('.jpg', '.txt').replace('.png', '.txt'))
        image = cv2.imread(img_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        density_map = generate_density_map((image.shape[0], image.shape[1]), annotation_name)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        density_map = cv2.resize(density_map, self.density_size, interpolation=cv2.INTER_CUBIC)
        image = image.float() / 255.0
        density_map = torch.tensor(density_map, dtype=torch.float32).unsqueeze(0)

        return image, density_map

data_transforms = A.Compose([
    A.Resize(1024, 1024),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.GaussNoise(p=0.2),
    A.RandomGamma(p=0.3),
    ToTensorV2(),
])

train_dataset = ImageDataset("backend/MCNN/dataset/train/images", "backend/MCNN/dataset/train/annotations", transform=data_transforms)
val_dataset = ImageDataset("backend/MCNN/dataset/val/images", "backend/MCNN/dataset/val/annotations", transform=data_transforms)
train_loader = DataLoader(train_dataset, batch_size=3, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=3, shuffle=False, pin_memory=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MCNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.00075)
criterion = nn.L1Loss()

def ssim_loss(pred, target):
    return 1 - F.cosine_similarity(pred.view(pred.shape[0], -1), target.view(target.shape[0], -1)).mean()

def train(model, train_loader, val_loader, optimizer, criterion, device, epochs=5):
    scaler = GradScaler(enabled=True)
    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        running_loss = 0.0
        for i, (inputs, density_maps) in enumerate(train_loader):
            inputs, density_maps = inputs.to(device), density_maps.to(device)
            optimizer.zero_grad()
            with autocast(device_type='cuda'):
                outputs = model(inputs)  
                loss = criterion(outputs, density_maps) + 0.1 * ssim_loss(outputs, density_maps)  
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
            if i % 10 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Batch {i+1}/{len(train_loader)}, Loss: {running_loss/10:.4f}')
                running_loss = 0.0
        validate(model, val_loader, criterion, device)
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1} took {epoch_time:.2f} seconds.")
        remaining_time = epoch_time * (epochs - (epoch + 1))
        print(f"Estimated remaining time: {remaining_time / 60:.2f} minutes.")
        clear_memory()

def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, density_maps in val_loader:
            inputs, density_maps = inputs.to(device), density_maps.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, density_maps) + 0.1 * ssim_loss(outputs, density_maps)  
            val_loss += loss.item()
    print(f'Validation Loss: {val_loss / len(val_loader):.4f}')

train(model, train_loader, val_loader, optimizer, criterion, device, epochs=50)
torch.save(model.state_dict(), 'object_counting_model.pth')
print("Model saved.")