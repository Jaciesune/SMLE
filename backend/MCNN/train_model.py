import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from model import MCNN
import torch.nn.functional as F
import time  # Do mierzenia czasu
from torch.amp import autocast, GradScaler  # Poprawione importy dla nowszego API

# Funkcja czyszcząca pamięć GPU
def clear_memory():
    torch.cuda.empty_cache()

def generate_density_map(image_size, annotation_path):
    """Tworzy lepszą mapę gęstości na podstawie anotacji."""
    density_map = np.zeros(image_size, dtype=np.float32)

    try:
        with open(annotation_path, 'r') as f:
            points = [list(map(float, line.strip().split()[1:3])) for line in f.readlines()]

        for x, y in points:
            x = int(x * image_size[1])  # Skalowanie do wymiarów obrazu
            y = int(y * image_size[0])

            if 0 <= x < image_size[1] and 0 <= y < image_size[0]:  
                density_map[y, x] += 1  # Sumujemy wartości zamiast nadpisywać
        
        # 🔥 Dynamiczny rozmiar filtra Gaussa (dopasowany do liczby punktów)
        num_objects = len(points)
        kernel_size = max(5, min(2 * (num_objects // 10 + 1) + 1, 35))  # Mniejszy zakres niż wcześniej
        sigma = kernel_size / 6  

        # 🛠 Zabezpieczenie przed wartościami ujemnymi
        density_map = np.clip(density_map, 0, None)

        # 🌀 Lepsza normalizacja przed Gaussiannym Blur
        density_map /= (density_map.max() + 1e-6)  # Unikanie dzielenia przez 0
        density_map = cv2.GaussianBlur(density_map, (kernel_size, kernel_size), sigma)

        # 🔄 Finalna normalizacja po filtracji
        density_map = np.clip(density_map, 0, None)  
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

        image = Image.open(img_name).convert('RGB')
        density_map = generate_density_map((image.height, image.width), annotation_name)

        if self.transform:
            image = self.transform(image)
        
        # 🏆 **Zwiększamy rozdzielczość mapy gęstości do (1024, 1024)**
        density_map = cv2.resize(density_map, self.density_size, interpolation=cv2.INTER_CUBIC)
        density_map = torch.tensor(density_map, dtype=torch.float32).unsqueeze(0)

        return image, density_map


# **🔹 Przekształcenia**
transform = transforms.Compose([
    transforms.Resize((1024, 1024)),  # Zmniejszamy rozdzielczość!
    transforms.ToTensor(),
])

# **🔹 Ścieżki do danych**
train_image_folder = 'dataset/train/images'
train_annotation_folder = 'dataset/train/annotations'
val_image_folder = 'dataset/val/images'
val_annotation_folder = 'dataset/val/annotations'

# **🔹 Datasety**
train_dataset = ImageDataset(train_image_folder, train_annotation_folder, transform=transform)
val_dataset = ImageDataset(val_image_folder, val_annotation_folder, transform=transform)

# **🔹 DataLoadery**
train_loader = DataLoader(train_dataset, batch_size=3, shuffle=True, pin_memory=True)  # Zmniejszony batch_size
val_loader = DataLoader(val_dataset, batch_size=3, shuffle=False, pin_memory=True)

# **🔹 Model**
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Przełączenie na CPU jeśli GPU nie jest dostępne

torch.cuda.set_per_process_memory_fraction(0.8, device=0)  # Ogranicz do 80% dostępnej pamięci GPU

model = MCNN().to(device)

# **🔹 Optymalizator i strata**
optimizer = optim.Adam(model.parameters(), lr=0.00075)  # Jeszcze mniejsze LR dla lepszego dopasowania
criterion = nn.L1Loss()  # MAE dalej najlepsze

# 🏆 **Dodatkowa strata SSIM dla poprawy jakości mapy gęstości**
def ssim_loss(pred, target):
    return 1 - F.cosine_similarity(pred.view(pred.shape[0], -1), target.view(target.shape[0], -1)).mean()

# **🔹 Funkcja treningowa**
def train(model, train_loader, val_loader, optimizer, criterion, device, epochs=5):
    scaler = GradScaler(enabled=True)  # Używamy GradScaler dla mieszanej precyzji

    for epoch in range(epochs):
        start_time = time.time()  # Start pomiaru czasu
        model.train()
        running_loss = 0.0

        for i, (inputs, density_maps) in enumerate(train_loader):
            inputs, density_maps = inputs.to(device), density_maps.to(device)

            optimizer.zero_grad()

            # Użycie autocast do treningu w mieszanej precyzji
            with autocast(device_type='cuda'):  # Poprawiona obsługa dla nowszego API
                outputs = model(inputs)  
                loss = criterion(outputs, density_maps) + 0.1 * ssim_loss(outputs, density_maps)  

            # Normalny backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

            if i % 10 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Batch {i+1}/{len(train_loader)}, Loss: {running_loss/10:.4f}')
                running_loss = 0.0

        # 🏆 Walidacja po każdej epoce
        validate(model, val_loader, criterion, device)

        # **⏳ Czas trwania epoki**
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1} took {epoch_time:.2f} seconds.")

        # **Przewidywany czas na pozostałe epoki**
        remaining_time = epoch_time * (epochs - (epoch + 1))
        print(f"Estimated remaining time: {remaining_time / 60:.2f} minutes.")

        # Czyszczenie pamięci po każdej epoce
        clear_memory()

# **🔹 Funkcja walidacji**
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


# **🚀 Start treningu**
train(model, train_loader, val_loader, optimizer, criterion, device, epochs=10)

# **📌 Zapisz model**
torch.save(model.state_dict(), 'object_counting_model.pth')
print("Model saved.")
