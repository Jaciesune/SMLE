"""
Skrypt do trenowania modelu MCNN (Multi-Column CNN) do zliczania obiektów
poprzez estymację map gęstości.

Implementuje pełny proces treningu z obsługą formatu COCO, zaawansowaną augmentacją
danych, mixed precision training i adaptacyjnym generowaniem map gęstości.
"""

import argparse
import logging
import os
import json
import sys
import time
import torch
import cv2
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from datetime import datetime
from torch import optim
from torch.amp import autocast, GradScaler
from model import MCNN

# Logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Transformacje treningowe
train_transforms = A.Compose([
    A.RandomRotate90(p=0.5),
    A.Rotate(limit=15, p=0.5),
    A.RandomResizedCrop(size=(1024, 1024), scale=(0.8, 1.0), ratio=(0.9, 1.1), p=0.5),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.GaussNoise(p=0.2),
    A.RandomGamma(p=0.3),
    A.Resize(height=1024, width=1024),
    ToTensorV2(),
])

# Transformacje walidacyjne (brak augmentacji)
val_transforms = A.Compose([
    A.Resize(height=1024, width=1024),
    ToTensorV2(),
])

def clear_memory():
    """Zwalnia pamięć CUDA."""
    torch.cuda.empty_cache()

def generate_density_map(image_size, annotations):
    """
    Generuje mapę gęstości na podstawie adnotacji obiektów.
    
    Proces:
    1. Umieszcza pojedyncze punkty w środkach ramek (bbox) obiektów
    2. Rozmywa punkty filtrem Gaussa z adaptacyjnym rozmiarem kernela
    3. Normalizuje mapę gęstości
    
    Args:
        image_size (tuple): Rozmiar obrazu (wysokość, szerokość)
        annotations (list): Lista adnotacji obiektów w formacie COCO
        
    Returns:
        numpy.ndarray: Mapa gęstości jako tablica 2D
    """
    density_map = np.zeros(image_size, dtype=np.float32)
    try:
        for annotation in annotations:
            x = annotation['bbox'][0] + annotation['bbox'][2] / 2
            y = annotation['bbox'][1] + annotation['bbox'][3] / 2
            x = int(x)
            y = int(y)
            if 0 <= x < image_size[1] and 0 <= y < image_size[0]:
                density_map[y, x] += 1
        kernel_size = max(5, min(2 * (len(annotations) // 10 + 1) + 1, 35))
        sigma = kernel_size / 6
        density_map = np.clip(density_map, 0, None)
        if density_map.max() > 0:
            density_map /= (density_map.max() + 1e-6)
        density_map = cv2.GaussianBlur(density_map, (kernel_size, kernel_size), sigma)
        if density_map.max() > 0:
            density_map /= (density_map.max() + 1e-6)
    except Exception as e:
        logger.error(f"Błąd generowania mapy gęstości: {e}")
    return density_map

class ImageDataset(Dataset):
    """
    Dataset do ładowania obrazów i generowania map gęstości z adnotacji COCO.
    
    Obsługuje:
    - Wczytywanie obrazów z katalogu
    - Parsowanie adnotacji w formacie COCO
    - Generowanie map gęstości obiektów
    - Augmentację danych
    """
    def __init__(self, image_folder, annotation_path, transform=None, density_size=(1024, 1024), num_augmentations=1):
        self.image_folder = image_folder
        self.annotation_path = annotation_path
        self.transform = transform
        self.density_size = density_size
        self.num_augmentations = num_augmentations
        self.images = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png'))]

        if not os.path.isfile(self.annotation_path):
            raise FileNotFoundError(f"Nie znaleziono pliku z adnotacjami: {self.annotation_path}")

        with open(self.annotation_path, 'r') as f:
            coco_data = json.load(f)

        # Tworzenie słowników mapujących nazwy plików na ID i ID na adnotacje
        self.file_to_id = {img['file_name']: img['id'] for img in coco_data['images']}
        self.id_to_annotations = {}
        for ann in coco_data['annotations']:
            image_id = ann['image_id']
            self.id_to_annotations.setdefault(image_id, []).append(ann)

    def __len__(self):
        return len(self.images) * self.num_augmentations

    def __getitem__(self, idx):
        """
        Pobiera obraz i odpowiadającą mu mapę gęstości.
        
        Proces:
        1. Wczytuje obraz z dysku
        2. Odszukuje adnotacje dla obrazu
        3. Generuje mapę gęstości
        4. Stosuje transformacje (jeśli podane)
        
        Returns:
            tuple: (obraz jako tensor, mapa gęstości jako tensor)
        """
        base_idx = idx % len(self.images)
        file_name = self.images[base_idx]
        img_path = os.path.join(self.image_folder, file_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_id = self.file_to_id.get(file_name, None)
        annotations = self.id_to_annotations.get(image_id, []) if image_id is not None else []
        density_map = generate_density_map((image.shape[0], image.shape[1]), annotations)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        density_map = cv2.resize(density_map, self.density_size, interpolation=cv2.INTER_CUBIC)
        image = image.float() / 255.0
        density_map = torch.tensor(density_map, dtype=torch.float32).unsqueeze(0)

        return image, density_map

def ssim_loss(pred, target):
    """
    Oblicza loss bazujący na podobieństwie strukturalnym poprzez podobieństwo kosinusowe.
    
    Args:
        pred (torch.Tensor): Predykcja modelu
        target (torch.Tensor): Wartość oczekiwana (ground truth)
        
    Returns:
        torch.Tensor: Wartość funkcji straty
    """
    return 1 - F.cosine_similarity(pred.view(pred.shape[0], -1), target.view(target.shape[0], -1)).mean()

def train_model(args):
    """
    Główna funkcja trenująca model MCNN.
    
    Implementuje:
    - Konfigurację urządzenia (CPU/GPU)
    - Inicjalizację/wczytywanie modelu
    - Przygotowanie danych treningowych i walidacyjnych
    - Pętlę treningową z walidacją
    - Zapisywanie checkpointów modelu
    - Obsługę błędów i awaryjne zapisywanie
    
    Args:
        args: Argumenty wywołania skryptu
    """
    logger.info("Rozpoczęcie treningu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Czy CUDA dostępna: {torch.cuda.is_available()}")

    nazwa_modelu = args.model_name
    os.makedirs(f"/app/backend/logs/train/{nazwa_modelu}", exist_ok=True)
    os.makedirs(f"/app/backend/logs/val/{nazwa_modelu}", exist_ok=True)
    os.makedirs("/app/backend/MCNN/models", exist_ok=True)

    train_dir = args.train_dir
    val_dir = args.val_dir
    coco_train_path = args.coco_train_path
    coco_val_path = args.coco_gt_path

    logger.info(f"Train path: {coco_train_path}")
    logger.info(f"Val path: {coco_val_path}")

    # Inicjalizacja modelu
    model = MCNN()
    if args.model_checkpoint:
        model.load_state_dict(torch.load(args.model_checkpoint, map_location=device))
        logger.info(f"Załadowano checkpoint: {args.model_checkpoint}")
    model.to(device)

    # Tworzenie datasetów i dataloaderów
    train_dataset = ImageDataset(
        os.path.join(train_dir, "images"), coco_train_path,
        transform=train_transforms, num_augmentations=args.num_augmentations
    )
    val_dataset = ImageDataset(
        os.path.join(val_dir, "images"), coco_val_path,
        transform=val_transforms, num_augmentations=1
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size or 3, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size or 3, shuffle=False, pin_memory=True)

    logger.info(f"Liczba obrazów treningowych: {len(train_dataset)}")
    logger.info(f"Liczba batchy treningowych: {len(train_loader)}")

    # Konfiguracja optymalizatora i funkcji straty
    lr = args.lr or 0.00075
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.L1Loss()
    scaler = GradScaler(enabled=True)  # Do treningu z mixed precision

    def validate(model, val_loader, criterion, device):
        """
        Waliduje model na zbiorze walidacyjnym.
        
        Args:
            model: Model do walidacji
            val_loader: DataLoader z danymi walidacyjnymi
            criterion: Funkcja straty
            device: Urządzenie (CPU/GPU)
        """
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, density_maps in val_loader:
                inputs, density_maps = inputs.to(device), density_maps.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, density_maps) + 0.1 * ssim_loss(outputs, density_maps)
                val_loss += loss.item()
        logger.info(f'Validation Loss: {val_loss / len(val_loader):.4f}')

    try:
        # Główna pętla treningowa
        for epoch in range(args.epochs):
            start_time = time.time()
            model.train()
            running_loss = 0.0
            for i, (inputs, density_maps) in enumerate(train_loader):
                inputs, density_maps = inputs.to(device), density_maps.to(device)
                optimizer.zero_grad()
                
                # Użycie mixed precision
                with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                    outputs = model(inputs)
                    # Kombinowana funkcja straty: L1 + SSIM
                    loss = criterion(outputs, density_maps) + 0.1 * ssim_loss(outputs, density_maps)
                
                # Skalowanie gradientu, backward pass i aktualizacja wag
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                running_loss += loss.item()

                if i % 10 == 0:
                    logger.info(f'Epoch {epoch+1}/{args.epochs}, Batch {i+1}/{len(train_loader)}, Loss: {running_loss/10:.4f}')
                    running_loss = 0.0

            # Walidacja po każdej epoce
            validate(model, val_loader, criterion, device)
            epoch_time = time.time() - start_time
            logger.info(f"Epoch {epoch+1} took {epoch_time:.2f} seconds.")

            # Zapisywanie checkpointów co 10 epok
            if (epoch + 1) % 10 == 0:
                save_path = f"/app/backend/MCNN/models/{nazwa_modelu}_epoch_{epoch+1}_checkpoint.pth"
                torch.save(model.state_dict(), save_path)
                logger.info(f"Zapisano checkpoint: {save_path}")

            clear_memory()

        # Zapisanie finalnego modelu
        model_path = f"/app/backend/MCNN/models/{nazwa_modelu}_checkpoint.pth"
        torch.save(model.state_dict(), model_path)
        logger.info(f"Model zapisany jako: {model_path}")

    except Exception as e:
        # Obsługa błędów z awaryjnym zapisem modelu
        logger.exception("Błąd podczas treningu! Zapisuję awaryjny checkpoint...")
        emergency_path = f"/app/backend/MCNN/models/{nazwa_modelu}_emergency_checkpoint.pth"
        torch.save(model.state_dict(), emergency_path)
        logger.info(f"Awaryjny model zapisany jako: {emergency_path}")

# === GŁÓWNE WYWOŁANIE ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', required=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.00075)
    parser.add_argument('--model_name', default=None)
    parser.add_argument('--model_checkpoint', default=None)
    parser.add_argument('--batch_size', type=int, default=3)
    parser.add_argument('--coco_train_path', required=True)
    parser.add_argument('--coco_gt_path', required=True)
    parser.add_argument('--val_dir', required=True)
    parser.add_argument('--num_augmentations', type=int, default=1, help="Liczba augmentacji danych treningowych")
    args = parser.parse_args()

    logger.debug("Arguments received: %s", sys.argv)
    logger.debug("Checking if coco_train_path exists: %s", os.path.exists(args.coco_train_path))
    logger.debug("Checking if coco_gt_path exists: %s", os.path.exists(args.coco_gt_path))

    train_model(args)