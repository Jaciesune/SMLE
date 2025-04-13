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

# Transformacje
data_transforms = A.Compose([
    A.Resize(1024, 1024),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.GaussNoise(p=0.2),
    A.RandomGamma(p=0.3),
    ToTensorV2(),
])

def clear_memory():
    torch.cuda.empty_cache()

def generate_density_map(image_size, annotations):
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
    def __init__(self, image_folder, annotation_path, transform=None, density_size=(1024, 1024)):
        self.image_folder = image_folder
        self.annotation_path = annotation_path
        self.transform = transform
        self.density_size = density_size
        self.images = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png'))]

        if not os.path.isfile(self.annotation_path):
            raise FileNotFoundError(f"Nie znaleziono pliku z adnotacjami: {self.annotation_path}")

        with open(self.annotation_path, 'r') as f:
            coco_data = json.load(f)

        self.file_to_id = {img['file_name']: img['id'] for img in coco_data['images']}
        self.id_to_annotations = {}
        for ann in coco_data['annotations']:
            image_id = ann['image_id']
            self.id_to_annotations.setdefault(image_id, []).append(ann)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        file_name = self.images[idx]
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
    return 1 - F.cosine_similarity(pred.view(pred.shape[0], -1), target.view(target.shape[0], -1)).mean()

def train_model(args):
    logger.info("Rozpoczęcie treningu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Czy CUDA dostępna: {torch.cuda.is_available()}")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_name = args.model_name or f"train_{timestamp}"
    nazwa_modelu = f"{model_name}_{timestamp}"

    os.makedirs(f"/app/backend/logs/train/{nazwa_modelu}", exist_ok=True)
    os.makedirs(f"/app/backend/logs/val/{nazwa_modelu}", exist_ok=True)
    os.makedirs("/app/backend/MCNN/models", exist_ok=True)

    train_dir = args.train_dir
    val_dir = os.path.join(train_dir, "..", "val")
    coco_train_path = args.coco_train_path
    coco_val_path = args.coco_gt_path

    logger.info(f"Train path: {coco_train_path}")
    logger.info(f"Val path: {coco_val_path}")

    model = MCNN()
    if args.model_checkpoint:
        model.load_state_dict(torch.load(args.model_checkpoint, map_location=device))
        logger.info(f"Załadowano checkpoint: {args.model_checkpoint}")
    model.to(device)

    train_dataset = ImageDataset(os.path.join(train_dir, "images"), coco_train_path, transform=data_transforms)
    val_dataset = ImageDataset(os.path.join('/app/backend/MCNN/dataset', 'val', 'images'), coco_val_path, transform=data_transforms)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size or 3, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size or 3, shuffle=False, pin_memory=True)

    logger.info(f"Liczba obrazów treningowych: {len(train_dataset)}")
    logger.info(f"Liczba batchy treningowych: {len(train_loader)}")

    lr = args.lr or 0.00075
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.L1Loss()
    scaler = GradScaler(enabled=True)

    def validate(model, val_loader, criterion, device):
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
        for epoch in range(args.epochs):
            start_time = time.time()
            model.train()
            running_loss = 0.0
            for i, (inputs, density_maps) in enumerate(train_loader):
                inputs, density_maps = inputs.to(device), density_maps.to(device)
                optimizer.zero_grad()
                with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                    outputs = model(inputs)
                    loss = criterion(outputs, density_maps) + 0.1 * ssim_loss(outputs, density_maps)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                running_loss += loss.item()

                if i % 10 == 0:
                    logger.info(f'Epoch {epoch+1}/{args.epochs}, Batch {i+1}/{len(train_loader)}, Loss: {running_loss/10:.4f}')
                    running_loss = 0.0

            validate(model, val_loader, criterion, device)
            epoch_time = time.time() - start_time
            logger.info(f"Epoch {epoch+1} took {epoch_time:.2f} seconds.")

            if (epoch + 1) % 10 == 0:
                save_path = f"/app/backend/MCNN/models/{nazwa_modelu}_epoch_{epoch+1}_checkpoint.pth"
                torch.save(model.state_dict(), save_path)
                logger.info(f"Zapisano checkpoint: {save_path}")

            clear_memory()

        model_path = f"/app/backend/MCNN/models/{nazwa_modelu}_final_checkpoint.pth"
        torch.save(model.state_dict(), model_path)
        logger.info(f"Model zapisany jako: {model_path}")

    except Exception as e:
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
    parser.add_argument('--num_augmentations', type=int, default=1, help="Liczba augmentacji (na razie nieużywana)")
    args = parser.parse_args()

    logger.debug("Arguments received: %s", sys.argv)
    logger.debug("Checking if coco_train_path exists: %s", os.path.exists(args.coco_train_path))
    logger.debug("Checking if coco_gt_path exists: %s", os.path.exists(args.coco_gt_path))

    train_model(args)
