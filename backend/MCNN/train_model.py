import os
# Wyłączenie ostrzeżeń Albumentations przed importem
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

import argparse
import logging
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
from torch.amp import autocast, GradScaler  # Nowy API dla PyTorch 2.7.0
from model import MCNN

# Logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Wyłączenie debugowania dla Pillow
logging.getLogger('PIL').setLevel(logging.INFO)

# Transformacje dla obrazów 1024x1024
data_transforms = A.Compose([
    A.Resize(1024, 1024),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.GaussNoise(p=0.2),
    A.RandomGamma(p=0.3),
    ToTensorV2(),
])

def clear_memory():
    """Bezpieczne czyszczenie pamięci VRAM."""
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        except RuntimeError as e:
            logger.warning(f"Błąd podczas czyszczenia pamięci VRAM: {e}. Kontynuuję bez czyszczenia.")
        finally:
            torch.cuda.synchronize()

def get_vram_available():
    """Sprawdzenie dostępnej pamięci VRAM (tylko fizyczna pamięć GPU)."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        total_vram = torch.cuda.get_device_properties(0).total_memory
        allocated_vram = torch.cuda.memory_allocated()
        reserved_vram = torch.cuda.memory_reserved()
        # Używamy bardziej konserwatywnego podejścia: bierzemy maksimum z alokowanej i zarezerwowanej pamięci
        used_vram = max(allocated_vram, reserved_vram)
        available_vram = total_vram - used_vram
        # Dodajemy większy margines bezpieczeństwa (20% całkowitej VRAM)
        safety_margin = total_vram * 0.2
        available_vram = max(0, available_vram - safety_margin)
        return available_vram
    return float('inf')

def estimate_memory_per_batch(image_size=(1024, 1024), channels=3, density_size=(1024, 1024), batch_size=1):
    """Estymacja zapotrzebowania na VRAM na batch."""
    # Rozmiar obrazu wejściowego (float32, 4 bajty na element)
    image_memory = image_size[0] * image_size[1] * channels * 4 * batch_size  # w bajtach
    # Rozmiar mapy gęstości (float32, 1 kanał)
    density_memory = density_size[0] * density_size[1] * 1 * 4 * batch_size  # w bajtach
    # Szacunkowe zużycie pamięci przez model (aktywacje, gradienty, wagi)
    # Zwiększamy mnożnik do 25x, aby uwzględnić większe zużycie przez MCNN
    model_memory = (image_memory + density_memory) * 25
    # Dodajemy bufor CUDA (szacunkowo 150 MB na batch)
    cuda_buffer = 150 * 1024 * 1024 * batch_size  # 150 MB w bajtach
    # Całkowite zużycie
    total_memory = image_memory + density_memory + model_memory + cuda_buffer
    return total_memory / 1e9  # w GB

def find_optimal_batch_size(model, dataset, device, criterion, optimizer, min_batch_size=1, max_batch_size=16):
    """Automatyczne dopasowanie rozmiaru partii z marginesem bezpieczeństwa dla VRAM."""
    logger.info("Rozpoczynanie wyszukiwania optymalnego batch_size...")

    # Estymacja zapotrzebowania na pamięć dla batch_size=1
    memory_per_batch = estimate_memory_per_batch()
    total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9  # w GB
    logger.info(f"Estymowane zapotrzebowanie na VRAM na batch: {memory_per_batch:.2f} GB")
    logger.info(f"Całkowita VRAM GPU: {total_vram:.2f} GB")

    # Obliczenie teoretycznego maksymalnego batch_size na podstawie dostępnej VRAM
    theoretical_max_batch = int(total_vram // memory_per_batch)
    theoretical_max_batch = min(theoretical_max_batch, max_batch_size)
    theoretical_max_batch = max(theoretical_max_batch, min_batch_size)
    logger.info(f"Teoretyczny maksymalny batch_size na podstawie VRAM: {theoretical_max_batch}")

    # Zaczynamy od min_batch_size i zwiększamy
    batch_sizes = list(range(min_batch_size, theoretical_max_batch + 1))
    if not batch_sizes:
        batch_sizes = [min_batch_size]

    model.train()
    scaler = GradScaler('cuda', enabled=torch.cuda.is_available())
    optimal_batch_size = min_batch_size
    vram_threshold = 0.3  # Minimalny próg VRAM w GB (300 MB)
    # Dodajemy konserwatywne ograniczenie maksymalnego batch_size
    conservative_max_batch = min(theoretical_max_batch, 4)  # Ograniczamy do 4, aby być bezpiecznym

    for batch_size in batch_sizes:
        if batch_size > conservative_max_batch:
            logger.warning(f"Batch_size {batch_size} przekracza konserwatywne ograniczenie ({conservative_max_batch}). Przerywam testowanie.")
            break

        available_vram = get_vram_available() / 1e9  # w GB
        logger.info(f"Testowanie batch_size: {batch_size}, dostępna VRAM: {available_vram:.2f} GB")
        
        # Przerwanie, jeśli dostępna VRAM jest poniżej progu
        if available_vram < vram_threshold:
            logger.warning(f"Dostępna VRAM ({available_vram:.2f} GB) poniżej progu ({vram_threshold} GB). Przerywam testowanie.")
            break

        try:
            temp_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
            for inputs, density_maps in temp_loader:
                inputs, density_maps = inputs.to(device), density_maps.to(device)
                optimizer.zero_grad()
                with autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                    outputs = model(inputs)
                    if outputs.shape[2:] != density_maps.shape[2:]:
                        outputs = F.interpolate(outputs, size=density_maps.shape[2:], mode='bilinear', align_corners=False)
                    loss = criterion(outputs, density_maps) + 0.1 * ssim_loss(outputs, density_maps)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                del inputs, density_maps, outputs, loss
                clear_memory()
                break
            logger.info(f"Batch_size {batch_size} jest poprawny!")
            optimal_batch_size = batch_size
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning(f"Batch_size {batch_size} zbyt duży: brak VRAM. Przerywam testowanie.")
                clear_memory()
                break
            else:
                logger.error(f"Błąd przy testowaniu batch_size {batch_size}: {e}")
                raise
        except Exception as e:
            logger.error(f"Błąd przy testowaniu batch_size {batch_size}: {e}")
            raise
        finally:
            clear_memory()

    logger.info(f"Wybrano optymalny batch_size: {optimal_batch_size}")
    return optimal_batch_size

def generate_density_map(image_size, annotations):
    """Generowanie mapy gęstości."""
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
    """Obliczenie straty SSIM."""
    return 1 - F.cosine_similarity(pred.view(pred.shape[0], -1), target.view(target.shape[0], -1)).mean()

def train_model(args):
    logger.info("Rozpoczęcie treningu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Czy CUDA dostępna: {torch.cuda.is_available()}")
    logger.info(f"Wersja PyTorch: {torch.__version__}")
    logger.info(f"Całkowita VRAM GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_name = args.model_name or f"train_{timestamp}"
    nazwa_modelu = f"{model_name}_{timestamp}"

    os.makedirs(f"/app/backend/logs/train/{nazwa_modelu}", exist_ok=True)
    os.makedirs(f"/app/backend/logs/val/{nazwa_modelu}", exist_ok=True)
    os.makedirs("/app/backend/MCNN/models", exist_ok=True)

    train_dir = args.train_dir
    val_dir = args.val_dir
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
    val_dataset = ImageDataset(os.path.join(val_dir, "images"), coco_val_path, transform=data_transforms)

    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr or 0.00075)
    batch_size = args.batch_size or find_optimal_batch_size(model, train_dataset, device, criterion, optimizer, min_batch_size=1, max_batch_size=16)
    logger.info(f"Używany batch_size: {batch_size}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    logger.info(f"Liczba obrazów treningowych: {len(train_dataset)}")
    logger.info(f"Liczba batchy treningowych: {len(train_loader)}")

    lr = args.lr or 0.00075
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.L1Loss()
    scaler = GradScaler('cuda', enabled=torch.cuda.is_available())

    def validate(model, val_loader, criterion, device):
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, density_maps in val_loader:
                inputs, density_maps = inputs.to(device), density_maps.to(device)
                with autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                    outputs = model(inputs)
                    if outputs.shape[2:] != density_maps.shape[2:]:
                        outputs = F.interpolate(outputs, size=density_maps.shape[2:], mode='bilinear', align_corners=False)
                    loss = criterion(outputs, density_maps) + 0.1 * ssim_loss(outputs, density_maps)
                    val_loss += loss.item()
                del inputs, density_maps, outputs, loss
                clear_memory()
        avg_val_loss = val_loss / len(val_loader)
        logger.info(f'Validation Loss: {avg_val_loss:.4f}')
        return avg_val_loss

    try:
        for epoch in range(args.epochs):
            start_time = time.time()
            model.train()
            running_loss = 0.0
            retry_batch_size = batch_size
            while retry_batch_size >= 1:
                try:
                    train_loader = DataLoader(train_dataset, batch_size=retry_batch_size, shuffle=True, pin_memory=True)
                    for i, (inputs, density_maps) in enumerate(train_loader):
                        inputs, density_maps = inputs.to(device), density_maps.to(device)
                        optimizer.zero_grad()
                        with autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                            outputs = model(inputs)
                            if outputs.shape[2:] != density_maps.shape[2:]:
                                outputs = F.interpolate(outputs, size=density_maps.shape[2:], mode='bilinear', align_corners=False)
                            loss = criterion(outputs, density_maps) + 0.1 * ssim_loss(outputs, density_maps)
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                        running_loss += loss.item()
                        del inputs, density_maps, outputs, loss
                        clear_memory()

                        if (i + 1) % 10 == 0:
                            logger.info(f'Epoch {epoch+1}/{args.epochs}, Batch {i+1}/{len(train_loader)}, Loss: {running_loss/10:.4f}, VRAM: {get_vram_available() / 1e9:.2f} GB')
                            running_loss = 0.0
                    break
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        logger.warning(f"Błąd VRAM przy batch_size={retry_batch_size}. Zmniejszam do {retry_batch_size // 2}...")
                        retry_batch_size //= 2
                        clear_memory()
                        if retry_batch_size < 1:
                            raise RuntimeError("Nie można znaleźć odpowiedniego batch_size. Spróbuj zmniejszyć rozdzielczość obrazów lub zwiększyć VRAM.")
                    else:
                        raise
                finally:
                    clear_memory()

            val_loss = validate(model, val_loader, criterion, device)
            epoch_time = time.time() - start_time
            logger.info(f"Epoch {epoch+1} took {epoch_time:.2f} seconds.")

            if (epoch + 1) % 10 == 0:
                save_path = f"/app/backend/MCNN/models/{nazwa_modelu}_epoch_{epoch+1}_checkpoint.pth"
                try:
                    model_cpu = model.cpu()
                    torch.save(model_cpu.state_dict(), save_path)
                    model.to(device)
                    logger.info(f"Zapisano checkpoint: {save_path}")
                except RuntimeError as e:
                    logger.error(f"Błąd zapisu checkpointu: {e}")
                finally:
                    clear_memory()

        model_path = f"/app/backend/MCNN/models/{nazwa_modelu}_final_checkpoint.pth"
        model_cpu = model.cpu()
        torch.save(model_cpu.state_dict(), model_path)
        logger.info(f"Model zapisany jako: {model_path}")

    except Exception as e:
        logger.exception("Błąd podczas treningu! Zapisuję awaryjny checkpoint...")
        emergency_path = f"/app/backend/MCNN/models/{nazwa_modelu}_emergency_checkpoint.pth"
        try:
            model_cpu = model.cpu()
            torch.save(model_cpu.state_dict(), emergency_path)
            logger.info(f"Awaryjny model zapisany jako: {emergency_path}")
        except Exception as save_error:
            logger.error(f"Błąd zapisu awaryjnego checkpointu: {save_error}")
        sys.exit(1)

# === GŁÓWNE WYWOŁANIE ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', required=True)
    parser.add_argument('--val_dir', required=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.00075)
    parser.add_argument('--model_name', default=None)
    parser.add_argument('--model_checkpoint', default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--coco_train_path', required=True)
    parser.add_argument('--coco_gt_path', required=True)
    parser.add_argument('--num_augmentations', type=int, default=1, help="Liczba augmentacji (na razie nieużywana)")
    args = parser.parse_args()

    logger.debug("Arguments received: %s", sys.argv)
    logger.debug("Checking if coco_train_path exists: %s", os.path.exists(args.coco_train_path))
    logger.debug("Checking if coco_gt_path exists: %s", os.path.exists(args.coco_gt_path))

    train_model(args)