# data_loader.py
import os
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from pycocotools.coco import COCO
from PIL import Image
import psutil
import logging

# Konfiguracja logowania
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def estimate_batch_size(image_size, max_batch_size=16, min_batch_size=1, use_amp=True, is_training=True):
    """
    Estymuje batch size na podstawie dostępnej pamięci RAM i VRAM, uwzględniając pesymistyczną liczbę obiektów na obraz.
    
    Args:
        image_size (tuple): Rozmiar obrazu (wysokość, szerokość).
        max_batch_size (int): Maksymalny batch size.
        min_batch_size (int): Minimalny batch size.
        use_amp (bool): Czy używane jest mixed precision (torch.cuda.amp).
        is_training (bool): Czy estymacja dotyczy treningu (True) czy inferencji (False).
    
    Returns:
        int: Estymowany batch size.
    """
    if image_size is None:
        raise ValueError("image_size musi być podane jako argument w estimate_batch_size")

    # Pamięć na obrazy (RGB, float32)
    image_memory = image_size[0] * image_size[1] * 3 * 4  # np. 12 MB dla 1024x1024
    
    # Pamięć na aktywacje (dostosowana do Faster R-CNN, ~0.5 GB przy batch_size=1)
    activations_memory_per_image = 0.7 * 1024 ** 3  # 0.5 GB na aktywacje
    
    # Pamięć na model (wagi Faster R-CNN, ~40M parametrów)
    model_memory = 0.6 * 1024 ** 3  # 0.6 GB na wagi
    
    # Overhead CUDA/PyTorch
    cuda_overhead = 1.0 * 1024 ** 3  # 1 GB
    
    # Mnożnik dla mixed precision
    amp_factor = 0.6 if use_amp else 1.0
    
    # Całkowita pamięć na obraz dla GPU
    memory_per_image_gpu = (image_memory + activations_memory_per_image) * amp_factor
    
    # Dodatkowa pamięć na gradienty i optymalizator w treningu
    if is_training:
        model_memory *= 3  # Wagi + gradienty + optymalizator (SGD) = 1.8 GB
        memory_per_image_gpu *= 1.5  # Gradienty dla aktywacji
    
    # Dostępna pamięć systemowa
    available_memory = psutil.virtual_memory().available
    cpu_count = os.cpu_count()
    usable_memory = available_memory * 0.7
    max_images_by_memory = int(usable_memory // image_memory)
    max_images_by_cpu = cpu_count

    # Dostępna pamięć GPU
    max_images_by_gpu = max_batch_size
    if torch.cuda.is_available():
        try:
            gpu_memory_total = torch.cuda.get_device_properties(0).total_memory
            gpu_memory_free = gpu_memory_total - torch.cuda.memory_reserved(0) - torch.cuda.memory_allocated(0)
            usable_gpu_memory = gpu_memory_free * 0.7  # 70% wolnej pamięci
            total_memory_per_image = memory_per_image_gpu + (model_memory + cuda_overhead) / max_batch_size
            max_images_by_gpu = int(usable_gpu_memory // total_memory_per_image)
            logger.info("Dostępna pamięć GPU: %.2f GB, szacowane zużycie na obraz: %.2f MB, model: %.2f GB",
                        gpu_memory_free / (1024 ** 3), total_memory_per_image / (1024 ** 2), model_memory / (1024 ** 3))
        except Exception as e:
            logger.warning("Nie można sprawdzić pamięci GPU: %s. Używam konserwatywnego batch_size.", str(e))
            max_images_by_gpu = 1

    # Ostateczny batch size
    max_possible_batch_size = min(max_images_by_memory, max_images_by_cpu, max_images_by_gpu, max_batch_size)
    batch_size = max(min_batch_size, min(max_possible_batch_size, max_batch_size))

    logger.info("Estymowany batch_size: %d (pamięć RAM: %.2f GB, pamięć GPU: %.2f GB, CPU: %d rdzeni, AMP: %s, training: %s)",
                batch_size, available_memory / (1024 ** 3), gpu_memory_free / (1024 ** 3) if torch.cuda.is_available() else 0,
                cpu_count, use_amp, is_training)
    return batch_size

class CocoDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, annotation_path, transforms=None):
        self.image_dir = image_dir
        self.coco = COCO(annotation_path)
        self.image_ids = list(self.coco.imgs.keys())
        self.transforms = transforms

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)

        img_info = self.coco.loadImgs(image_id)[0]
        path = img_info['file_name']
        image = Image.open(os.path.join(self.image_dir, path)).convert("RGB")

        boxes = []
        labels = []

        for ann in anns:
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])
            labels.append(ann['category_id'])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([image_id])

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": image_id
        }

        if self.transforms:
            image = self.transforms(image)

        return image, target

    def __len__(self):
        return len(self.image_ids)

def get_data_loaders(train_path, val_path, train_annotations, val_annotations, batch_size=None, num_workers=4):
    transforms = T.Compose([T.ToTensor()])
    
    # Ustalanie image_size
    image_size = (1024, 1024)  # Zgodne z Mask R-CNN

    # Automatyczne dostosowanie batch_size, jeśli nie podano
    if batch_size is None:
        batch_size = estimate_batch_size(
            image_size=image_size,
            max_batch_size=8,
            min_batch_size=1,
            use_amp=True,
            is_training=True
        )

    # Automatyczne dostosowanie num_workers
    available_memory = psutil.virtual_memory().available / (1024 ** 3)  # Dostępna pamięć w GB
    cpu_count = os.cpu_count()
    cpu_usage = psutil.cpu_percent(interval=1)
    
    if available_memory < 4:  # Jeśli mniej niż 4GB wolnej pamięci
        num_workers = min(num_workers, max(1, cpu_count // 2))
        logger.warning("Mało pamięci systemowej (%.2f GB), zmniejszam num_workers do %d", available_memory, num_workers)
    elif cpu_usage > 80:  # Jeśli obciążenie CPU jest wysokie
        num_workers = min(num_workers, max(1, cpu_count // 2))
        logger.warning("Wysokie obciążenie CPU (%.1f%%), zmniejszam num_workers do %d", cpu_usage, num_workers)
    else:
        num_workers = min(num_workers, cpu_count)

    # Dynamiczne zarządzanie pin_memory
    use_pin_memory = False
    if torch.cuda.is_available():
        try:
            gpu_memory_total = torch.cuda.get_device_properties(0).total_memory
            gpu_memory_free = gpu_memory_total - torch.cuda.memory_reserved(0) - torch.cuda.memory_allocated(0)
            model_memory = 1.8 * 1024 ** 3  # 1.8 GB (model + gradienty)
            cuda_overhead = 1.0 * 1024 ** 3  # 1 GB
            memory_per_image = (image_size[0] * image_size[1] * 3 * 4 + 0.5 * 1024 ** 3 * 0.6 * 1.5)
            estimated_memory_usage = model_memory + cuda_overhead + memory_per_image * batch_size
            if gpu_memory_free - estimated_memory_usage > 1.0 * 1024 ** 3:  # >1 GB wolnej pamięci
                use_pin_memory = True
                logger.info("Włączam pin_memory, dostępna pamięć GPU: %.2f GB, szacowane zużycie: %.2f GB",
                            gpu_memory_free / (1024 ** 3), estimated_memory_usage / (1024 ** 3))
            else:
                logger.warning("Za mało pamięci GPU (%.2f GB wolnej, szacowane zużycie: %.2f GB), wyłączam pin_memory",
                               gpu_memory_free / (1024 ** 3), estimated_memory_usage / (1024 ** 3))
        except Exception as e:
            logger.warning("Nie można sprawdzić pamięci GPU: %s. Wyłączam pin_memory.", str(e))

    logger.info("Używam batch_size=%d, %d wątków w DataLoader, pin_memory=%s", batch_size, num_workers, use_pin_memory)

    train_dataset = CocoDataset(train_path, train_annotations, transforms=transforms)
    val_dataset = CocoDataset(val_path, val_annotations, transforms=transforms)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=lambda x: tuple(zip(*x)),
        pin_memory=use_pin_memory,
        prefetch_factor=2 if num_workers > 0 else None
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda x: tuple(zip(*x)),
        pin_memory=use_pin_memory,
        prefetch_factor=2 if num_workers > 0 else None
    )

    return train_loader, val_loader