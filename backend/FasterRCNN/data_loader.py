"""
Moduł ładowania danych dla modelu Faster R-CNN

Ten moduł dostarcza funkcje i klasy do efektywnego ładowania i przetwarzania 
danych treningowych i walidacyjnych. Implementuje zaawansowane mechanizmy 
augmentacji danych, automatycznego dostosowywania parametrów ładowania do 
dostępnych zasobów systemowych oraz konwersji między różnymi formatami danych.
"""

#######################
# Importy bibliotek
#######################
import os                # Do operacji na systemie plików
import torch             # Framework PyTorch do treningu modeli głębokich sieci neuronowych
from torch.utils.data import DataLoader  # Klasa do wydajnego ładowania danych
import torchvision.transforms as T       # Transformacje obrazów
import albumentations as A               # Zaawansowana biblioteka augmentacji obrazów
from pycocotools.coco import COCO        # Narzędzia do obsługi formatu COCO
from PIL import Image                    # Biblioteka do operacji na obrazach
import numpy as np                       # Biblioteka do operacji numerycznych
import psutil                           # Do monitorowania zasobów systemowych
import logging                          # Do logowania informacji i błędów

#######################
# Konfiguracja logowania
#######################
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

#######################
# Estymacja rozmiaru wsadu (batch size)
#######################
def estimate_batch_size(image_size, max_batch_size=16, min_batch_size=1, use_amp=True, is_training=True):
    """
    Estymuje batch size na podstawie dostępnej pamięci RAM i VRAM, uwzględniając pesymistyczną liczbę obiektów na obraz.
    
    Funkcja analizuje dostępne zasoby systemowe (pamięć RAM, VRAM) i na tej podstawie
    oblicza optymalny rozmiar wsadu, aby uniknąć błędów out-of-memory (OOM).
    
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
    activations_memory_per_image = 0.7 * 1024 ** 3  # 0.7 GB na aktywacje
    
    # Pamięć na model (wagi Faster R-CNN, ~40M parametrów)
    model_memory = 0.7 * 1024 ** 3  # 0.7 GB na wagi
    
    # Overhead CUDA/PyTorch
    cuda_overhead = 1.2 * 1024 ** 3  # 1.2 GB
    
    # Mnożnik dla mixed precision
    amp_factor = 0.7 if use_amp else 1.0
    
    # Całkowita pamięć na obraz dla GPU
    memory_per_image_gpu = (image_memory + activations_memory_per_image) * amp_factor
    
    # Dodatkowa pamięć na gradienty i optymalizator w treningu
    if is_training:
        model_memory *= 3  # Wagi + gradienty + optymalizator (SGD) = 1.8 GB
        memory_per_image_gpu *= 1.5  # Gradienty dla aktywacji
    
    #######################
    # Analiza dostępnych zasobów systemowych
    #######################
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

#######################
# Klasa zbioru danych
#######################
class CocoDataset(torch.utils.data.Dataset):
    """
    Klasa zbioru danych obsługująca format adnotacji COCO.
    
    Implementuje interfejs Dataset PyTorch, umożliwiając ładowanie obrazów i adnotacji
    w formacie COCO oraz stosowanie zaawansowanych technik augmentacji danych.
    
    Attributes:
        image_dir (str): Ścieżka do katalogu z obrazami.
        coco (COCO): Obiekt pycocotools.COCO do obsługi adnotacji.
        image_ids (list): Lista identyfikatorów obrazów.
        image_size (tuple): Docelowy rozmiar obrazów (wysokość, szerokość).
        augment (bool): Czy stosować augmentację danych.
        num_augmentations (int): Liczba wersji augmentowanych dla każdego obrazu.
    """
    def __init__(self, image_dir, annotation_path, image_size=(1024, 1024), augment=False, num_augmentations=1):
        """
        Inicjalizuje zbiór danych COCO z opcjonalną augmentacją.
        
        Args:
            image_dir (str): Ścieżka do katalogu z obrazami.
            annotation_path (str): Ścieżka do pliku adnotacji w formacie COCO.
            image_size (tuple): Docelowy rozmiar obrazów (wysokość, szerokość).
            augment (bool): Czy stosować augmentację danych.
            num_augmentations (int): Liczba wersji augmentowanych dla każdego obrazu.
        """
        self.image_dir = image_dir
        self.coco = COCO(annotation_path)
        self.image_ids = list(self.coco.imgs.keys())
        self.image_size = image_size
        self.augment = augment
        self.num_augmentations = num_augmentations if augment else 1

        # Transformacja minimalna (dla walidacji lub bez augmentacji)
        self.base_transform = T.Compose([T.ToTensor()])

        # Pipeline augmentacji (zgodny z albumentations)
        all_transforms = [
            # Geometryczne transformacje z SomeOf
            A.SomeOf([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Rotate(limit=15, p=0.2),
                A.Perspective(scale=(0.05, 0.15), keep_size=True, p=0.4),
                A.Affine(
                    scale=(0.85, 1.15),
                    translate_percent=(-0.1, 0.1),
                    rotate=(-30, 30),
                    shear=(-5, 5),
                    p=0.4
                ),
                A.RandomResizedCrop(
                    size=self.image_size,
                    scale=(0.9, 1.0),
                    ratio=(0.75, 1.33),
                    p=0.3
                ),
                A.Affine(shear=(-15, 15), p=0.3),
                A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.3),
                A.PiecewiseAffine(scale=(0.01, 0.05), nb_rows=4, nb_cols=4, p=0.2),
                A.RandomScale(scale_limit=(-0.2, 0.2), p=0.3),
                A.CoarseDropout(
                    num_holes_range=(1, 5),
                    fill=0,
                    fill_mask=0,
                    p=0.3
                ),
            ], n=4, p=0.8),

            # Wizualne transformacje z SomeOf
            A.SomeOf([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
                A.GaussNoise(p=0.3),
                A.MultiplicativeNoise(multiplier=(0.5, 1.5), per_channel=True, p=0.2),
                A.Blur(blur_limit=(3, 7), p=0.2),
                A.MotionBlur(blur_limit=(3, 15), p=0.2),
                A.RandomFog(alpha_coef=0.2, p=0.1),
                A.RandomRain(brightness_coefficient=0.9, drop_length=20, p=0.1),
                A.RandomSnow(p=0.1),
                A.RandomSunFlare(
                    flare_roi=(0, 0, 1, 0.5),
                    src_radius=150,
                    p=0.2
                ),
                A.RandomShadow(
                    shadow_roi=(0, 0.5, 1, 1),
                    num_shadows_limit=(1, 3),
                    shadow_dimension=5,
                    p=0.3
                ),
                A.RandomGamma(gamma_limit=(80, 120), p=0.3),
                A.Emboss(p=0.2),
                A.Sharpen(p=0.2),
                A.CLAHE(p=0.2),
                A.ImageCompression(quality_range=(50, 90), p=0.2),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
                A.RandomToneCurve(scale=0.1, p=0.3),
                A.Solarize(p=0.2),
                A.Posterize(num_bits=(4, 8), p=0.2),
                A.Downscale(scale_range=(0.25, 0.5), p=0.2),
                A.Superpixels(p_replace=0.1, n_segments=100, p=0.2),
            ], n=4, p=0.8),

            # Końcowe skalowanie
            A.Resize(height=image_size[1], width=image_size[0]),
        ]

        self.augment_transform = A.Compose(
            all_transforms,
            bbox_params=A.BboxParams(
                format='coco',
                label_fields=['category_ids'],
                min_area=8,  # Minimalna powierzchnia bbox po augmentacji
                min_visibility=0.1  # Minimalna widoczność bbox po augmentacji
            )
        )

    def __len__(self):
        """
        Zwraca liczbę próbek w zbiorze danych, uwzględniając augmentacje.
        
        Returns:
            int: Liczba próbek w zbiorze danych.
        """
        return len(self.image_ids) * self.num_augmentations

    def __getitem__(self, idx):
        """
        Zwraca parę (obraz, adnotacje) dla danego indeksu.
        
        W przypadku gdy augmentacja jest włączona, każdy oryginalny obraz
        jest augmentowany num_augmentations razy, tworząc różne wersje tego samego obrazu.
        
        Args:
            idx (int): Indeks próbki do pobrania.
            
        Returns:
            tuple: Para (obraz jako tensor, adnotacje jako słownik).
        """
        #######################
        # Przygotowanie danych
        #######################
        orig_idx = idx // self.num_augmentations
        aug_idx = idx % self.num_augmentations
        image_id = self.image_ids[orig_idx]
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)

        img_info = self.coco.loadImgs(image_id)[0]
        path = img_info['file_name']
        image = Image.open(os.path.join(self.image_dir, path)).convert("RGB")
        image_np = np.array(image)

        boxes = []
        labels = []

        for ann in anns:
            x, y, w, h = ann['bbox']
            boxes.append([x, y, w, h])
            labels.append(ann['category_id'])

        #######################
        # Augmentacja danych (jeśli włączona)
        #######################
        if self.augment and aug_idx > 0:
            aug_data = {
                'image': image_np,
                'bboxes': boxes,
                'category_ids': labels
            }
            augmented = self.augment_transform(**aug_data)
            image_np = augmented['image']
            boxes = augmented['bboxes']
            labels = augmented['category_ids']

            # Walidacja po augmentacji
            if not boxes:
                logger.warning(f"Obraz {path} (aug_idx={aug_idx}) nie ma bboxów po augmentacji")
                boxes = [[0, 0, 1, 1]]  # Dummy bbox
                labels = [0]  # Dummy label
            else:
                # Filtrowanie niepoprawnych bboxów
                filtered_boxes = []
                filtered_labels = []
                height, width = image_np.shape[:2]
                for bbox, label in zip(boxes, labels):
                    x, y, w, h = map(float, bbox)  # Upewniamy się, że wartości są float
                    x = max(0, min(x, width - 1))
                    y = max(0, min(y, height - 1))
                    x_end = max(0, min(x + w, width - 1))
                    y_end = max(0, min(y + h, height - 1))
                    w = x_end - x
                    h = y_end - y
                    if w > 0 and h > 0:
                        filtered_boxes.append([x, y, w, h])
                        filtered_labels.append(label)
                boxes = filtered_boxes
                labels = filtered_labels

            image = Image.fromarray(image_np)
        else:
            image = image

        #######################
        # Konwersja do formatu PyTorch
        #######################
        image = self.base_transform(image)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([image_id])

        # Konwersja bboxów do formatu [x_min, y_min, x_max, y_max]
        if len(boxes) > 0:
            boxes = torch.stack([
                boxes[:, 0],
                boxes[:, 1],
                boxes[:, 0] + boxes[:, 2],
                boxes[:, 1] + boxes[:, 3]
            ], dim=1)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": image_id
        }

        return image, target

#######################
# Funkcja tworzenia loaderów danych
#######################
def get_data_loaders(train_path, val_path, train_annotations, val_annotations, batch_size=None, num_workers=2, num_augmentations=0):
    """
    Tworzy i konfiguruje DataLoader'y do treningu i walidacji.
    
    Funkcja automatycznie dostosowuje parametry ładowania danych (batch_size, num_workers, pin_memory)
    w zależności od dostępnych zasobów systemowych.
    
    Args:
        train_path (str): Ścieżka do katalogu z obrazami treningowymi.
        val_path (str): Ścieżka do katalogu z obrazami walidacyjnymi.
        train_annotations (str): Ścieżka do pliku adnotacji treningowych.
        val_annotations (str): Ścieżka do pliku adnotacji walidacyjnych.
        batch_size (int, optional): Rozmiar wsadu. Jeśli None, będzie estymowany automatycznie.
        num_workers (int, optional): Liczba wątków roboczych. Domyślnie 4.
        num_augmentations (int, optional): Liczba augmentacji na obraz. Domyślnie 0.
        
    Returns:
        tuple: Para (train_loader, val_loader) z skonfigurowanymi DataLoader'ami.
    """
    # Ustalanie image_size
    image_size = (1024, 1024)

    #######################
    # Automatyczne dostosowanie parametrów
    #######################
    # Automatyczne dostosowanie batch_size, jeśli nie podano
    if batch_size is None:
        batch_size = estimate_batch_size(
            image_size=image_size,
            max_batch_size=16,
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

    #######################
    # Tworzenie zbiorów danych i loaderów
    #######################
    train_dataset = CocoDataset(
        image_dir=train_path,
        annotation_path=train_annotations,
        image_size=image_size,
        augment=True,
        num_augmentations=num_augmentations
    )
    val_dataset = CocoDataset(
        image_dir=val_path,
        annotation_path=val_annotations,
        image_size=image_size,
        augment=False,
        num_augmentations=1
    )

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