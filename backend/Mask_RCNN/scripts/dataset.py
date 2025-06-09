"""
Moduł obsługi danych dla modelu Mask R-CNN

Ten moduł implementuje zaawansowaną obsługę zbiorów danych dla modelu Mask R-CNN,
umożliwiając ładowanie, przetwarzanie i augmentację danych z adnotacjami w formacie COCO.
Zapewnia mechanizmy adaptacyjnego zarządzania pamięcią, równoległego przetwarzania
oraz rozbudowanej augmentacji danych.
"""

#######################
# Importy bibliotek
#######################
import os                  # Do operacji na systemie plików
import json                # Do operacji na plikach JSON
import cv2                 # OpenCV do operacji na obrazach
import numpy as np         # Do operacji na tablicach numerycznych
import torch               # Framework PyTorch
from torch.utils.data import Dataset, DataLoader  # Klasy bazowe do obsługi danych
import torchvision.transforms.v2 as T            # Transformacje obrazów
import albumentations as A                       # Zaawansowane augmentacje obrazów
from pycocotools import mask as mask_utils       # Narzędzia do obsługi masek COCO
import logging                                   # Do logowania
from functools import partial                    # Do częściowej aplikacji funkcji
import psutil                                    # Do monitorowania zasobów systemowych
from multiprocessing import Manager               # Do współdzielenia danych w multiprocessing
from utils import custom_collate_fn, estimate_batch_size  # Funkcje pomocnicze do obliczeń
from collections import OrderedDict             # Do uporządkowanych słowników

#######################
# Konfiguracja logowania
#######################
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

#######################
# Ustawienia domyślne
#######################
NUM_WORKERS = 4  # Domyślna liczba wątków do równoległego przetwarzania

#######################
# Klasa zbioru danych
#######################
class MyDataset(Dataset):
    """
    Klasa implementująca zbiór danych dla segmentacji instancji z formatem COCO.
    
    Zapewnia zaawansowane funkcje, takie jak:
    - Dynamiczne zarządzanie pamięcią
    - Rozbudowana augmentacja danych
    - Cachowanie masek dla wydajności
    - Walidacja adnotacji i obrazów
    - Obsługa błędów podczas ładowania
    
    Attributes:
        root (str): Katalog główny ze zbiorem danych
        split (str): Podział zbioru ('train' lub 'val')
        image_size (tuple): Docelowy rozmiar obrazów (szerokość, wysokość)
        augment (bool): Czy stosować augmentację danych
        num_augmentations (int): Liczba augmentacji na obraz
        annotations (dict): Załadowane adnotacje w formacie COCO
        image_ids (list): Lista identyfikatorów obrazów
        max_objects (int): Maksymalna liczba obiektów na obrazie
        mask_cache (dict): Współdzielona pamięć podręczna zdekodowanych masek
    """
    def __init__(self, root, split, image_size, augment=False, num_augmentations=1, annotation_path=None, num_processes=4):
        """
        Args:
            root (str): Ścieżka do katalogu z danymi (np. ../../data/train lub ../../data/val).
            split (str): "train" lub "val".
            image_size (tuple): Rozmiar obrazów (wysokość, szerokość) - wymagany parametr.
            augment (bool): Czy stosować augmentacje.
            num_augmentations (int): Liczba augmentacji na obraz.
            annotation_path (str, optional): Ścieżka do pliku COCO z adnotacjami. Jeśli None, używa domyślnej lokalizacji.
            num_processes (int): Liczba procesów do równoległego wczytywania danych.
        """
        if image_size is None:
            raise ValueError("image_size musi być podane jako argument w konstruktorze MyDataset")

        self.root = root
        self.split = split
        self.image_size = image_size
        self.augment = augment
        self.num_augmentations = num_augmentations if augment else 1
        self.image_dir = os.path.join(root, "images")
        self.num_processes = num_processes

        # Współdzielony cache masek dla multiprocessing
        self.manager = Manager()
        self.mask_cache = self.manager.dict()

        # Lokalny cache masek zamiast współdzielonego
        self.mask_cache = OrderedDict()
        self.max_cache_size = 1000  # Maksymalna liczba masek w cache

        # Dostosowanie num_processes na podstawie obciążenia CPU
        cpu_count = os.cpu_count()
        cpu_usage = psutil.cpu_percent(interval=1)
        if cpu_usage > 80:
            self.num_processes = min(num_processes, max(1, cpu_count // 2))
            logger.warning("Wysokie obciążenie CPU (%.1f%%), zmniejszam num_processes do %d", cpu_usage, self.num_processes)
        else:
            self.num_processes = min(num_processes, cpu_count)

        # Ustalanie ścieżki do pliku adnotacji
        if annotation_path is None:
            annotation_file = "instances_train.json" if split == "train" else "instances_val.json"
            self.annotation_path = os.path.join(root, "annotations", annotation_file)
        else:
            self.annotation_path = annotation_path

        if not os.path.exists(self.annotation_path):
            raise FileNotFoundError(f"Nie znaleziono pliku adnotacji: {self.annotation_path}")
        with open(self.annotation_path, 'r') as f:
            self.annotations = json.load(f)

        self.image_info = {img['id']: img for img in self.annotations['images']}

        # Walidacja adnotacji – sprawdzanie poprawności category_id
        category_ids = {cat['id'] for cat in self.annotations['categories']}
        invalid_anns = [ann for ann in self.annotations['annotations'] if ann['category_id'] not in category_ids]
        if invalid_anns:
            logger.warning("Znaleziono adnotacje z niepoprawnymi category_id: %s", invalid_anns)
            self.annotations['annotations'] = [ann for ann in self.annotations['annotations'] if ann['category_id'] in category_ids]

        # Statystyki liczby obiektów i sortowanie według liczby adnotacji
        anns_per_image = {}
        for ann in self.annotations['annotations']:
            img_id = ann['image_id']
            anns_per_image[img_id] = anns_per_image.get(img_id, 0) + 1
        if anns_per_image:
            avg_objects = sum(anns_per_image.values()) / len(anns_per_image)
            self.max_objects = max(anns_per_image.values())  # Przechowujemy max_objects
            logger.info("Średnia liczba obiektów na obraz: %.2f, maksymalna: %d", avg_objects, self.max_objects)

        # Filtruj obrazy: tylko te, które istnieją i mają adnotacje z bboxami
        self.image_ids = []
        for img_id in self.image_info.keys():
            image_path = os.path.join(self.image_dir, self.image_info[img_id]['file_name'])
            if not os.path.exists(image_path):
                logger.warning("Pomijam obraz, plik nie istnieje: %s", image_path)
                continue
            anns = [a for a in self.annotations['annotations'] if a['image_id'] == img_id]
            if not anns:
                logger.warning("Pomijam obraz bez adnotacji: %s", image_path)
                continue
            has_bboxes = any('bbox' in ann and len(ann['bbox']) == 4 for ann in anns)
            if not has_bboxes:
                logger.warning("Pomijam obraz bez bboxów: %s", image_path)
                continue
            self.image_ids.append(img_id)

        self.image_ids.sort(key=lambda img_id: anns_per_image.get(img_id, 0))
        logger.info("Posortowano obrazy według liczby adnotacji: pierwsza próbka ma %d adnotacji, ostatnia %d",
                    anns_per_image.get(self.image_ids[0], 0), anns_per_image.get(self.image_ids[-1], 0))

        logger.info("Załadowano %d obrazów z adnotacjami w %s", len(self.image_ids), self.image_dir)

        if not self.image_ids:
            raise ValueError(f"Brak obrazów z adnotacjami i bboxami w %s", self.image_dir)

        # Transformacje z torchvision.transforms.v2
        self.base_transform = T.Compose([
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
        ])

        # Pipeline augmentacji
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
                min_area=10,
                min_visibility=0.1
            ),
            additional_targets={'masks': 'masks'}
        )

    def __len__(self):
        """Zwraca liczbę próbek w zbiorze danych, uwzględniając augmentacje."""
        return len(self.image_ids) * self.num_augmentations

    def decode_rle(self, segmentation, bbox, target_size):
        """
        Dekoduje maskę z formatu RLE (Run-Length Encoding) do maski binarnej.
        """
        rle_key = str(segmentation["counts"]) + str(segmentation["size"])
        if rle_key in self.mask_cache:
            return self.mask_cache[rle_key]

        rle = {"counts": segmentation["counts"].encode('utf-8'), "size": segmentation["size"]}
        mask = mask_utils.decode(rle)
        
        x, y, w, h = map(int, bbox)
        if w <= 0 or h <= 0:
            mask = np.zeros((target_size[0], target_size[1]), dtype=np.uint8)
        else:
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            full_mask = np.zeros((target_size[0], target_size[1]), dtype=np.uint8)
            x_end = min(x + w, target_size[1])
            y_end = min(y + h, target_size[0])
            full_mask[y:y_end, x:x_end] = mask[:y_end - y, :x_end - x]
            mask = full_mask

        self.mask_cache[rle_key] = mask
        if len(self.mask_cache) > self.max_cache_size:
            self.mask_cache.popitem(last=False)  # Usuń najstarszy wpis
        return mask
    
    def _load_item(self, idx):
        """
        Wewnętrzna funkcja ładująca i przetwarzająca pojedynczą próbkę.
        
        Args:
            idx (int): Indeks próbki
            
        Returns:
            tuple: (obraz, cel) lub None w przypadku błędu
        """
        # Wyznaczenie oryginalnego indeksu obrazu i indeksu augmentacji
        orig_idx = idx // self.num_augmentations
        aug_idx = idx % self.num_augmentations

        # Pobranie informacji o obrazie
        image_id = self.image_ids[orig_idx]
        image_info = self.image_info[image_id]
        image_path = os.path.join(self.image_dir, image_info['file_name'])

        try:
            # Wczytanie obrazu
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Nie można wczytać obrazu: {image_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            orig_height, orig_width = image.shape[:2]

            # Pobranie adnotacji dla obrazu
            anns = [a for a in self.annotations['annotations'] if a['image_id'] == image_id]
            if not anns:
                raise ValueError(f"Brak adnotacji dla obrazu: {image_path}")

            # Wyodrębnienie boxów, masek i etykiet
            boxes = [ann['bbox'] for ann in anns]
            masks = [self.decode_rle(ann['segmentation'], ann['bbox'], (orig_height, orig_width))
                    if 'segmentation' in ann else None for ann in anns]
            labels = [ann['category_id'] for ann in anns]

            # Zastosowanie augmentacji dla odpowiednich indeksów
            if self.augment and aug_idx > 0:
                valid_boxes = boxes
                valid_labels = labels
                valid_masks = masks

                # Zapewnienie, że wszystkie maski są niepuste
                if not any(mask is not None for mask in valid_masks):
                    valid_masks = [np.zeros((orig_height, orig_width), dtype=np.uint8) for _ in range(len(valid_boxes))]

                # Przygotowanie danych dla augmentacji
                aug_data = {
                    'image': image,
                    'bboxes': valid_boxes,
                    'category_ids': valid_labels,
                    'masks': valid_masks
                }
                
                # Zastosowanie transformacji
                augmented = self.augment_transform(**aug_data)
                image = augmented['image']
                aug_boxes = augmented['bboxes']
                aug_labels = augmented['category_ids']
                aug_masks = augmented['masks']

                # Walidacja augmentowanego obrazu
                if np.any(image < 0) or np.any(image > 255) or np.any(np.isnan(image)):
                    raise ValueError(f"Niepoprawne wartości w obrazie po augmentacji: {image_path}")

                # Filtrowanie i normalizacja boxów po augmentacji
                height, width = image.shape[:2]
                filtered_boxes = []
                filtered_labels = []
                filtered_masks = []

                for bbox, label, mask in zip(aug_boxes, aug_labels, aug_masks):
                    x, y, w, h = map(float, bbox)  # Upewniamy się, że wartości są float
                    # Normalizacja współrzędnych
                    x = max(0, min(x, width - 1))
                    y = max(0, min(y, height - 1))
                    x_end = max(0, min(x + w, width - 1))
                    y_end = max(0, min(y + h, height - 1))
                    w = x_end - x
                    h = y_end - y

                    if w > 0 and h > 0:
                        filtered_boxes.append([x, y, w, h])
                        filtered_labels.append(label)
                        filtered_masks.append(mask)

                boxes = filtered_boxes
                labels = filtered_labels
                masks = filtered_masks

                if len(boxes) == 0:
                    raise ValueError(f"Obraz {image_path} (aug_idx={aug_idx}) nie ma bboxów po augmentacji")
            else:
                # Standardowa zmiana rozmiaru bez augmentacji
                image = cv2.resize(image, self.image_size)
                scale_x = self.image_size[0] / orig_width
                scale_y = self.image_size[1] / orig_height
                boxes = [[x * scale_x, y * scale_y, w * scale_x, h * scale_y]
                        for x, y, w, h in boxes]
                masks = [cv2.resize(mask, self.image_size, interpolation=cv2.INTER_NEAREST) if mask is not None else None
                        for mask in masks]

                # Normalizacja boxów po skalowaniu
                filtered_boxes = []
                for box in boxes:
                    x, y, w, h = map(float, box)
                    x = max(0, min(x, self.image_size[0] - 1))
                    y = max(0, min(y, self.image_size[1] - 1))
                    x_end = max(0, min(x + w, self.image_size[0] - 1))
                    y_end = max(0, min(y + h, self.image_size[1] - 1))
                    w = x_end - x
                    h = y_end - y
                    if w > 0 and h > 0:
                        filtered_boxes.append([x, y, w, h])
                boxes = filtered_boxes

            if len(boxes) == 0:
                raise ValueError(f"Obraz {image_path} nie ma bboxów po skalowaniu")

            # Konwersja do tensorów PyTorch
            image = self.base_transform(image)
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

            # Przygotowanie masek
            if masks and all(m is not None for m in masks):
                masks = torch.stack([torch.as_tensor(m, dtype=torch.uint8) for m in masks], dim=0)
            else:
                masks = torch.zeros((0, self.image_size[1], self.image_size[0]), dtype=torch.uint8)

            # Konwersja boxów z formatu COCO [x, y, width, height] na format PyTorch [x_min, y_min, x_max, y_max]
            if len(boxes) > 0:
                boxes = torch.stack([
                    boxes[:, 0],
                    boxes[:, 1],
                    boxes[:, 0] + boxes[:, 2],
                    boxes[:, 1] + boxes[:, 3]
                ], dim=1)
                
                # Ostateczna normalizacja boxów w formacie [x_min, y_min, x_max, y_max]
                boxes[:, 0] = torch.clamp(boxes[:, 0], 0, self.image_size[0] - 1)  # x_min
                boxes[:, 1] = torch.clamp(boxes[:, 1], 0, self.image_size[1] - 1)  # y_min
                boxes[:, 2] = torch.clamp(boxes[:, 2], 0, self.image_size[0] - 1)  # x_max
                boxes[:, 3] = torch.clamp(boxes[:, 3], 0, self.image_size[1] - 1)  # y_max

                # Filtracja niepoprawnych boxów
                invalid_boxes = (boxes[:, 2] <= boxes[:, 0]) | (boxes[:, 3] <= boxes[:, 1])
                if invalid_boxes.any():
                    logger.warning("Niepoprawne bboxy w %s: %s", image_info['file_name'], boxes[invalid_boxes])
                    boxes = boxes[~invalid_boxes]
                    labels = labels[~invalid_boxes]
                    masks = masks[~invalid_boxes] if masks.shape[0] > 0 else masks

            if len(boxes) == 0:
                raise ValueError(f"Obraz {image_path} nie ma bboxów po finalnym filtrowaniu")

            # Przygotowanie słownika celu dla modelu
            target = {
                'boxes': boxes,
                'labels': labels,
                'masks': masks,
                'image_id': torch.tensor([image_id]),
                'area': (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) if len(boxes) > 0 else torch.tensor([]),
                'iscrowd': torch.zeros((len(labels),), dtype=torch.int64)
            }

            # Ostateczna walidacja tensorów (sprawdzenie NaN/Inf)
            if torch.isnan(image).any() or torch.isinf(image).any():
                raise ValueError(f"NaN/Inf w obrazie: {image_info['file_name']}")
            if torch.isnan(boxes).any() or torch.isinf(boxes).any():
                raise ValueError(f"NaN/Inf w bboxach: {image_info['file_name']}")
            if torch.isnan(masks).any() or torch.isinf(masks).any():
                raise ValueError(f"NaN/Inf w maskach: {image_info['file_name']}")

            return image, target

        except Exception as e:
            logger.error("Błąd podczas ładowania próbki idx=%d (image_id=%d): %s", idx, image_id, str(e))
            return None

    def __getitem__(self, idx):
        """
        Zwraca pojedynczą próbkę ze zbioru danych.
        
        Implementuje mechanizm odporności na błędy - jeśli próbka jest niepoprawna,
        próbuje załadować inną próbkę.
        
        Args:
            idx (int): Indeks próbki
            
        Returns:
            tuple: (obraz, cel)
            
        Raises:
            ValueError: Jeśli nie udało się znaleźć poprawnej próbki po wielu próbach
        """
        max_attempts = len(self.image_ids)
        attempts = 0
        while attempts < max_attempts:
            result = self._load_item(idx)
            if result is not None:
                return result
            logger.warning("Próba %d/%d: Pomijam próbkę idx=%d po błędzie", attempts + 1, max_attempts, idx)
            idx = ((idx // self.num_augmentations + 1) % len(self.image_ids)) * self.num_augmentations + (idx % self.num_augmentations)
            attempts += 1
        raise ValueError(f"Nie udało się znaleźć poprawnego obrazu po %d próbach w zbiorze danych: %s", max_attempts, self.image_dir)
    
    def _shut_down_workers(self):
        """Zamyka zasoby multiprocessing, jeśli istnieją."""
        self.mask_cache.clear()
        logger.info("Wyczyszczono cache masek")
        # Dodatkowe czyszczenie zasobów multiprocessing
        try:
            self.manager._process.terminate()
            self.manager._process.join()
            logger.info("Zakończono procesy managera multiprocessing")
        except AttributeError:
            logger.debug("Brak aktywnych procesów managera do zakończenia")


def get_data_loaders(train_dir, val_dir, batch_size=None, num_workers=NUM_WORKERS, num_augmentations=1, coco_train_path=None, coco_val_path=None, num_processes=4):
    """
    Tworzy i konfiguruje DataLoadery dla zbiorów treningowego i walidacyjnego.
    
    Funkcja automatycznie estymuje optymalny batch size jeśli nie jest podany,
    dostosowuje liczbę wątków do obciążenia systemu i określa, czy używać pin_memory.
    
    Args:
        train_dir (str): Ścieżka do katalogu z danymi treningowymi
        val_dir (str): Ścieżka do katalogu z danymi walidacyjnymi
        batch_size (int, optional): Rozmiar batcha (jeśli None, estymowany automatycznie)
        num_workers (int): Liczba wątków dla DataLoadera
        num_augmentations (int): Liczba augmentacji na obraz
        coco_train_path (str): Ścieżka do pliku COCO z adnotacjami treningowymi
        coco_val_path (str): Ścieżka do pliku COCO z adnotacjami walidacyjnymi
        num_processes (int): Liczba procesów do równoległego wczytywania danych
        
    Returns:
        tuple: (train_loader, val_loader) - DataLoadery dla zbiorów treningowego i walidacyjnego
    """
    # Ustalenie docelowego rozmiaru obrazów
    image_size = (1024, 1024)

    # Ustalenie ścieżek do plików adnotacji COCO
    if coco_train_path is None:
        coco_train_path = os.path.join(train_dir, "annotations", "instances_train.json")
    if coco_val_path is None:
        coco_val_path = os.path.join(val_dir, "annotations", "instances_val.json")

    # Inicjalizacja datasetów
    train_dataset = MyDataset(
        root=train_dir,
        split="train",
        image_size=image_size,
        augment=True,
        num_augmentations=num_augmentations,
        annotation_path=coco_train_path,
        num_processes=num_processes
    )

    val_dataset = MyDataset(
        root=val_dir,
        split="val",
        image_size=image_size,
        augment=False,
        annotation_path=coco_val_path,
        num_processes=num_processes
    )

    # Użycie większej wartości max_objects z obu datasetów
    max_objects = max(getattr(train_dataset, 'max_objects', 400), getattr(val_dataset, 'max_objects', 400))

    # Automatyczna estymacja batch_size
    if batch_size is None:
        batch_size = estimate_batch_size(
            image_size=image_size,
            max_objects=max_objects,
            max_batch_size=8,
            min_batch_size=1,
            use_amp=True,
            is_training=True
        )

    # Monitorowanie zasobów systemowych i dostosowanie num_workers
    available_memory = psutil.virtual_memory().available / (1024 ** 3)
    cpu_count = os.cpu_count()
    cpu_usage = psutil.cpu_percent(interval=1)

    # Dostosowanie num_workers w zależności od obciążenia systemu
    if available_memory < 4:
        num_workers = min(num_workers, max(1, cpu_count // 3))
        logger.warning("Mało pamięci systemowej (%.2f GB), zmniejszam num_workers do %d", available_memory, num_workers)
    elif cpu_usage > 70:
        num_workers = min(num_workers, max(1, cpu_count // 3))
        logger.warning("Wysokie obciążenie CPU (%.1f%%), zmniejszam num_workers do %d", cpu_usage, num_workers)
    else:
        num_workers = min(num_workers, cpu_count // 1.5)

    # Decyzja o użyciu pin_memory na podstawie dostępności zasobów
    use_pin_memory = False
    if torch.cuda.is_available():
        try:
            # Analiza pamięci GPU
            gpu_memory_total = torch.cuda.get_device_properties(0).total_memory
            gpu_memory_free = gpu_memory_total - torch.cuda.memory_reserved(0) - torch.cuda.memory_allocated(0)
            
            # Szacowanie zużycia pamięci
            model_memory = 1.8 * 1024 ** 3
            cuda_overhead = 1.0 * 1024 ** 3
            memory_per_image = (image_size[0] * image_size[1] * 3 * 4 +
                                max_objects * image_size[0] * image_size[1] * 1 +
                                0.5 * 1024 ** 3 * 0.6 * 1.5)
            estimated_memory_usage = model_memory + cuda_overhead + memory_per_image * batch_size
            
            # Włączenie pin_memory tylko jeśli jest wystarczająco pamięci
            if gpu_memory_free - estimated_memory_usage > 2.0 * 1024 ** 3 and available_memory > 8:
                use_pin_memory = False
                logger.info("Włączam pin_memory, dostępna pamięć GPU: %.2f GB, szacowane zużycie: %.2f GB, RAM: %.2f GB",
                            gpu_memory_free / (1024 ** 3), estimated_memory_usage / (1024 ** 3), available_memory)
            else:
                logger.warning("Za mało pamięci GPU (%.2f GB wolnej, szacowane zużycie: %.2f GB) lub RAM (%.2f GB), wyłączam pin_memory",
                               gpu_memory_free / (1024 ** 3), estimated_memory_usage / (1024 ** 3), available_memory)
        except Exception as e:
            logger.warning("Nie można sprawdzić pamięci GPU: %s. Wyłączam pin_memory.", str(e))

    # Logowanie wybranych parametrów
    logger.info("Używam batch_size=%d, %d wątków w DataLoader, pin_memory=%s, max obiektów: %d",
                batch_size, num_workers, use_pin_memory, max_objects)

    # Zwolnienie pamięci GPU przed utworzeniem DataLoaderów
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Tworzenie DataLoaderów
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=custom_collate_fn,
        pin_memory=False,       # Pin memory nie działa z albumentations, więc wyłączamy, TODO poprawić, False / use_pin_memory
        prefetch_factor=2 if num_workers > 0 else None
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=max(1, batch_size // 2),
        shuffle=False,
        num_workers=num_workers,
        collate_fn=custom_collate_fn,
        pin_memory=False, # Pin memory nie działa z albumentations, więc wyłączamy, TODO poprawić, False / use_pin_memory
        prefetch_factor=1 if num_workers > 0 else None
    )

    return train_loader, val_loader