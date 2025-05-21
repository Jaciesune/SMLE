import os
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as T
import albumentations as A
from pycocotools import mask as mask_utils
import logging
from multiprocessing import Pool
from functools import partial
import psutil

# Konfiguracja logowania
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class RuryDataset(Dataset):
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
            raise ValueError("image_size musi być podane jako argument w konstruktorze RuryDataset")

        self.root = root
        self.split = split
        self.image_size = image_size
        self.augment = augment
        self.num_augmentations = num_augmentations if augment else 1
        self.image_dir = os.path.join(root, "images")

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

        # Pipeline augmentacji (albumentations)
        self.augment_transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=50, p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.HueSaturationValue(p=0.5),
            A.GaussNoise(p=0.2),
            A.MultiplicativeNoise(multiplier=[0.5, 1.5], per_channel=True, p=0.2),
            A.Blur(blur_limit=(3, 7), p=0.2),
            A.MedianBlur(blur_limit=5, p=0.1),
            A.ISONoise(p=0.1),
            A.Perspective(scale=(0.05, 0.1), keep_size=True, p=0.3),
            A.RandomResizedCrop(
                size=(image_size[1], image_size[0]),
                scale=(0.9, 1.0),
                ratio=(0.75, 1.33),
                p=0.3
            ),
            A.Affine(
                scale=(0.9, 1.1),
                translate_percent=(-0.1, 0.1),
                rotate=(-30, 30),
                shear=(-5, 5),
                p=0.3
            ),
            A.MotionBlur(blur_limit=(3, 15), p=0.2),
            A.Defocus(radius=(3, 10), alias_blur=(0.1, 0.5), p=0.2),
            A.RandomShadow(
                shadow_roi=(0, 0.5, 1, 1),
                num_shadows_limit=(1, 3),
                shadow_dimension=5,
                p=0.3
            ),
            A.RandomSunFlare(
                flare_roi=(0, 0, 1, 0.5),
                num_flare_circles_range=(1, 3),
                src_radius=150,
                p=0.2
            ),
            A.RandomFog(
                fog_coef_range=(0.1, 0.3),
                p=0.1
            ),
            A.RandomRain(brightness_coefficient=0.9, drop_length=20, p=0.1),
            A.RandomSnow(
                snow_point_range=(0.1, 0.3),
                p=0.1
            ),
            A.CoarseDropout(
                num_holes_range=(5, 10),
                hole_height_range=(32, 64),
                hole_width_range=(32, 64),
                p=0.3
            ),
            A.Resize(height=image_size[1], width=image_size[0]),
        ], bbox_params=A.BboxParams(
            format='coco',
            label_fields=['category_ids'],
            min_area=3,
            min_visibility=0.3
        ), additional_targets={'masks': 'masks'})

        self.mask_cache = {}

    def __len__(self):
        return len(self.image_ids) * self.num_augmentations

    def decode_rle(self, segmentation, bbox, target_size):
        """Dekoduje RLE do maski binarnej i pozycjonuje ją w granicach bboxa."""
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
        return self.mask_cache[rle_key]

    def _load_item(self, idx):
        orig_idx = idx // self.num_augmentations
        aug_idx = idx % self.num_augmentations

        image_id = self.image_ids[orig_idx]
        image_info = self.image_info[image_id]
        image_path = os.path.join(self.image_dir, image_info['file_name'])

        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Nie można wczytać obrazu: {image_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            orig_height, orig_width = image.shape[:2]

            anns = [a for a in self.annotations['annotations'] if a['image_id'] == image_id]
            if not anns:
                raise ValueError(f"Brak adnotacji dla obrazu: {image_path}")

            boxes = [ann['bbox'] for ann in anns]
            masks = [self.decode_rle(ann['segmentation'], ann['bbox'], (orig_height, orig_width))
                     if 'segmentation' in ann else None for ann in anns]
            labels = [ann['category_id'] for ann in anns]

            if self.augment and aug_idx > 0:
                valid_boxes = boxes
                valid_labels = labels
                valid_masks = masks

                if not any(mask is not None for mask in valid_masks):
                    valid_masks = [np.zeros((orig_height, orig_width), dtype=np.uint8) for _ in range(len(valid_boxes))]

                aug_data = {
                    'image': image,
                    'bboxes': valid_boxes,
                    'category_ids': valid_labels,
                    'masks': valid_masks
                }
                augmented = self.augment_transform(**aug_data)
                image = augmented['image']
                aug_boxes = augmented['bboxes']
                aug_labels = augmented['category_ids']
                aug_masks = augmented['masks']

                if np.any(image < 0) or np.any(image > 255) or np.any(np.isnan(image)):
                    raise ValueError(f"Niepoprawne wartości w obrazie po augmentacji: {image_path}")

                height, width = image.shape[:2]
                filtered_boxes = []
                filtered_labels = []
                filtered_masks = []

                for bbox, label, mask in zip(aug_boxes, aug_labels, aug_masks):
                    x, y, w, h = map(int, bbox)
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
                image = cv2.resize(image, self.image_size)
                scale_x = self.image_size[0] / orig_width
                scale_y = self.image_size[1] / orig_height
                boxes = [[x * scale_x, y * scale_y, w * scale_x, h * scale_y]
                         for x, y, w, h in boxes]
                masks = [cv2.resize(mask, self.image_size, interpolation=cv2.INTER_NEAREST) if mask is not None else None
                         for mask in masks]

            if len(boxes) == 0:
                raise ValueError(f"Obraz {image_path} nie ma bboxów po skalowaniu")

            image = self.base_transform(image)
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

            if masks and all(m is not None for m in masks):
                masks = torch.stack([torch.as_tensor(m, dtype=torch.uint8) for m in masks], dim=0)
            else:
                masks = torch.zeros((0, self.image_size[1], self.image_size[0]), dtype=torch.uint8)

            if len(boxes) > 0:
                boxes = torch.stack([
                    boxes[:, 0],
                    boxes[:, 1],
                    boxes[:, 0] + boxes[:, 2],
                    boxes[:, 1] + boxes[:, 3]
                ], dim=1)
                invalid_boxes = (boxes[:, 2] <= boxes[:, 0]) | (boxes[:, 3] <= boxes[:, 1])
                if invalid_boxes.any():
                    logger.warning("Niepoprawne bboxy w %s: %s", image_info['file_name'], boxes[invalid_boxes])
                    boxes = boxes[~invalid_boxes]
                    labels = labels[~invalid_boxes]
                    masks = masks[~invalid_boxes] if masks.shape[0] > 0 else masks

            if len(boxes) == 0:
                raise ValueError(f"Obraz {image_path} nie ma bboxów po finalnym filtrowaniu")

            target = {
                'boxes': boxes,
                'labels': labels,
                'masks': masks,
                'image_id': torch.tensor([image_id]),
                'area': (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) if len(boxes) > 0 else torch.tensor([]),
                'iscrowd': torch.zeros((len(labels),), dtype=torch.int64)
            }

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

def custom_collate_fn(batch):
    return tuple(zip(*batch))

def estimate_batch_size(image_size, max_objects, max_batch_size=16, min_batch_size=1, use_amp=True, is_training=True):
    """
    Estymuje batch size na podstawie dostępnej pamięci RAM i VRAM, uwzględniając maksymalną liczbę obiektów na obraz.

    Args:
        image_size (tuple): Rozmiar obrazu (wysokość, szerokość) - wymagany parametr.
        max_objects (int): Maksymalna liczba obiektów na obraz w zbiorze danych.
        max_batch_size (int): Maksymalny batch size.
        min_batch_size (int): Minimalny batch size.
        use_amp (bool): Czy używane jest mixed precision (torch.cuda.amp).
        is_training (bool): Czy estymacja dotyczy treningu (True) czy inferencji (False).

    Returns:
        int: Estymowany batch size.
    """
    if image_size is None:
        raise ValueError("image_size musi być podane jako argument w estimate_batch_size")

    image_memory = image_size[0] * image_size[1] * 3 * 4  # Obraz RGB (float32)
    masks_memory_per_image = max_objects * image_size[0] * image_size[1] * 1  # Maski (uint8)
    activations_memory_per_image = 0.5 * 1024 ** 3  # Aktywacje
    model_memory = 0.6 * 1024 ** 3  # Model
    cuda_overhead = 1.0 * 1024 ** 3  # Overhead CUDA
    amp_factor = 0.6 if use_amp else 1.0
    memory_per_image_gpu = (image_memory + masks_memory_per_image + activations_memory_per_image) * amp_factor

    if is_training:
        model_memory *= 3  # Gradienty i optymalizator
        memory_per_image_gpu *= 1.5  # Dodatkowe zużycie podczas treningu

    available_memory = psutil.virtual_memory().available
    cpu_count = os.cpu_count()
    usable_memory = available_memory * 0.7
    max_images_by_memory = int(usable_memory // (image_memory + masks_memory_per_image))
    max_images_by_cpu = cpu_count

    max_images_by_gpu = max_batch_size
    if torch.cuda.is_available():
        try:
            gpu_memory_total = torch.cuda.get_device_properties(0).total_memory
            gpu_memory_free = gpu_memory_total - torch.cuda.memory_reserved(0) - torch.cuda.memory_allocated(0)
            usable_gpu_memory = gpu_memory_free * 0.7
            total_memory_per_image = memory_per_image_gpu + (model_memory + cuda_overhead) / max_batch_size
            max_images_by_gpu = int(usable_gpu_memory // total_memory_per_image)
            logger.info("Dostępna pamięć GPU: %.2f GB, szacowane zużycie na obraz: %.2f MB, model: %.2f GB, max obiektów: %d",
                        gpu_memory_free / (1024 ** 3), total_memory_per_image / (1024 ** 2), model_memory / (1024 ** 3), max_objects)
        except Exception as e:
            logger.warning("Nie można sprawdzić pamięci GPU: %s. Używam konserwatywnego batch_size.", str(e))
            max_images_by_gpu = 1

    max_possible_batch_size = min(max_images_by_memory, max_images_by_cpu, max_images_by_gpu, max_batch_size)
    batch_size = max(min_batch_size, min(max_possible_batch_size, max_batch_size))

    logger.info("Estymowany batch_size: %d (pamięć RAM: %.2f GB, pamięć GPU: %.2f GB, CPU: %d rdzeni, AMP: %s, training: %s, max obiektów: %d)",
                batch_size, available_memory / (1024 ** 3), gpu_memory_free / (1024 ** 3) if torch.cuda.is_available() else 0,
                cpu_count, use_amp, is_training, max_objects)
    return batch_size

def get_data_loaders(train_dir, val_dir, batch_size=None, num_workers=3, num_augmentations=1, coco_train_path=None, coco_val_path=None, num_processes=4):
    """
    Tworzy DataLoader'y dla danych treningowych i walidacyjnych.

    Args:
        train_dir (str): Ścieżka do katalogu z danymi treningowymi (np. ../../data/train).
        val_dir (str): Ścieżka do katalogu z danymi walidacyjnymi (np. ../../data/val).
        batch_size (int, optional): Rozmiar partii (batch size). Jeśli None, estymowany automatycznie.
        num_workers (int): Liczba wątków dla DataLoadera. Domyślnie 4.
        num_augmentations (int): Liczba augmentacji na obraz. Domyślnie 1.
        coco_train_path (str): Ścieżka do pliku COCO z adnotacjami treningowymi.
        coco_val_path (str): Ścieżka do pliku COCO z adnotacjami walidacyjnymi.
        num_processes (int): Liczba procesów do równoległego wczytywania danych.

    Returns:
        tuple: (train_loader, val_loader) - DataLoader'y dla danych treningowych i walidacyjnych.
    """
    image_size = (1024, 1024)

    # Tworzenie datasetów, aby uzyskać max_objects
    if coco_train_path is None:
        coco_train_path = os.path.join(train_dir, "annotations", "instances_train.json")
    if coco_val_path is None:
        coco_val_path = os.path.join(val_dir, "annotations", "instances_val.json")

    train_dataset = RuryDataset(
        root=train_dir,
        split="train",
        image_size=image_size,
        augment=True,
        num_augmentations=num_augmentations,
        annotation_path=coco_train_path,
        num_processes=num_processes
    )

    val_dataset = RuryDataset(
        root=val_dir,
        split="val",
        image_size=image_size,
        augment=False,
        annotation_path=coco_val_path,
        num_processes=num_processes
    )

    # Użycie większej wartości max_objects z obu datasetów
    max_objects = max(getattr(train_dataset, 'max_objects', 400), getattr(val_dataset, 'max_objects', 400))

    if batch_size is None:
        batch_size = estimate_batch_size(
            image_size=image_size,
            max_objects=max_objects,  # Przekazujemy max_objects
            max_batch_size=8,
            min_batch_size=1,
            use_amp=True,
            is_training=True
        )

    available_memory = psutil.virtual_memory().available / (1024 ** 3)
    cpu_count = os.cpu_count()
    cpu_usage = psutil.cpu_percent(interval=1)

    if available_memory < 4:
        num_workers = min(num_workers, max(1, cpu_count // 2))
        logger.warning("Mało pamięci systemowej (%.2f GB), zmniejszam num_workers do %d", available_memory, num_workers)
    elif cpu_usage > 70:
        num_workers = min(num_workers, max(1, cpu_count // 2))
        logger.warning("Wysokie obciążenie CPU (%.1f%%), zmniejszam num_workers do %d", cpu_usage, num_workers)
    else:
        num_workers = min(num_workers, cpu_count)

    use_pin_memory = False
    if torch.cuda.is_available():
        try:
            gpu_memory_total = torch.cuda.get_device_properties(0).total_memory
            gpu_memory_free = gpu_memory_total - torch.cuda.memory_reserved(0) - torch.cuda.memory_allocated(0)
            model_memory = 1.8 * 1024 ** 3
            cuda_overhead = 1.0 * 1024 ** 3
            memory_per_image = (image_size[0] * image_size[1] * 3 * 4 +
                                max_objects * image_size[0] * image_size[1] * 1 +
                                0.5 * 1024 ** 3 * 0.6 * 1.5)
            estimated_memory_usage = model_memory + cuda_overhead + memory_per_image * batch_size
            if gpu_memory_free - estimated_memory_usage > 2.0 * 1024 ** 3 and available_memory > 8:
                use_pin_memory = False
                logger.info("Włączam pin_memory, dostępna pamięć GPU: %.2f GB, szacowane zużycie: %.2f GB, RAM: %.2f GB",
                            gpu_memory_free / (1024 ** 3), estimated_memory_usage / (1024 ** 3), available_memory)
            else:
                logger.warning("Za mało pamięci GPU (%.2f GB wolnej, szacowane zużycie: %.2f GB) lub RAM (%.2f GB), wyłączam pin_memory",
                               gpu_memory_free / (1024 ** 3), estimated_memory_usage / (1024 ** 3), available_memory)
        except Exception as e:
            logger.warning("Nie można sprawdzić pamięci GPU: %s. Wyłączam pin_memory.", str(e))

    logger.info("Używam batch_size=%d, %d wątków w DataLoader, pin_memory=%s, max obiektów: %d",
                batch_size, num_workers, use_pin_memory, max_objects)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=custom_collate_fn,
        pin_memory=False,
        prefetch_factor=2 if num_workers > 0 else None
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=custom_collate_fn,
        pin_memory=use_pin_memory,
        prefetch_factor=2 if num_workers > 0 else None
    )

    return train_loader, val_loader