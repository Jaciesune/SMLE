import os
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import albumentations as A
from pycocotools import mask as mask_utils
import logging

# Konfiguracja logowania
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class RuryDataset(Dataset):
    def __init__(self, root, split, image_size=(1024, 1024), augment=False, num_augmentations=1, annotation_path=None):
        """
        Args:
            root (str): Ścieżka do katalogu z danymi (np. /app/train_data lub /app/data/val).
            split (str): "train" lub "val".
            image_size (tuple): Rozmiar obrazów (wysokość, szerokość).
            augment (bool): Czy stosować augmentacje.
            num_augmentations (int): Liczba augmentacji na obraz.
            annotation_path (str, optional): Ścieżka do pliku COCO z adnotacjami. Jeśli None, używa domyślnej lokalizacji.
        """
        self.root = root
        self.split = split
        self.image_size = image_size
        self.augment = augment
        self.num_augmentations = num_augmentations if augment else 1
        self.image_dir = os.path.join(root, "images")  # Zakładamy, że obrazy są w podkatalogu "images"

        # Ustalanie ścieżki do pliku adnotacji
        if annotation_path is None:
            self.annotation_path = os.path.join(root, "annotations", "coco.json")
        else:
            self.annotation_path = annotation_path

        # Sprawdzenie, czy plik adnotacji istnieje
        if not os.path.exists(self.annotation_path):
            raise FileNotFoundError(f"Nie znaleziono pliku adnotacji: {self.annotation_path}")
        with open(self.annotation_path, 'r') as f:
            self.annotations = json.load(f)

        self.image_info = {img['id']: img for img in self.annotations['images']}

        # Filtruj obrazy: tylko te, które istnieją i mają adnotacje
        self.image_ids = []
        for img_id in self.image_info.keys():
            image_path = os.path.join(self.image_dir, self.image_info[img_id]['file_name'])
            # Sprawdź, czy obraz istnieje
            if not os.path.exists(image_path):
                logger.warning("Pomijam obraz, plik nie istnieje: %s", image_path)
                continue
            # Sprawdź, czy obraz ma adnotacje
            anns = [a for a in self.annotations['annotations'] if a['image_id'] == img_id]
            if not anns:
                logger.warning("Pomijam obraz bez adnotacji: %s", image_path)
                continue
            self.image_ids.append(img_id)

        logger.info("Załadowano %d obrazów z adnotacjami w %s", len(self.image_ids), self.image_dir)

        if not self.image_ids:
            raise ValueError(f"Brak obrazów z adnotacjami w {self.image_dir}")

        self.base_transform = T.Compose([T.ToTensor()])

        self.augment_transform = A.Compose([
            A.HorizontalFlip(p=0.5),  # Odbicie w poziomie
            A.VerticalFlip(p=0.5),  # Odbicie w pionie
            A.Rotate(limit=50, p=0.5),  # Obrót o maksymalnie 50 stopni
            A.RandomBrightnessContrast(p=0.5),  # Losowa zmiana jasności i kontrastu
            A.HueSaturationValue(p=0.5),  # Zmiana kolorów
            A.GaussNoise(p=0.2),  # Szum Gaussa
            A.MultiplicativeNoise(multiplier=[0.5, 1.5], per_channel=True, p=0.2),  # Szum mnożący
            A.Blur(blur_limit=(3, 7), p=0.2),  # Rozmazanie
            A.MedianBlur(blur_limit=5, p=0.1),  # Rozmazanie medianowe
            A.ISONoise(p=0.1),  # Szum ISO
            A.Resize(height=image_size[1], width=image_size[0]),  # Zmiana rozmiaru
        ], bbox_params=A.BboxParams(
            format='coco', 
            label_fields=['category_ids'],
            min_area=3,
            min_visibility=0.1
        ), additional_targets={'masks': 'masks'})

    def __len__(self):
        return len(self.image_ids) * self.num_augmentations

    def decode_rle(self, segmentation, bbox, target_size):
        """Dekoduje RLE do maski binarnej i pozycjonuje ją w granicach bboxa"""
        rle = {"counts": segmentation["counts"].encode('utf-8'), "size": segmentation["size"]}
        mask = mask_utils.decode(rle)  # Dekoduje do rozmiaru segmentation["size"]
        # Przeskaluj maskę do rozmiaru bboxa
        x, y, w, h = map(int, bbox)
        if w <= 0 or h <= 0:
            return np.zeros((target_size[0], target_size[1]), dtype=np.uint8)
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        # Utwórz pełnowymiarową maskę i umieść maskę w granicach bboxa
        full_mask = np.zeros((target_size[0], target_size[1]), dtype=np.uint8)
        x_end = min(x + w, target_size[1])
        y_end = min(y + h, target_size[0])
        full_mask[y:y_end, x:x_end] = mask[:y_end - y, :x_end - x]
        return full_mask

    def __getitem__(self, idx):
        orig_idx = idx // self.num_augmentations
        aug_idx = idx % self.num_augmentations

        image_id = self.image_ids[orig_idx]
        image_info = self.image_info[image_id]
        image_path = os.path.join(self.image_dir, image_info['file_name'])

        # Wczytaj obraz
        image = cv2.imread(image_path)
        if image is None:
            logger.warning("Nie można wczytać obrazu: %s, pomijam...", image_path)
            # Przejdź do następnego obrazu
            next_idx = (orig_idx + 1) % len(self.image_ids)
            if next_idx == orig_idx:  # Jeśli to jedyny obraz w zbiorze
                raise ValueError(f"Brak dostępnych obrazów w zbiorze danych: {self.image_dir}")
            return self.__getitem__(next_idx * self.num_augmentations + aug_idx)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_height, orig_width = image.shape[:2]

        # Wczytaj adnotacje
        anns = [a for a in self.annotations['annotations'] if a['image_id'] == image_id]
        if not anns:
            logger.warning("Brak adnotacji dla obrazu: %s, pomijam...", image_path)
            # Przejdź do następnego obrazu
            next_idx = (orig_idx + 1) % len(self.image_ids)
            if next_idx == orig_idx:
                raise ValueError(f"Brak obrazów z adnotacjami w zbiorze danych: {self.image_dir}")
            return self.__getitem__(next_idx * self.num_augmentations + aug_idx)

        boxes = [ann['bbox'] for ann in anns]  # [x, y, w, h]
        masks = [self.decode_rle(ann['segmentation'], ann['bbox'], (orig_height, orig_width)) 
                 if 'segmentation' in ann else None for ann in anns]
        labels = [ann['category_id'] for ann in anns]

        # Augmentacja
        if self.augment and aug_idx > 0:
            aug_data = {
                'image': image,
                'bboxes': boxes,
                'category_ids': labels,
                'masks': masks
            }
            augmented = self.augment_transform(**aug_data)
            image = augmented['image']
            aug_boxes = augmented['bboxes']
            aug_masks = augmented['masks']
            aug_labels = augmented['category_ids']

            # Filtruj bboxy i maski
            height, width = image.shape[:2]
            filtered_boxes = []
            filtered_labels = []
            filtered_masks = []

            for bbox, label, mask in zip(aug_boxes, aug_labels, aug_masks):
                x, y, w, h = map(int, bbox)
                if (x >= width or y >= height or x + w <= 0 or y + h <= 0):
                    continue
                x = max(0, min(x, width - 1))
                y = max(0, min(y, height - 1))
                w = min(w, width - x)
                h = min(h, height - y)
                if w > 0 and h > 0:
                    filtered_boxes.append([x, y, w, h])
                    filtered_labels.append(label)
                    filtered_masks.append(mask)

            boxes = filtered_boxes
            labels = filtered_labels
            masks = filtered_masks
        else:
            image = cv2.resize(image, self.image_size)
            scale_x = self.image_size[0] / orig_width
            scale_y = self.image_size[1] / orig_height
            boxes = [[x * scale_x, y * scale_y, w * scale_x, h * scale_y] 
                    for x, y, w, h in boxes]
            masks = [cv2.resize(mask, self.image_size, interpolation=cv2.INTER_NEAREST) if mask is not None else None 
                    for mask in masks]

        # Konwersja na tensory
        image = self.base_transform(image)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        if masks and all(m is not None for m in masks):
            masks = torch.stack([torch.as_tensor(m, dtype=torch.uint8) for m in masks], dim=0)
        else:
            masks = torch.zeros((0, self.image_size[1], self.image_size[0]), dtype=torch.uint8)

        # Konwersja bboxów na [x_min, y_min, x_max, y_max]
        if len(boxes) > 0:
            boxes = torch.stack([
                boxes[:, 0],              # x_min
                boxes[:, 1],              # y_min
                boxes[:, 0] + boxes[:, 2], # x_max
                boxes[:, 1] + boxes[:, 3]  # y_max
            ], dim=1)
            # Sprawdzanie poprawności bboxów
            invalid_boxes = (boxes[:, 2] <= boxes[:, 0]) | (boxes[:, 3] <= boxes[:, 1])
            if invalid_boxes.any():
                logger.warning("Niepoprawne bboxy w %s: %s", image_info['file_name'], boxes[invalid_boxes])
                boxes = boxes[~invalid_boxes]
                labels = labels[~invalid_boxes]
                masks = masks[~invalid_boxes] if masks.shape[0] > 0 else masks

        target = {
            'boxes': boxes,
            'labels': labels,
            'masks': masks,  # [N, H, W]
            'image_id': torch.tensor([image_id]),
            'area': (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) if len(boxes) > 0 else torch.tensor([]),
            'iscrowd': torch.zeros((len(labels),), dtype=torch.int64)
        }

        # Sprawdzanie NaN/Inf
        if torch.isnan(image).any() or torch.isinf(image).any():
            raise ValueError(f"NaN/Inf w obrazie: {image_info['file_name']}")
        if torch.isnan(boxes).any() or torch.isinf(boxes).any():
            raise ValueError(f"NaN/Inf w bboxach: {image_info['file_name']}")
        if torch.isnan(masks).any() or torch.isinf(masks).any():
            raise ValueError(f"NaN/Inf w maskach: {image_info['file_name']}")

        return image, target

def custom_collate_fn(batch):
    return tuple(zip(*batch))

def get_data_loaders(train_dir, val_dir, batch_size=2, num_workers=4, num_augmentations=1, coco_train_path=None, coco_val_path=None):
    """
    Tworzy DataLoader'y dla danych treningowych i walidacyjnych.

    Args:
        train_dir (str): Ścieżka do katalogu z danymi treningowymi (np. /app/train_data).
        val_dir (str): Ścieżka do katalogu z danymi walidacyjnymi (np. /app/data/val).
        batch_size (int): Rozmiar partii (batch size). Domyślnie 2.
        num_workers (int): Liczba wątków dla DataLoadera. Domyślnie 4.
        num_augmentations (int): Liczba augmentacji na obraz. Domyślnie 1.
        coco_train_path (str): Ścieżka do pliku COCO z adnotacjami treningowymi.
        coco_val_path (str): Ścieżka do pliku COCO z adnotacjami walidacyjnymi.

    Returns:
        tuple: (train_loader, val_loader) - DataLoader'y dla danych treningowych i walidacyjnych.
    """
    # Tworzenie datasetu treningowego
    train_dataset = RuryDataset(
        root=train_dir,
        split="train",
        image_size=(1024, 1024),
        augment=True,
        num_augmentations=num_augmentations,
        annotation_path=coco_train_path  # Przekazujemy ścieżkę do adnotacji
    )

    # Tworzenie datasetu walidacyjnego
    val_dataset = RuryDataset(
        root=val_dir,
        split="val",
        image_size=(1024, 1024),
        augment=False,
        annotation_path=coco_val_path  # Przekazujemy ścieżkę do adnotacji
    )

    # Tworzenie DataLoadera dla danych treningowych
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=custom_collate_fn
    )

    # Tworzenie DataLoadera dla danych walidacyjnych
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=custom_collate_fn
    )

    return train_loader, val_loader