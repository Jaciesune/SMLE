import os
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import albumentations as A

class RuryDataset(Dataset):
    def __init__(self, dataset_dir, subset, image_size=(1024, 1024), augment=False, num_augmentations=1):
        self.dataset_dir = dataset_dir
        self.subset = subset
        self.image_size = image_size
        self.augment = augment
        self.num_augmentations = num_augmentations if augment else 1
        self.image_dir = os.path.join(dataset_dir, subset, "images")
        self.annotation_dir = os.path.join(dataset_dir, subset, "annotations")

        # Wczytaj adnotacje z JSON
        annotation_files = [f for f in os.listdir(self.annotation_dir) if f.endswith('.json')]
        if not annotation_files:
            raise ValueError(f"Brak plików JSON w {self.annotation_dir}")
        annotation_path = os.path.join(self.annotation_dir, annotation_files[0])
        with open(annotation_path, 'r') as f:
            self.annotations = json.load(f)

        self.image_ids = [img['id'] for img in self.annotations['images']]
        self.image_info = {img['id']: img for img in self.annotations['images']}

        # Transformacje podstawowe
        self.base_transform = T.Compose([
            T.ToTensor(),
        ])

        # Augmentacje z albumentations
        self.augment_transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=15, p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.HueSaturationValue(p=0.5),
            A.Resize(height=image_size[1], width=image_size[0]),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

    def __len__(self):
        return len(self.image_ids) * self.num_augmentations

    def __getitem__(self, idx):
        orig_idx = idx // self.num_augmentations
        aug_idx = idx % self.num_augmentations

        image_id = self.image_ids[orig_idx]
        image_info = self.image_info[image_id]
        image_path = os.path.join(self.image_dir, image_info['file_name'])

        # Wczytaj obraz
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Nie można wczytać obrazu: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Wczytaj adnotacje
        anns = [a for a in self.annotations['annotations'] if a['image_id'] == image_id]
        boxes = []
        masks = []
        labels = []

        for ann in anns:
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            if isinstance(ann['segmentation'][0], list):
                points = np.array([(int(x), int(y)) 
                                  for x, y in zip(ann['segmentation'][0][::2], ann['segmentation'][0][1::2])], 
                                  dtype=np.int32)
                mask = cv2.fillPoly(mask, [points], 1)
            masks.append(mask)
            labels.append(ann['category_id'])

        # Augmentacja
        if self.augment and aug_idx > 0:
            augmented = self.augment_transform(
                image=image,
                bboxes=boxes,
                masks=masks,
                labels=labels
            )
            image = augmented['image']
            boxes = augmented['bboxes']
            masks = augmented['masks']
            labels = augmented['labels']
        else:
            image = cv2.resize(image, self.image_size)
            scale_x = self.image_size[0] / image_info['width']
            scale_y = self.image_size[1] / image_info['height']
            boxes = [[int(x * scale_x), int(y * scale_y), int(x_max * scale_x), int(y_max * scale_y)] 
                     for x, y, x_max, y_max in boxes]
            masks = [cv2.resize(mask, self.image_size, interpolation=cv2.INTER_NEAREST) for mask in masks]

        # Konwersja na tensory
        image = self.base_transform(image)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.as_tensor(np.array(masks), dtype=torch.uint8)

        target = {
            'boxes': boxes,
            'labels': labels,
            'masks': masks,
            'image_id': torch.tensor([image_id]),
            'area': (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
            'iscrowd': torch.zeros((len(labels),), dtype=torch.int64)
        }

        return image, target

def custom_collate_fn(batch):
    """Funkcja do batchowania danych."""
    return tuple(zip(*batch))

def get_data_loaders(dataset_dir, batch_size=2, num_workers=4, num_augmentations=1):
    """Zwraca DataLoader dla zbiorów treningowego i walidacyjnego."""
    train_dataset = RuryDataset(dataset_dir, "train", image_size=(1024, 1024), augment=True, num_augmentations=num_augmentations)
    val_dataset = RuryDataset(dataset_dir, "val", image_size=(1024, 1024), augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=custom_collate_fn  # Użyto zdefiniowanej funkcji zamiast lambdy
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=custom_collate_fn
    )

    return train_loader, val_loader