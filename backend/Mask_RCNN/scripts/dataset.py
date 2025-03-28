import os
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import albumentations as A
from pycocotools import mask as mask_utils

class RuryDataset(Dataset):
    def __init__(self, dataset_dir, subset, image_size=(1024, 1024), augment=False, num_augmentations=1):
        self.dataset_dir = dataset_dir
        self.subset = subset
        self.image_size = image_size
        self.augment = augment
        self.num_augmentations = num_augmentations if augment else 1
        self.image_dir = os.path.join(dataset_dir, subset, "images")
        self.annotation_path = os.path.join(dataset_dir, subset, "annotations", "coco.json")

        if not os.path.exists(self.annotation_path):
            raise FileNotFoundError(f"Nie znaleziono pliku adnotacji: {self.annotation_path}")
        with open(self.annotation_path, 'r') as f:
            self.annotations = json.load(f)

        self.image_info = {img['id']: img for img in self.annotations['images']}
        self.image_ids = list(self.image_info.keys())

        self.base_transform = T.Compose([T.ToTensor()])

        self.augment_transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=20, p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.HueSaturationValue(p=0.5),
            A.Resize(height=image_size[1], width=image_size[0]),
        ], bbox_params=A.BboxParams(
            format='coco', 
            label_fields=['category_ids'],
            min_area=8,
            min_visibility=0.2
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
            raise ValueError(f"Nie można wczytać obrazu: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_height, orig_width = image.shape[:2]

        # Wczytaj adnotacje
        anns = [a for a in self.annotations['annotations'] if a['image_id'] == image_id]
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
                print(f"Niepoprawne bboxy w {image_info['file_name']}: {boxes[invalid_boxes]}")
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

def get_data_loaders(dataset_dir, batch_size=2, num_workers=4, num_augmentations=1):
    train_dataset = RuryDataset(
        dataset_dir, "train", image_size=(1024, 1024), augment=True, num_augmentations=num_augmentations
    )
    val_dataset = RuryDataset(
        dataset_dir, "val", image_size=(1024, 1024), augment=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=custom_collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=custom_collate_fn
    )

    return train_loader, val_loader

if __name__ == "__main__":
    dataset_dir = "../data"
    train_loader, val_loader = get_data_loaders(dataset_dir, batch_size=2, num_workers=4, num_augmentations=3)

    for images, targets in train_loader:
        print(f"Batch size: {len(images)}")
        print(f"Image shape: {images[0].shape}")
        print(f"Target keys: {targets[0].keys()}")
        print(f"Mask shape: {targets[0]['masks'].shape}")
        break