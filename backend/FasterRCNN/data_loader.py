import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
import os
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_dataset_paths():
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset"))
    return {
        "train_images": os.path.join(base, "train/images"),
        "train_annotations": os.path.join(base, "train/annotations.json"),
        "val_images": os.path.join(base, "val/images"),
        "val_annotations": os.path.join(base, "val/annotations.json"),
        "test_images": os.path.join(base, "test/images")
    }

def collate_fn(batch):
    return tuple(zip(*batch))

def get_train_transform():
    return A.Compose([
        A.RandomScale(scale_limit=(-0.3, 0.0), p=0.5),
        A.SmallestMaxSize(max_size=1024, p=1.0),
        A.PadIfNeeded(min_height=1024, min_width=1024, border_mode=0, value=0, p=1.0),
        A.RandomCrop(height=1024, width=1024, p=0.4),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.RandomGamma(p=0.3),
        A.ISONoise(p=0.3),
        A.Blur(blur_limit=3, p=0.2),
        A.MotionBlur(blur_limit=3, p=0.2),
        A.ColorJitter(p=0.4),
        A.RandomRotate90(p=0.3),
        A.Resize(1024, 1024),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']))

def get_val_transform():
    return A.Compose([
        A.Resize(1024, 1024),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']))

class CocoDetectionWithAlbumentations(CocoDetection):
    def __init__(self, root, annFile, transform=None):
        super().__init__(root, annFile)
        self.transform = transform

    def __getitem__(self, index):
        image, target = super().__getitem__(index)
        boxes, labels = [], []
        for obj in target:
            x, y, w, h = obj["bbox"]
            if w > 0 and h > 0:
                boxes.append([x, y, x + w, y + h])
                labels.append(obj["category_id"])
        image_np = np.array(image).astype(np.float32) / 255.0
        transformed = self.transform(image=image_np, bboxes=boxes, category_ids=labels)
        return transformed["image"], {
            "boxes": torch.tensor(transformed["bboxes"], dtype=torch.float32),
            "labels": torch.tensor(transformed["category_ids"], dtype=torch.int64)
        }

class UnannotatedImageFolder(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.image_paths = sorted([
            os.path.join(root, f) for f in os.listdir(root) if f.endswith(('.jpg', '.png'))
        ])
        self.transform = transform

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index]).convert("RGB")
        image = np.array(image)
        if self.transform:
            image = self.transform(image=image)["image"]
        return image, {}

    def __len__(self):
        return len(self.image_paths)

def get_data_loaders(batch_size=2, num_workers=0):
    paths = get_dataset_paths()
    train_dataset = CocoDetectionWithAlbumentations(paths["train_images"], paths["train_annotations"], get_train_transform())
    val_dataset = CocoDetectionWithAlbumentations(paths["val_images"], paths["val_annotations"], get_val_transform()) if os.path.exists(paths["val_annotations"]) else None
    test_dataset = UnannotatedImageFolder(paths["test_images"], A.Compose([A.Resize(1024, 1024), ToTensorV2()]))

    print(f"DataLoader gotowy! Trening: {len(train_dataset)} | Walidacja: {len(val_dataset) if val_dataset else 0} | Test: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn) if val_dataset else None
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader