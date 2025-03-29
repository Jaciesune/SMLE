
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
import os
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

def find_dataset_folder():
    possible_paths = ["dataset", "../dataset", "../../dataset"]
    for path in possible_paths:
        if os.path.exists(path):
            return path
    raise FileNotFoundError("Nie znaleziono folderu dataset!")

dataset_path = find_dataset_folder()
train_images = os.path.join(dataset_path, "train/images")
train_annotations = os.path.join(dataset_path, "train/annotations.json")
val_images = os.path.join(dataset_path, "val/images")
val_annotations = os.path.join(dataset_path, "val/annotations.json")
test_images = os.path.join(dataset_path, "test/images")

def collate_fn(batch):
    return tuple(zip(*batch))

def get_train_transform():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.7),
        A.RandomGamma(p=0.5),
        A.MotionBlur(blur_limit=5, p=0.4),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
        A.ISONoise(p=0.4),
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.5),
        A.Resize(height=1024, width=1024),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']))

def get_val_transform():
    return A.Compose([
        A.Resize(height=1024, width=1024),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']))

class CocoDetectionWithAlbumentations(CocoDetection):
    def __init__(self, root, annFile, transform=None):
        super().__init__(root, annFile)
        self.transform = transform

    def __getitem__(self, index):
        image, target = super().__getitem__(index)

        boxes = []
        labels = []
        for obj in target:
            x_min, y_min, width, height = obj["bbox"]
            x_max = x_min + width
            y_max = y_min + height

            if width > 0 and height > 0 and x_max > x_min and y_max > y_min:
                boxes.append([x_min, y_min, x_max, y_max])
                labels.append(obj["category_id"])

        image_np = np.array(image)
        transformed = self.transform(image=image_np, bboxes=boxes, category_ids=labels)
        image_tensor = transformed["image"]
        image_np = np.array(image)
        transformed = self.transform(image=image_np, bboxes=boxes, category_ids=labels)
        image_tensor = transformed["image"].float() / 255.0  # <- najważniejsze

        target = {
            "boxes": torch.as_tensor(transformed["bboxes"], dtype=torch.float32),
            "labels": torch.as_tensor(transformed["category_ids"], dtype=torch.int64)
        }
        return image_tensor, target

class UnannotatedImageFolder(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.image_paths = sorted([os.path.join(root, img) for img in os.listdir(root)
                                   if img.endswith(".jpg") or img.endswith(".png")])

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index]).convert("RGB")
        image = np.array(image)
        if self.transform:
            image = self.transform(image=image)["image"]
        return image, {}

    def __len__(self):
        return len(self.image_paths)

def get_data_loaders(batch_size=2, num_workers=0):
    train_dataset = CocoDetectionWithAlbumentations(
        root=train_images,
        annFile=train_annotations,
        transform=get_train_transform()
    )

    if os.path.exists(val_annotations):
        val_dataset = CocoDetectionWithAlbumentations(
            root=val_images,
            annFile=val_annotations,
            transform=get_val_transform()
        )
        print(f"Załadowano zbór walidacyjny: {len(val_dataset)} obrazów.")
    else:
        val_dataset = None
        print("Brak pliku `val/annotations.json`, pomijam walidację.")

    test_dataset = UnannotatedImageFolder(
        root=test_images,
        transform=A.Compose([
            A.Resize(height=1024, width=1024),
            ToTensorV2(transpose_mask=True)
        ])
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn) if val_dataset else None
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print(f"DataLoader gotowy! Trening: {len(train_dataset)} | Walidacja: {len(val_dataset) if val_dataset else 0} | Test: {len(test_dataset)}")
    return train_loader, val_loader, test_loader
