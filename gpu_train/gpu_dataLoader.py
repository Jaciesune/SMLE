import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from torchvision import transforms
import os
from PIL import Image, ImageFilter
from torchvision.transforms import functional as F
import random
import torchvision.transforms.v2 as transforms_v2
import numpy as np

# --- AUGMENTACJE DODATKOWE ---
class RandomColorJitter:
    def __init__(self, brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.5):
        self.transform = transforms.ColorJitter(brightness, contrast, saturation, hue)
        self.p = p

    def __call__(self, image, target):
        if random.random() < self.p:
            image = self.transform(image)
        return image, target

class RandomGaussianBlur:
    def __init__(self, p=0.3):
        self.p = p

    def __call__(self, image, target):
        if random.random() < self.p:
            image = image.filter(ImageFilter.GaussianBlur(radius=1))
        return image, target

class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, target):
        if random.random() < self.p:
            image = F.hflip(image)
            w, _ = image.size
            boxes = target["boxes"]
            boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
            target["boxes"] = boxes
        return image, target

class RandomRotation:
    def __init__(self, degrees=5, p=0.3):
        self.degrees = degrees
        self.p = p

    def __call__(self, image, target):
        if random.random() < self.p:
            angle = random.uniform(-self.degrees, self.degrees)
            image = F.rotate(image, angle)
        return image, target

class RandomBrightness:
    def __init__(self, factor_range=(0.7, 1.3), p=0.4):
        self.factor_range = factor_range
        self.p = p

    def __call__(self, image, target):
        if random.random() < self.p:
            factor = random.uniform(*self.factor_range)
            image = F.adjust_brightness(image, factor)
        return image, target

class RandomNoise:
    def __init__(self, p=0.3):
        self.p = p

    def __call__(self, image, target):
        if random.random() < self.p:
            np_img = np.array(image)
            noise = np.random.normal(0, 5, np_img.shape).astype(np.uint8)
            np_img = np.clip(np_img + noise, 0, 255).astype(np.uint8)
            image = Image.fromarray(np_img)
        return image, target

# --- KOMBINACJA AUGMENTACJI ---
def custom_transform(image, target):
    for aug in [
        RandomHorizontalFlip(p=0.5),
        RandomRotation(degrees=5, p=0.3),
        RandomColorJitter(p=0.5),
        RandomBrightness(p=0.4),
        RandomGaussianBlur(p=0.2),
        RandomNoise(p=0.2),
    ]:
        image, target = aug(image, target)
    image = F.to_tensor(image)
    return image, target

# --- COCO DANE ---
def collate_fn(batch):
    return tuple(zip(*batch))

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

class CocoDetectionWithTransform(CocoDetection):
    def __init__(self, root, annFile, transform=None):
        super().__init__(root, annFile)
        self.transform = transform

    def __getitem__(self, index):
        image, anns = super().__getitem__(index)
        boxes = []
        labels = []
        for obj in anns:
            bbox = obj["bbox"]
            label = obj["category_id"]
            x_min, y_min, width, height = bbox
            x_max = x_min + width
            y_max = y_min + height
            if width > 0 and height > 0 and x_max > x_min and y_max > y_min:
                boxes.append([x_min, y_min, x_max, y_max])
                labels.append(label)
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
        }
        if self.transform is not None:
            image, target = self.transform(image, target)
        return image, target

class UnannotatedImageFolder(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.image_paths = sorted([os.path.join(root, img) for img in os.listdir(root)
                                   if img.endswith(".jpg") or img.endswith(".png")])

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index]).convert("RGB")
        image = transforms.functional.pil_to_tensor(image) / 255.0
        if self.transform:
            image = self.transform(image)
        return image, {}

    def __len__(self):
        return len(self.image_paths)

test_dataset = UnannotatedImageFolder(root=test_images, transform=lambda x: F.to_tensor(x))

def get_data_loaders(batch_size=2, num_workers=0):
    train_dataset = CocoDetectionWithTransform(root=train_images, annFile=train_annotations, transform=custom_transform)
    if os.path.exists(val_annotations):
        val_dataset = CocoDetectionWithTransform(root=val_images, annFile=val_annotations, transform=custom_transform)
        print(f"Załadowano zbór walidacyjny: {len(val_dataset)} obrazów.")
    else:
        val_dataset = None
        print("Brak pliku `val/annotations.json`, pomijam walidację.")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn) if val_dataset else None
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print(f"DataLoader gotowy! Trening: {len(train_dataset)} | Walidacja: {len(val_dataset) if val_dataset else 0} | Test: {len(test_dataset)}")
    return train_loader, val_loader, test_loader
