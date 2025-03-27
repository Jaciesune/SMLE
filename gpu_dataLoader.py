import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from torchvision import transforms
import os
from PIL import Image
from torchvision.transforms import functional as F
import random
import torchvision.transforms.v2 as transforms_v2
import numpy as np
import torchvision.transforms.functional as TF
import cv2

def collate_fn(batch):
    images, targets = zip(*batch)
    images = list(images)
    valid_targets = []
    for target in targets:
        if isinstance(target, dict):
            valid_targets.append(target)
        else:
            boxes = []
            labels = []
            for obj in target:
                bbox = obj["bbox"]
                label = obj["category_id"]
                x_min, y_min, width, height = bbox
                x_max = x_min + width
                y_max = y_min + height
                if width > 0 and height > 0 and x_max > x_min and y_max > y_min:
                    boxes.append([x_min, y_min, x_max, y_max])
                    labels.append(label)
            valid_targets.append({
                "boxes": torch.tensor(boxes, dtype=torch.float32),
                "labels": torch.tensor(labels, dtype=torch.int64),
            })
    return images, valid_targets

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

class RandomPerspectiveWithBoxes:
    def __init__(self, distortion_scale=0.5, p=0.5):
        self.distortion_scale = distortion_scale
        self.p = p

    def _get_perspective_matrix(self, startpoints, endpoints):
        startpoints = np.array(startpoints, dtype=np.float32)
        endpoints = np.array(endpoints, dtype=np.float32)
        matrix = cv2.getPerspectiveTransform(startpoints, endpoints)
        return matrix

    def _transform_boxes(self, boxes, matrix):
        transformed_boxes = []
        for box in boxes:
            x0, y0, x1, y1 = box.tolist()
            corners = np.array([
                [x0, y0], [x1, y0], [x1, y1], [x0, y1]
            ], dtype=np.float32)
            ones = np.ones((corners.shape[0], 1), dtype=np.float32)
            corners_hom = np.concatenate([corners, ones], axis=1)
            transformed = np.dot(matrix, corners_hom.T).T
            transformed = transformed[:, :2] / transformed[:, 2:]

            x_min, y_min = transformed.min(axis=0)
            x_max, y_max = transformed.max(axis=0)

            if x_max > x_min and y_max > y_min:
                transformed_boxes.append([x_min, y_min, x_max, y_max])
        return transformed_boxes

    def __call__(self, image, target):
        if random.random() < self.p and len(target["boxes"]):
            width, height = image.size
            startpoints, endpoints = transforms_v2.RandomPerspective.get_params(width, height, self.distortion_scale)
            image = F.perspective(image, startpoints, endpoints, interpolation=Image.BILINEAR)
            matrix = self._get_perspective_matrix(startpoints, endpoints)
            new_boxes = self._transform_boxes(target["boxes"], matrix)
            if new_boxes:
                target["boxes"] = torch.tensor(new_boxes, dtype=torch.float32)
        return image, target

def custom_transform(image, target):
    image, target = RandomPerspectiveWithBoxes(p=0.3)(image, target)
    image = F.to_tensor(image)
    return image, target

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

try:
    from pycocotools.coco import COCO
except ImportError:
    raise ImportError("Brakuje biblioteki pycocotools. Zainstaluj ja: pip install pycocotools")

train_dataset = CocoDetectionWithTransform(root=train_images, annFile=train_annotations, transform=custom_transform)

if os.path.exists(val_annotations):
    val_dataset = CocoDetectionWithTransform(root=val_images, annFile=val_annotations, transform=custom_transform)
    print(f"Załadowano zbór walidacyjny: {len(val_dataset)} obrazów.")
else:
    val_dataset = None
    print("Brak pliku `val/annotations.json`, pomijam walidację.")

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
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)

    if val_dataset:
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
    else:
        val_loader = None

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print(f"DataLoader gotowy! Trening: {len(train_dataset)} | Walidacja: {len(val_dataset) if val_dataset else 0} | Test: {len(test_dataset)}")
    return train_loader, val_loader, test_loader