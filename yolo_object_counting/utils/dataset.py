import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image

class YOLODataset(Dataset):
    def __init__(self, images_dir, labels_dir, input_size=(416, 416)):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.image_files = sorted(os.listdir(images_dir))
        self.input_size = input_size

        # Definicja augmentacji Albumentations
        self.aug_transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=10, p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Resize(*input_size),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)

        # Wczytanie obrazu
        image = cv2.imread(img_path)
        if image is None:
            raise Exception(f"Failed to read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h_orig, w_orig = image.shape[:2]

        # Wczytanie anotacji
        label_filename = os.path.splitext(self.image_files[idx])[0] + ".txt"
        label_path = os.path.join(self.labels_dir, label_filename)
        boxes = []
        class_labels = []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    cls, x_center, y_center, width, height = map(float, line.strip().split())
                    boxes.append([x_center, y_center, width, height])
                    class_labels.append(cls)

        # Zastosowanie augmentacji
        augmented = self.aug_transform(image=image, bboxes=boxes, class_labels=class_labels)
        image = augmented["image"].float() / 255.0  # Konwersja na float i normalizacja
        boxes = augmented["bboxes"]

        # Konwersja bounding boxów na tensor
        boxes_tensor = torch.tensor(
    [ [cls] + box for cls, box in zip(class_labels, boxes) ], dtype=torch.float32
) if boxes else torch.zeros((0, 5), dtype=torch.float32)

        labels_tensor = torch.tensor(class_labels, dtype=torch.long) if class_labels else torch.zeros((0,), dtype=torch.long)

        return {"image": image, "boxes": boxes_tensor, "labels": labels_tensor}
