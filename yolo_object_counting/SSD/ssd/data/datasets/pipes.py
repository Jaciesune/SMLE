import os
import cv2
import numpy as np
from .base import BaseDataset

class PipesDataset(BaseDataset):
    def __init__(self, root, image_set, transform=None, target_transform=None):
        """
        Dataset dla danych w formacie YOLO (pipes).
        """
        super().__init__(transform=transform, target_transform=target_transform)
        self.root = root
        self.image_set = image_set  # "train" lub "val"
        self.img_dir = os.path.join(root, "images", image_set)
        self.ann_dir = os.path.join(root, "labels", image_set)

        if not os.path.exists(self.img_dir):
            raise FileNotFoundError(f"Folder obrazów nie istnieje: {self.img_dir}")
        if not os.path.exists(self.ann_dir):
            raise FileNotFoundError(f"Folder etykiet nie istnieje: {self.ann_dir}")

        self.image_files = [f for f in os.listdir(self.img_dir) if f.lower().endswith(('.jpg', '.png'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.image_files[index])
        label_path = os.path.join(self.ann_dir, self.image_files[index].replace('.jpg', '.txt').replace('.png', '.txt'))

        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Nie można wczytać obrazu: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        boxes = []
        labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    class_id, x, y, w, h = map(float, line.strip().split())
                    xmin = (x - w / 2) * img.shape[1]
                    ymin = (y - h / 2) * img.shape[0]
                    xmax = (x + w / 2) * img.shape[1]
                    ymax = (y + h / 2) * img.shape[0]
                    boxes.append([xmin, ymin, xmax, ymax])
                    labels.append(int(class_id) + 1)  # +1, bo tło ma indeks 0
        else:
            pass  # Puste etykiety dla obrazów bez rur

        boxes = np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32)
        labels = np.array(labels, dtype=np.int64) if labels else np.zeros((0,), dtype=np.int64)

        if self.transform:
            img, boxes, labels = self.transform(img, boxes, labels)
        if self.target_transform:
            boxes, labels = self.target_transform(boxes, labels)

        # Tworzymy słownik targets
        targets = {
            "boxes": boxes,
            "labels": labels
        }

        # Zwracamy tuple: obraz, słownik targets, ID obrazu (tutaj indeks)
        return img, targets, index

def register_pipes_datasets():
    from ssd.config.path_catlog import DatasetCatalog
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "data"))
    for image_set in ["train", "val"]:
        DatasetCatalog.register(
            image_set,
            lambda transform=None, target_transform=None, image_set=image_set: PipesDataset(
                root, image_set, transform=transform, target_transform=target_transform
            )
        )

# Wywołanie rejestracji nie jest już potrzebne tutaj, bo robi to __init__.py
register_pipes_datasets()