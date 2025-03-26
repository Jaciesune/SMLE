import os
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

class RuryDataset(Dataset):
    def __init__(self, dataset_dir, subset, image_size=(1024, 1024), augment=False):
        """
        dataset_dir: katalog główny danych (np. "../data")
        subset: "train" lub "val"
        image_size: rozmiar docelowy obrazów
        augment: czy stosować augmentację (opcjonalnie)
        """
        self.dataset_dir = dataset_dir
        self.subset = subset
        self.image_size = image_size
        self.augment = augment
        self.image_dir = os.path.join(dataset_dir, subset, "images")
        self.annotation_dir = os.path.join(dataset_dir, subset, "annotations")

        # Wczytaj adnotacje z JSON
        annotation_files = [f for f in os.listdir(self.annotation_dir) if f.endswith('.json')]
        if not annotation_files:
            raise ValueError(f"Brak plików JSON w {self.annotation_dir}")
        annotation_path = os.path.join(self.annotation_dir, annotation_files[0])
        with open(annotation_path, 'r') as f:
            self.annotations = json.load(f)

        # Lista ID obrazów
        self.image_ids = [img['id'] for img in self.annotations['images']]
        self.image_info = {img['id']: img for img in self.annotations['images']}

        # Transformacje (tylko zmiana rozmiaru na razie)
        self.transform = T.Compose([
            T.ToTensor(),  # Konwersja na tensor [C, H, W], normalizacja do [0, 1]
        ])

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_info = self.image_info[image_id]
        image_path = os.path.join(self.image_dir, image_info['file_name'])

        # Wczytaj obraz
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Nie można wczytać obrazu: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_height, orig_width = image.shape[:2]

        # Przeskaluj obraz
        image = cv2.resize(image, self.image_size)

        # Wczytaj adnotacje dla tego obrazu
        anns = [a for a in self.annotations['annotations'] if a['image_id'] == image_id]
        boxes = []
        masks = []
        labels = []

        for ann in anns:
            # Bounding box
            x, y, w, h = ann['bbox']
            scale_x = self.image_size[0] / orig_width
            scale_y = self.image_size[1] / orig_height
            x_min = int(x * scale_x)
            y_min = int(y * scale_y)
            x_max = int((x + w) * scale_x)
            y_max = int((y + h) * scale_y)
            boxes.append([x_min, y_min, x_max, y_max])

            # Maska (segmentacja)
            mask = np.zeros((orig_height, orig_width), dtype=np.uint8)
            if isinstance(ann['segmentation'][0], list):  # Poligon
                points = np.array([(int(x * scale_x), int(y * scale_y)) 
                                  for x, y in zip(ann['segmentation'][0][::2], ann['segmentation'][0][1::2])], 
                                  dtype=np.int32)
                mask = cv2.fillPoly(mask, [points], 1)
            mask = cv2.resize(mask, self.image_size, interpolation=cv2.INTER_NEAREST)
            masks.append(mask)

            # Etykieta (zakładam klasę "rura" = 1, tło = 0 jest implicite)
            labels.append(ann['category_id'])

        # Konwersja na tensory
        image = self.transform(image)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.as_tensor(np.array(masks), dtype=torch.uint8)

        # Target w formacie wymaganym przez Mask R-CNN
        target = {
            'boxes': boxes,
            'labels': labels,
            'masks': masks,
            'image_id': torch.tensor([image_id]),
            'area': (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
            'iscrowd': torch.zeros((len(labels),), dtype=torch.int64)
        }

        return image, target

def get_data_loaders(dataset_dir, batch_size=2, num_workers=4):
    """Zwraca DataLoader dla zbiorów treningowego i walidacyjnego."""
    from torch.utils.data import DataLoader

    train_dataset = RuryDataset(dataset_dir, "train", image_size=(512, 512), augment=True)
    val_dataset = RuryDataset(dataset_dir, "val", image_size=(512, 512), augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=lambda x: tuple(zip(*x))  # Specjalna funkcja do batchowania
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda x: tuple(zip(*x))
    )

    return train_loader, val_loader