import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from torchvision import transforms
import os
from PIL import Image

def collate_fn(batch):
    images, targets = zip(*batch)
    images = list(images)
    
    valid_targets = []
    for target in targets:
        if isinstance(target, list): 
            boxes = []
            labels = []
            for obj in target:
                bbox = obj["bbox"]
                label = obj["category_id"]
                
                x_min, y_min, width, height = bbox
                x_max = x_min + width
                y_max = y_min + height

                if width > 0 and height > 0:
                    boxes.append([x_min, y_min, x_max, y_max])
                    labels.append(label)

            valid_targets.append({
                "boxes": torch.tensor(boxes, dtype=torch.float32),
                "labels": torch.tensor(labels, dtype=torch.int64),
            })
        else:
            valid_targets.append(target)

    return images, valid_targets

def find_dataset_folder():
    possible_paths = ["dataset", "../dataset", "../../dataset"]
    for path in possible_paths:
        if os.path.exists(path):
            return path
    raise FileNotFoundError("Nie znaleziono folderu dataset!")

dataset_path = find_dataset_folder()

# Ścieżki do danych
train_images = os.path.join(dataset_path, "train/images")
train_annotations = os.path.join(dataset_path, "train/annotations.json")

val_images = os.path.join(dataset_path, "val/images")
val_annotations = os.path.join(dataset_path, "val/annotations.json")

test_images = os.path.join(dataset_path, "test/images")

transform = transforms.Compose([
    transforms.ToTensor(),
])

# Wczytywanie train_dataset
train_dataset = CocoDetection(root=train_images, annFile=train_annotations, transform=transform)

# Wczytywanie val_dataset jeśli istnieje
if os.path.exists(val_annotations):
    val_dataset = CocoDetection(root=val_images, annFile=val_annotations, transform=transform)
    print(f"Załadowano zbiór walidacyjny: {len(val_dataset)} obrazów.")
else:
    val_dataset = None
    print("Brak pliku `val/annotations.json`, pomijam walidację.")

# Wczytywanie test_dataset bez adnotacji
class UnannotatedImageFolder(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.image_paths = sorted([os.path.join(root, img) for img in os.listdir(root) if img.endswith(".jpg") or img.endswith(".png")])

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index]).convert("RGB")
        image = transforms.functional.pil_to_tensor(image) / 255.0
        if self.transform:
            image = self.transform(image)
        return image, {}

    def __len__(self):
        return len(self.image_paths)

test_dataset = UnannotatedImageFolder(root=test_images, transform=transform)

def get_data_loaders(batch_size=2, num_workers=0):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)

    if val_dataset:
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
    else:
        val_loader = None

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print(f"DataLoader gotowy! Trening: {len(train_dataset)} | Walidacja: {len(val_dataset) if val_dataset else 0} | Test: {len(test_dataset)}")
    return train_loader, val_loader, test_loader
