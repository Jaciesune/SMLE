import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from torchvision import transforms
import os

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

train_images = os.path.join(dataset_path, "train/images")
train_annotations = os.path.join(dataset_path, "train/annotations.json")
test_images = os.path.join(dataset_path, "test/images")
test_annotations = os.path.join(dataset_path, "test/annotations.json")

transform = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = CocoDetection(root=train_images, annFile=train_annotations, transform=transform)
test_dataset = CocoDetection(root=test_images, annFile=test_annotations, transform=transform)

def get_data_loaders(batch_size=2, num_workers=0):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)

    print(f"DataLoader gotowy! Liczba obrazów w treningu: {len(train_dataset)}, test: {len(test_dataset)}")
    return train_loader, test_loader
