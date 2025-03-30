import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.ops import nms
import cv2
import numpy as np
from SSD.ssd import build_ssd  # Import modelu SSD z repozytorium

# Niestandardowy Dataset dla Twoich danych
class PipeDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None, input_size=(300, 300)):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform
        self.input_size = input_size
        self.image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.png'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.image_files[idx])
        label_path = os.path.join(self.labels_dir, self.image_files[idx].replace('.jpg', '.txt').replace('.png', '.txt'))

        img = cv2.imread(img_path)
        img = cv2.resize(img, self.input_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        boxes = []
        labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    class_id, x, y, w, h = map(float, line.strip().split())
                    xmin = (x - w / 2) * self.input_size[0]
                    ymin = (y - h / 2) * self.input_size[1]
                    xmax = (x + w / 2) * self.input_size[0]
                    ymax = (y + h / 2) * self.input_size[1]
                    boxes.append([xmin, ymin, xmax, ymax])
                    labels.append(int(class_id) + 1)  # +1, bo tło ma indeks 0 w SSD

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)

        if self.transform:
            img = self.transform(img)

        target = {'boxes': boxes, 'labels': labels}
        return img, target

# Funkcja kolacji dla DataLoader
def custom_collate_fn(batch):
    images = []
    targets = []
    for img, target in batch:
        images.append(img)
        targets.append(target)
    images = torch.stack(images, dim=0)
    return images, targets

# Funkcja treningowa
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = PipeDataset("data/images/train/", "data/labels/train/", transform=train_transform)
    val_dataset = PipeDataset("data/images/val/", "data/labels/val/", transform=transforms.ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0, collate_fn=custom_collate_fn)

    model = build_ssd('train', size=300, num_classes=2)  # 2 klasy: tło + rury
    model.load_state_dict(torch.load("weights/ssd300_mAP_77.43_v2.pth", map_location=device))

    for param in model.vgg.parameters():
        param.requires_grad = False

    model.to(device)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0005, weight_decay=1e-4)

    num_epochs = 200
    best_loss = float('inf')
    patience = 20
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for images, targets in train_loader:
            images = images.to(device)
            boxes = [t['boxes'].to(device) for t in targets]
            labels = [t['labels'].to(device) for t in targets]

            optimizer.zero_grad()
            outputs = model(images)
            loss = model(images, targets={'boxes': boxes, 'labels': labels})['loss']
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}")

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                boxes = [t['boxes'].to(device) for t in targets]
                labels = [t['labels'].to(device) for t in targets]
                loss = model(images, targets={'boxes': boxes, 'labels': labels})['loss']
                val_loss += loss.item()
        val_loss /= len(val_loader)
        print(f"Validation Loss: {val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), "best_ssd_model.pth")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping")
                break

# Funkcja detekcji
def detect_images_in_folder(folder_path, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_ssd('test', size=300, num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()

    output_dir = "runs/detect/"
    os.makedirs(output_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    for image_name in os.listdir(folder_path):
        if not image_name.lower().endswith(('.jpg', '.png')):
            continue
        image_path = os.path.join(folder_path, image_name)
        img = cv2.imread(image_path)
        orig_img = img.copy()
        img = cv2.resize(img, (300, 300))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            detections = model(img)

        boxes = []
        scores = []
        for i in range(detections.size(1)):
            for det in detections[0, i]:
                if det[0] > 0.5:
                    score, x_min, y_min, x_max, y_max = det
                    boxes.append([x_min, y_min, x_max, y_max])
                    scores.append(score)

        if boxes:
            boxes = torch.tensor(boxes, dtype=torch.float32).to(device)
            scores = torch.tensor(scores, dtype=torch.float32).to(device)
            keep = nms(boxes, scores, iou_threshold=0.5)
            boxes = boxes[keep].cpu().numpy()
            scores = scores[keep].cpu().numpy()

            for box, score in zip(boxes, scores):
                x_min, y_min, x_max, y_max = map(int, [box[0] * orig_img.shape[1], box[1] * orig_img.shape[0], 
                                                       box[2] * orig_img.shape[1], box[3] * orig_img.shape[0]])
                cv2.rectangle(orig_img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(orig_img, f"Pipe: {score:.2f}", (x_min, y_min - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imwrite(os.path.join(output_dir, image_name), orig_img)

if __name__ == "__main__":
    train()
    detect_images_in_folder("data/images/test/", "best_ssd_model.pth")