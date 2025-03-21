import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from models.yolo import YOLO
from utils.dataset import YOLODataset
from utils.loss import YOLOLoss
import numpy as np
from models.yolo import YOLO

def generate_anchors(dataset, num_anchors=8, grid_sizes = [52, 26, 13, 7]):
    # Ręcznie zdefiniowane anchory
    anchors = torch.tensor([
        [0.03, 0.05],
        [0.05, 0.08],
        [0.07, 0.10],
        [0.02, 0.03],
        [0.04, 0.06],
        [0.08, 0.12],
        [0.06, 0.09],
        [0.10, 0.15]
    ], dtype=torch.float32)
    print(f"Używane anchory (width, height): {anchors.numpy()}")
    return anchors

def custom_collate_fn(batch):
    """
    Niestandardowa funkcja kolacji dla DataLoader.
    Zachowuje obrazy jako tensor [batch_size, C, H, W], a bounding boxy jako listę tensorów.
    """
    images = []
    targets = []
    
    for img, boxes in batch:
        images.append(img)
        targets.append(boxes)
    
    # Złóż obrazy w tensor [batch_size, C, H, W]
    images = torch.stack(images, dim=0)
    
    # Zachowaj targets jako listę tensorów
    return images, targets

def format_targets(targets, grid_sizes, num_anchors, num_classes, device):
    batch_size = len(targets)
    formatted_targets = []
    
    for scale_idx, grid_size in enumerate(grid_sizes):
        target_scale = torch.zeros(batch_size, grid_size, grid_size, num_anchors, 6, device=device)
        for b in range(batch_size):
            for box in targets[b]:
                if len(box) == 0:
                    continue
                class_id, x, y, w, h = box
                # Oblicz pozycję w siatce
                grid_x = int(x * grid_size)
                grid_y = int(y * grid_size)
                if grid_x >= grid_size or grid_y >= grid_size:
                    continue
                # Znajdź najlepszy anchor
                best_iou = 0
                best_anchor = 0
                for anchor_idx in range(num_anchors):
                    anchor_w, anchor_h = anchors[anchor_idx]
                    iou = min(w / anchor_w, anchor_w / w) * min(h / anchor_h, anchor_h / h)
                    if iou > best_iou:
                        best_iou = iou
                        best_anchor = anchor_idx
                # Wypełnij target
                # Normalizuj x, y do [0, 1] w obrębie komórki
                target_scale[b, grid_y, grid_x, best_anchor, 0] = (x * grid_size - grid_x)  # x w [0, 1]
                target_scale[b, grid_y, grid_x, best_anchor, 1] = (y * grid_size - grid_y)  # y w [0, 1]
                target_scale[b, grid_y, grid_x, best_anchor, 2] = torch.log(w / anchors[best_anchor, 0] + 1e-16)
                target_scale[b, grid_y, grid_x, best_anchor, 3] = torch.log(h / anchors[best_anchor, 1] + 1e-16)
                target_scale[b, grid_y, grid_x, best_anchor, 4] = 1.0  # Obiekt jest obecny
                target_scale[b, grid_y, grid_x, best_anchor, 5] = class_id
        formatted_targets.append(target_scale)
        print(f"Target shape for scale {scale_idx}: {target_scale.shape}")
    
    return formatted_targets

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Dataset i DataLoader
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_dataset = YOLODataset("data/images/train/", "data/labels/train/", transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0, collate_fn=custom_collate_fn)
    
    # Generowanie anchorów
    global anchors
    anchors = generate_anchors(train_dataset, num_anchors=8)  # 8 anchorów
    anchors = anchors.to(device)
    
    # Model
    model = YOLO(num_classes=1, num_anchors=8).to(device)  # 8 anchorów
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = YOLOLoss(lambda_coord=2.0, lambda_noobj=0.1).to(device)
    
    grid_sizes = [104, 52, 26, 13]
    
    num_epochs = 100  # Zwiększ do 100 epok
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_idx, (images, targets) in enumerate(train_loader):
            images = images.to(device)
            targets = [[box.to(device) for box in target] for target in targets]
            
            optimizer.zero_grad()
            outputs = model(images)
            
            # Formatowanie targetów
            formatted_targets = format_targets(targets, grid_sizes, num_anchors=8, num_classes=1, device=device)
            
            # Obliczanie straty
            loss = criterion(outputs, formatted_targets)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}")
        
        # Zapisz model
        torch.save(model.state_dict(), "yolo_model.pth")

if __name__ == "__main__":
    from sklearn.cluster import KMeans
    train()