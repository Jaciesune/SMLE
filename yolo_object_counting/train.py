import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from models.yolo import YOLO
from utils.dataset import YOLODataset
from utils.loss import YOLOLoss
from torchvision import transforms
import matplotlib.pyplot as plt
import random
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
loss_history = []


def collate_fn(batch):
    images = [item["image"] for item in batch]
    boxes = [item["boxes"] for item in batch]

    images = torch.stack(images, dim=0)
    return {"image": images, "boxes": boxes}

def encode_targets(targets, grid_size, num_anchors, num_classes):
    batch_size = len(targets)
    target_tensor = torch.zeros((batch_size, grid_size, grid_size, num_anchors, 5 + num_classes))

    for b, boxes in enumerate(targets):
        for box in boxes:
            cls, x_center, y_center, width, height = box

            # Calculate grid cell indices
            grid_x = int(x_center * grid_size)
            grid_y = int(y_center * grid_size)

            # Ensure grid indices are within valid range
            grid_x = min(grid_size - 1, max(0, grid_x))
            grid_y = min(grid_size - 1, max(0, grid_y))

            # Choose the best anchor (simplified for now)
            best_anchor = 0

            # Assign values to the target tensor
            target_tensor[b, grid_y, grid_x, best_anchor, 0] = x_center
            target_tensor[b, grid_y, grid_x, best_anchor, 1] = y_center
            target_tensor[b, grid_y, grid_x, best_anchor, 2] = width
            target_tensor[b, grid_y, grid_x, best_anchor, 3] = height
            target_tensor[b, grid_y, grid_x, best_anchor, 4] = 1  # Confidence
            target_tensor[b, grid_y, grid_x, best_anchor, 5 + int(cls)] = 1  # Class
    return target_tensor

def visualize_predictions(model, dataloader, device):
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            outputs = model(images).cpu().numpy()

            for i in range(len(images)):
                image = images[i].permute(1, 2, 0).cpu().numpy()
                predictions = outputs[i]

                plt.imshow(image)

                for anchor in range(predictions.shape[2]):
                    for grid_y in range(predictions.shape[0]):
                        for grid_x in range(predictions.shape[1]):
                            if predictions.shape[-1] < 5:
                                continue
                            
                            pred_box = predictions[grid_y, grid_x, anchor]
                            confidence = torch.sigmoid(torch.tensor(pred_box[4])).item()  # Poprawna normalizacja

                            if confidence > 0.7:  # Zwiększono próg do 0.7
                                x_center_norm, y_center_norm, width_norm, height_norm = pred_box[:4]
                                h, w, _ = image.shape
                                x_center = int(x_center_norm * w)
                                y_center = int(y_center_norm * h)
                                width = int(width_norm * w)
                                height = int(height_norm * h)

                                x_min = x_center - width // 2
                                y_min = y_center - height // 2
                #                 rect = plt.Rectangle((x_min, y_min), width, height,
                #                                      color='green', fill=False)
                #                 plt.gca().add_patch(rect)
                #                 plt.text(x_min, y_min - 5, f"Conf: {confidence:.2f}", color='green', fontsize=8)

                # plt.title("Predykcje na zestawie testowym")
                # plt.show()
                break

def transform_annotations(boxes, image_width, image_height, transform):
    """Aktualizuje bounding boxy po transformacjach."""
    new_boxes = []
    for box in boxes:
        cls_id, x_center, y_center, width, height = box

        # Konwersja do formatu (x1, y1, x2, y2)
        x1 = (x_center - width / 2) * image_width
        y1 = (y_center - height / 2) * image_height
        x2 = (x_center + width / 2) * image_width
        y2 = (y_center + height / 2) * image_height

        # Transformacja bounding boxa (odkomentowano i poprawiono)
        transformed_box = transform.apply_box([x1, y1, x2, y2])
        new_x_center = ((transformed_box[0] + transformed_box[2]) / 2) / image_width
        new_y_center = ((transformed_box[1] + transformed_box[3]) / 2) / image_height
        new_width = (transformed_box[2] - transformed_box[0]) / image_width
        new_height = (transformed_box[3] - transformed_box[1]) / image_height
        new_boxes.append([cls_id, new_x_center, new_y_center, new_width, new_height])
    return new_boxes

def train(epochs=50, batch_size=16, lr=5e-5, input_size=(416, 416)):
    train_images = 'data/images/train'
    train_labels = 'data/labels/train'

    dataset = YOLODataset(train_images, train_labels, input_size=input_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)

    model = YOLO(num_classes=1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = YOLOLoss(lambda_coord=5, lambda_noobj=0.5)

    best_loss = float('inf')
    patience = 10  # Liczba epok bez poprawy przed wczesnym zatrzymaniem
    trigger_times = 0

    for epoch in range(epochs):
        running_loss = 0.0
        for batch in dataloader:
            images = batch['image'].to(device)
            targets = batch['boxes']

            optimizer.zero_grad()
            outputs = model(images)

            grid_size = 416 // 16
            formatted_targets = encode_targets(targets, grid_size=grid_size, num_anchors=3, num_classes=1).to(device)

            loss = criterion(outputs, formatted_targets)
            if torch.isnan(loss) or torch.isinf(loss):
                print("Warning: Loss is NaN or Inf, skipping this batch!")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader) if running_loss > 0 else 0.0
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
        loss_history.append(avg_loss)

        # Wczesne zatrzymanie
        if avg_loss < best_loss:
            best_loss = avg_loss
            trigger_times = 0
            torch.save(model.state_dict(), "yolo_model.pth")
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print(f"Wczesne zatrzymanie na epoce {epoch+1}!")
                break

        if epoch % 2 == 0:
            visualize_predictions(model, dataloader, device)

    plt.plot(range(1, len(loss_history) + 1), loss_history, marker='o', linestyle='-')
    plt.xlabel("Epoka")
    plt.ylabel("Loss")
    plt.title("Zmiana lossu podczas treningu")
    plt.grid()
    plt.show()

if __name__ == '__main__':
    # Ustawienie ziarna losowości dla powtarzalności wyników
    seed_value = 42
    torch.manual_seed(seed_value)
    np.random.seed(seed_value)
    random.seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

    train()
