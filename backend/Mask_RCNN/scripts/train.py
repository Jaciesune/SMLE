import torch
import torchvision.models.detection
import torch.optim as optim
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import time
import argparse
from datetime import datetime
from dataset import get_data_loaders

# KONFIGURACJA
CONFIDENCE_THRESHOLD = 0.7  # Próg pewności dla detekcji
NMS_THRESHOLD = 40000  # Ilość propozycji przed NMS
DETECTION_PER_IMAGE = 200  # Maksymalna liczba detekcji na obraz

# Pobranie modelu Mask R-CNN (wersja v2)
def get_model(num_classes, device):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(
        in_channels=256, dim_reduced=256, num_classes=num_classes
    )

    # Ustawienia NMS i progu pewności
    if isinstance(model.rpn.pre_nms_top_n, dict):
        model.rpn.pre_nms_top_n["training"] = NMS_THRESHOLD
        model.rpn.pre_nms_top_n["testing"] = NMS_THRESHOLD
        model.rpn.post_nms_top_n["training"] = NMS_THRESHOLD
        model.rpn.post_nms_top_n["testing"] = NMS_THRESHOLD
    model.roi_heads.score_thresh = CONFIDENCE_THRESHOLD
    model.roi_heads.detections_per_img = DETECTION_PER_IMAGE  # Maksymalna liczba detekcji na obraz
    model.to(device)
    print(f"Model działa na: {device}")
    return model

# Funkcja walidacji
def validate_model(model, dataloader, device, epoch, model_name):
    model.train()  # Tryb treningowy do obliczania strat
    total_val_loss = 0
    total_pred_objects = 0
    total_gt_objects = 0
    save_path = f"val/{model_name}/epoch_{epoch:02d}"
    os.makedirs(save_path, exist_ok=True)

    for idx, (images, targets) in enumerate(dataloader):
        images = [img.to(device) for img in images]
        new_targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Obliczanie straty
        loss_dict = model(images, new_targets)
        val_loss = sum(loss for loss in loss_dict.values())
        total_val_loss += val_loss.item()

        # Przełączenie w tryb ewaluacji dla predykcji
        model.eval()
        with torch.no_grad():
            outputs = model(images)

        for i, (image, output, target) in enumerate(zip(images, outputs, targets)):
            image_np = (image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

            pred_count = 0
            gt_count = target["boxes"].shape[0]
            total_gt_objects += gt_count

            for box, score, mask in zip(output["boxes"], output["scores"], output["masks"]):
                if score >= CONFIDENCE_THRESHOLD:
                    x_min, y_min, x_max, y_max = map(int, box.cpu().numpy())
                    cv2.rectangle(image_np, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    mask = (mask.cpu().numpy()[0] > 0.5).astype(np.uint8) * 255
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(image_np, contours, -1, (0, 0, 255), 2)
                    pred_count += 1

            total_pred_objects += pred_count
            filename = f"{save_path}/img_{idx}_{i}.png"
            cv2.imwrite(filename, image_np)

        model.train()  # Wróć do trybu treningowego

    avg_val_loss = total_val_loss / len(dataloader) if len(dataloader) > 0 else 0
    return avg_val_loss, total_pred_objects, total_gt_objects

# Trening jednej epoki
def train_one_epoch(model, dataloader, optimizer, device, epoch):
    model.train()
    total_loss = 0

    print(f"\nEpoka {epoch}... (Batchy: {len(dataloader)})")

    for batch_idx, (images, targets) in enumerate(dataloader):
        images = [img.to(device) for img in images]
        new_targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, new_targets)
        loss = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        print(f"Batch {batch_idx+1}/{len(dataloader)} - Strata: {loss.item():.4f}")

    return total_loss / len(dataloader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trening Mask R-CNN (v2)")
    parser.add_argument("--dataset_dir", type=str, default="../data", help="Ścieżka do danych")
    parser.add_argument("--num_workers", type=int, default=4, help="Liczba wątków dla DataLoadera")
    parser.add_argument("--batch_size", type=int, default=2, help="Rozmiar batcha")
    parser.add_argument("--epochs", type=int, default=10, help="Liczba epok")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_name = input(f"Podaj nazwę modelu (Enter dla domyślnej: mask_rcnn_v2_{timestamp}): ").strip() or f"mask_rcnn_v2_{timestamp}"

    os.makedirs(f"../logs/train/{model_name}", exist_ok=True)
    os.makedirs(f"../logs/val/{model_name}", exist_ok=True)
    os.makedirs("../logs/saved_models", exist_ok=True)

    print("\nWczytywanie danych...")
    train_loader, val_loader = get_data_loaders(args.dataset_dir, batch_size=args.batch_size, num_workers=args.num_workers)

    model = get_model(num_classes=2, device=device)  # Tło + rura
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    train_losses = []
    val_losses = []
    pred_counts = []
    gt_counts = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)
        val_loss, pred_count, gt_count = validate_model(model, val_loader, device, epoch, model_name)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        pred_counts.append(pred_count)
        gt_counts.append(gt_count)

        print(f"Epoka {epoch}/{args.epochs} - Strata treningowa: {train_loss:.4f}, Strata walidacyjna: {val_loss:.4f}, Pred: {pred_count}, GT: {gt_count}")

    model_filename = f"../models/{model_name}.pth"
    torch.save(model.state_dict(), model_filename)
    print(f"Model zapisano jako: {model_filename}")

    # Wykresy
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Strata treningowa")
    plt.plot(val_losses, label="Strata walidacyjna")
    plt.xlabel("Epoka")
    plt.ylabel("Strata")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"train/{model_name}/loss_plot.png")

    plt.figure(figsize=(10, 5))
    plt.plot(pred_counts, label="Wykryte obiekty")
    plt.plot(gt_counts, label="Obiekty GT")
    plt.xlabel("Epoka")
    plt.ylabel("Liczba obiektów")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"train/{model_name}/detections_plot.png")

    print("Trening zakończony!")