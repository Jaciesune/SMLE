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
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as coco_mask

# KONFIGURACJA
CONFIDENCE_THRESHOLD = 0.7  # Próg pewności dla detekcji
NMS_THRESHOLD = 40000  # Ilość propozycji przed NMS
DETECTION_PER_IMAGE = 2000  # Maksymalna liczba detekcji na obraz

# Pobranie modelu Mask R-CNN (wersja v2)
def get_model(num_classes, device):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(
        in_channels=256, dim_reduced=256, num_classes=num_classes
    )

    if isinstance(model.rpn.pre_nms_top_n, dict):
        model.rpn.pre_nms_top_n["training"] = NMS_THRESHOLD
        model.rpn.pre_nms_top_n["testing"] = NMS_THRESHOLD
        model.rpn.post_nms_top_n["training"] = NMS_THRESHOLD
        model.rpn.post_nms_top_n["testing"] = NMS_THRESHOLD
    model.roi_heads.score_thresh = CONFIDENCE_THRESHOLD
    model.roi_heads.detections_per_img = DETECTION_PER_IMAGE
    model.to(device)
    print(f"Model działa na: {device}")
    return model

# Funkcja walidacji z mAP i maskami
def validate_model(model, dataloader, device, epoch, model_name, coco_gt_path):
    model.eval()  # Tryb ewaluacji
    total_val_loss = 0
    total_pred_objects = 0
    total_gt_objects = 0
    save_path = f"../logs/val/{model_name}/epoch_{epoch:02d}"
    os.makedirs(save_path, exist_ok=True)

    # Przygotowanie wyników do oceny COCO
    coco_gt = COCO(coco_gt_path)
    coco_dt = []

    print(f"Walidacja epoki {epoch}...")

    for idx, (images, targets) in enumerate(dataloader):
        images = [img.to(device) for img in images]
        new_targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Obliczanie straty w trybie treningowym
        model.train()
        loss_dict = model(images, new_targets)
        val_loss = sum(loss for loss in loss_dict.values())
        total_val_loss += val_loss.item()

        # Predykcje w trybie ewaluacji
        model.eval()
        with torch.no_grad():
            outputs = model(images)

        for i, (image, output, target) in enumerate(zip(images, outputs, targets)):
            image_np = (image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

            pred_count = 0
            gt_count = target["boxes"].shape[0] if "boxes" in target and target["boxes"].numel() > 0 else 0
            total_gt_objects += gt_count

            # Sprawdzenie, czy obraz ma adnotacje
            if gt_count == 0:
                print(f"Batch {idx}, Image {i}: Brak adnotacji (GT boxes: 0), pomijam.")
                continue  # Pomijamy obraz bez adnotacji

            # Zakładamy, że image_id jest w targets
            if "image_id" not in target:
                print(f"Batch {idx}, Image {i}: Brak image_id w target, pomijam.")
                continue
            image_id = int(target["image_id"].cpu().numpy()[0])

            # Debug: Sprawdzanie zawartości output
            print(f"Batch {idx}, Image {i}: {len(output['boxes'])} predykcji, GT boxes: {gt_count}, image_id: {image_id}")

            for box, score, label, mask in zip(output["boxes"], output["scores"], output["labels"], output["masks"]):
                if score >= CONFIDENCE_THRESHOLD:
                    x_min, y_min, x_max, y_max = map(int, box.cpu().numpy())
                    width = x_max - x_min
                    height = y_max - y_min
                    pred_count += 1

                    # Przetwarzanie maski na format RLE
                    mask_np = (mask.cpu().numpy()[0] > 0.5).astype(np.uint8)
                    rle = coco_mask.encode(np.asfortranarray(mask_np))
                    rle["counts"] = rle["counts"].decode("utf-8")

                    # Dodanie predykcji do formatu COCO
                    coco_dt.append({
                        "image_id": image_id,
                        "category_id": int(label.cpu().numpy()),
                        "bbox": [x_min, y_min, width, height],
                        "score": float(score.cpu().numpy()),
                        "segmentation": rle
                    })

                    # Rysowanie predykcji
                    cv2.rectangle(image_np, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    mask = (mask_np * 255).astype(np.uint8)
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(image_np, contours, -1, (0, 0, 255), 2)
                else:
                    print(f"Wynik {score:.4f} poniżej progu {CONFIDENCE_THRESHOLD}")

            total_pred_objects += pred_count
            filename = f"{save_path}/img_{idx}_{i}.png"
            cv2.imwrite(filename, image_np)

    # Debug: Sprawdzenie, czy coco_dt jest wypełnione
    print(f"Łącznie dodano predykcji do coco_dt: {len(coco_dt)}")

    # Obsługa pustego coco_dt
    if len(coco_dt) == 0:
        print("Brak predykcji powyżej progu pewności lub brak obrazów z adnotacjami. Pomijam ocenę COCO.")
        mAP_bbox = 0.0
        mAP_seg = 0.0
    else:
        coco_dt = coco_gt.loadRes(coco_dt)
        
        # mAP dla bounding box
        coco_eval_bbox = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval_bbox.evaluate()
        coco_eval_bbox.accumulate()
        coco_eval_bbox.summarize()
        mAP_bbox = coco_eval_bbox.stats[0]  # mAP@IoU=0.5:0.95 dla bbox

        # mAP dla segmentacji
        coco_eval_seg = COCOeval(coco_gt, coco_dt, "segm")
        coco_eval_seg.evaluate()
        coco_eval_seg.accumulate()
        coco_eval_seg.summarize()
        mAP_seg = coco_eval_seg.stats[0]  # mAP@IoU=0.5:0.95 dla masek

    avg_val_loss = total_val_loss / len(dataloader) if len(dataloader) > 0 else 0
    return avg_val_loss, total_pred_objects, total_gt_objects, mAP_bbox, mAP_seg

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
    parser.add_argument("--lr", type=float, default=0.001, help="Początkowa wartość learning rate")
    parser.add_argument("--patience", type=int, default=3, help="Liczba epok bez poprawy dla Early Stopping")
    parser.add_argument("--coco_gt_path", type=str, default="../data/val/annotations/instances_val.json", help="Ścieżka do pliku COCO z adnotacjami walidacyjnymi")
    parser.add_argument("--num_augmentations", type=int, default=1, help="Liczba augmentacji na obraz")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_name = input(f"Podaj nazwę modelu (Enter dla domyślnej: mask_rcnn_v2_{timestamp}): ").strip() or f"mask_rcnn_v2_{timestamp}"

    os.makedirs(f"../logs/train/{model_name}", exist_ok=True)
    os.makedirs(f"../logs/val/{model_name}", exist_ok=True)
    os.makedirs("../logs/saved_models", exist_ok=True)

    print("\nWczytywanie danych...")
    train_loader, val_loader = get_data_loaders(
        args.dataset_dir, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers, 
        num_augmentations=args.num_augmentations
    )
    model = get_model(num_classes=2, device=device)  # Tło + rura
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    train_losses = []
    val_losses = []
    pred_counts = []
    gt_counts = []
    mAPs_bbox = []
    mAPs_seg = []

    # Early Stopping
    best_val_loss = float("inf")
    patience_counter = 0
    patience = args.patience

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)
        val_loss, pred_count, gt_count, mAP_bbox, mAP_seg = validate_model(model, val_loader, device, epoch, model_name, args.coco_gt_path)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        pred_counts.append(pred_count)
        gt_counts.append(gt_count)
        mAPs_bbox.append(mAP_bbox)
        mAPs_seg.append(mAP_seg)

        print(f"Epoka {epoch}/{args.epochs} - Strata treningowa: {train_loss:.4f}, Strata walidacyjna: {val_loss:.4f}, Pred: {pred_count}, GT: {gt_count}, mAP_bbox: {mAP_bbox:.4f}, mAP_seg: {mAP_seg:.4f}")

        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f"../models/{model_name}_best.pth")
            print(f"Zapisano najlepszy model: ../models/{model_name}_best.pth")
        else:
            patience_counter += 1
            print(f"Brak poprawy przez {patience_counter}/{patience} epok")
            if patience_counter >= patience:
                print("Early Stopping: Zakończono trening przedwcześnie.")
                break

    # Zapis końcowego modelu
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
    plt.savefig(f"../logs/train/{model_name}/loss_plot.png")

    plt.figure(figsize=(10, 5))
    plt.plot(pred_counts, label="Wykryte obiekty")
    plt.plot(gt_counts, label="Obiekty GT")
    plt.xlabel("Epoka")
    plt.ylabel("Liczba obiektów")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"../logs/train/{model_name}/detections_plot.png")

    plt.figure(figsize=(10, 5))
    plt.plot(mAPs_bbox, label="mAP (bbox)")
    plt.plot(mAPs_seg, label="mAP (segmentacja)")
    plt.xlabel("Epoka")
    plt.ylabel("mAP")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"../logs/train/{model_name}/mAP_plot.png")

    print("Trening zakończony!")