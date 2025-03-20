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
from dataLoader import get_data_loaders

# KONFIGURACJA
CONFIDENCE_THRESHOLD = 0.27  # Próg pewności
NMS_THRESHOLD = 40000  # Ilość propozycji

# Pobranie modelu Faster R-CNN
def get_model(num_classes, device):
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    
    # Zwiększenie liczby propozycji RPN do 5000+
    if isinstance(model.rpn.pre_nms_top_n, dict):
        model.rpn.pre_nms_top_n["training"] = NMS_THRESHOLD
        model.rpn.pre_nms_top_n["testing"] = NMS_THRESHOLD
        model.rpn.post_nms_top_n["training"] = NMS_THRESHOLD
        model.rpn.post_nms_top_n["testing"] = NMS_THRESHOLD
    else:
        model.rpn.pre_nms_top_n = lambda: NMS_THRESHOLD
        model.rpn.post_nms_top_n = lambda: NMS_THRESHOLD

    model.roi_heads.detections_per_img = 5000  

    model.roi_heads.score_thresh = CONFIDENCE_THRESHOLD  # Próg pewności
    model.to(device)
    print(f"Model działa na: {device}")
    return model

# Funkcja obliczania straty walidacyjnej i liczby wykryć
def validate_model(model, dataloader, device, epoch, model_name):
    model.eval()
    total_val_loss = 0
    total_pred_objects = 0
    total_gt_objects = 0
    save_path = f"val/{model_name}/epoch_{epoch:02d}"
    os.makedirs(save_path, exist_ok=True)

    with torch.no_grad():
        for idx, (images, targets) in enumerate(dataloader):
            images = [img.to(device) for img in images]
            new_targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Sprawdzenie, czy walidacja zawiera adnotacje (jeśli tak – model zwraca stratę)
            if model.training:
                loss_dict = model(images, new_targets)
                val_loss = sum(loss for loss in loss_dict.values())
            else:
                val_loss = 0  # W trybie ewaluacji model nie zwraca straty

            total_val_loss += val_loss

            # Przetwarzanie predykcji
            outputs = model(images)

            for i, (image, output, target) in enumerate(zip(images, outputs, targets)):
                image_np = (image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

                pred_count = 0
                gt_count = target["boxes"].shape[0]
                total_gt_objects += gt_count

                # Rysowanie predykcji modelu
                for box, score in zip(output["boxes"], output["scores"]):
                    if score >= CONFIDENCE_THRESHOLD:
                        x_min, y_min, x_max, y_max = map(int, box.cpu().numpy())
                        cv2.rectangle(image_np, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                        pred_count += 1

                total_pred_objects += pred_count

                filename = f"{save_path}/img_{idx}_{i}.png"
                cv2.imwrite(filename, image_np)

    avg_val_loss = total_val_loss / len(dataloader) if len(dataloader) > 0 else 0
    return avg_val_loss, total_pred_objects, total_gt_objects

# Wizualizacja wyników po każdej epoce
def visualize_predictions(model, dataloader, device, epoch, model_name, phase="train"):
    model.eval()
    save_path = f"{phase}/{model_name}/epoch_{epoch:02d}"
    os.makedirs(save_path, exist_ok=True)

    with torch.no_grad():
        for idx, (images, targets) in enumerate(dataloader):
            if idx >= 5:  # Ograniczamy do kilku obrazów
                break

            images = [img.to(device) for img in images]
            outputs = model(images)

            for i, (image, output, target) in enumerate(zip(images, outputs, targets)):
                image_np = (image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

                # ZIELONE RAMKI – Wykryte przez model
                for box in output["boxes"]:
                    x_min, y_min, x_max, y_max = map(int, box.cpu().numpy())
                    cv2.rectangle(image_np, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)  # Zielony

                # NIEBIESKIE RAMKI – Prawdziwe anotacje (tylko dla walidacji!)
                if phase == "val":
                    for box in target["boxes"]:
                        x_min, y_min, x_max, y_max = map(int, box.cpu().numpy())
                        cv2.rectangle(image_np, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)  # Niebieski

                filename = f"{save_path}/img_{idx}_{i}.png"
                cv2.imwrite(filename, image_np)

# Funkcja treningu
def train_one_epoch(model, dataloader, optimizer, device, epoch):
    model.train()
    total_loss = 0
    processed_images = 0
    epoch_start_time = time.time()

    print(f"\nEpoka {epoch}... (Batchy: {len(dataloader)})")

    for batch_idx, (images, targets) in enumerate(dataloader):
        images = [image.to(device) for image in images]
        new_targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, new_targets)
        loss = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        processed_images += len(images)

        print(f"Batch {batch_idx+1}/{len(dataloader)} - Strata: {loss.item():.4f}")

    return total_loss / len(dataloader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trening Faster R-CNN na CPU")
    parser.add_argument("--num_workers", type=int, default=4, help="Liczba wątków dla DataLoadera")
    parser.add_argument("--batch_size", type=int, default=2, help="Rozmiar batcha")
    parser.add_argument("--epochs", type=int, default=10, help="Liczba epok treningowych")
    args = parser.parse_args()

    torch.set_num_threads(args.num_workers)
    torch.set_num_interop_threads(args.num_workers)
    device = torch.device("cpu")

    # Nazwa modelu na początku programu
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_name = input(f"Podaj nazwę modelu (Enter dla domyślnej: faster_rcnn_{timestamp}): ").strip() or f"faster_rcnn_{timestamp}"

    os.makedirs(f"train/{model_name}", exist_ok=True)
    os.makedirs(f"val/{model_name}", exist_ok=True)

    print("\nWczytywanie danych...")
    train_loader, val_loader, test_loader = get_data_loaders(batch_size=args.batch_size, num_workers=args.num_workers)

    model = get_model(num_classes=2, device=device)
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

         #  Wizualizacja wyników (dla train i val)
        visualize_predictions(model, train_loader, device, epoch, model_name, "train")
        visualize_predictions(model, val_loader, device, epoch, model_name, "val")

        print(f"Epoka {epoch}/{args.epochs} - Strata treningowa: {train_loss:.4f}, Strata walidacyjna: {val_loss:.4f}, Pred: {pred_count}, GT: {gt_count}")

    # Zapis modelu
    model_filename = f"saved_models/{model_name}.pth"
    torch.save(model.state_dict(), model_filename)
    print(f"Model zapisano jako: {model_filename}")

    # Wykres strat
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Strata treningowa")
    plt.plot(val_losses, label="Strata walidacyjna")
    plt.xlabel("Epoka")
    plt.ylabel("Strata")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"train/{model_name}/loss_plot.png")

    # Wykres liczby wykryć
    plt.figure(figsize=(10, 5))
    plt.plot(pred_counts, label="Wykryte obiekty")
    plt.plot(gt_counts, label="Obiekty GT")
    plt.xlabel("Epoka")
    plt.ylabel("Liczba obiektów")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"train/{model_name}/detections_plot.png")

    print("Trening zakończony!") 
