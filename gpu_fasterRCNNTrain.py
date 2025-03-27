import torch
import torchvision.models.detection
import torch.optim as optim
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import argparse
from datetime import datetime
from gpu_dataLoader import get_data_loaders

# KONFIGURACJA
CONFIDENCE_THRESHOLD = 0.27  # Próg pewności
NMS_THRESHOLD = 40000  # Ilość propozycji
SAVE_PERFECT_MODEL_RATIO_RANGE = (0.9, 1.1)  # Zakres idealnego stosunku pred/gt

# Pobranie modelu Faster R-CNN
def get_model(num_classes, device):
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

    if isinstance(model.rpn.pre_nms_top_n, dict):
        model.rpn.pre_nms_top_n["training"] = NMS_THRESHOLD
        model.rpn.pre_nms_top_n["testing"] = NMS_THRESHOLD
        model.rpn.post_nms_top_n["training"] = NMS_THRESHOLD
        model.rpn.post_nms_top_n["testing"] = NMS_THRESHOLD
    else:
        model.rpn.pre_nms_top_n = lambda: NMS_THRESHOLD
        model.rpn.post_nms_top_n = lambda: NMS_THRESHOLD

    model.roi_heads.detections_per_img = 5000
    model.roi_heads.score_thresh = CONFIDENCE_THRESHOLD
    model.to(device)
    print(f"Model działa na: {device} ({torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'})")
    return model

# Funkcja walidacji
def validate_model(model, dataloader, device, epoch, model_name):
    model.train()
    total_val_loss = 0
    total_pred_objects = 0
    total_gt_objects = 0
    save_path = f"val/{model_name}/epoch_{epoch:02d}"
    os.makedirs(save_path, exist_ok=True)

    for idx, (images, targets) in enumerate(dataloader):
        images = [img.to(device) for img in images]
        new_targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, new_targets)
        val_loss = sum(loss for loss in loss_dict.values())
        total_val_loss += val_loss.item()

        model.eval()
        with torch.no_grad():
            outputs = model(images)

        for i, (image, output, target) in enumerate(zip(images, outputs, targets)):
            image_np = (image.detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

            pred_count = 0
            gt_count = target["boxes"].shape[0]
            total_gt_objects += gt_count

            for box, score in zip(output["boxes"], output["scores"]):
                if score >= CONFIDENCE_THRESHOLD:
                    x_min, y_min, x_max, y_max = map(int, box.detach().cpu().numpy())
                    cv2.rectangle(image_np, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    pred_count += 1

            total_pred_objects += pred_count
            filename = f"{save_path}/img_{idx}_{i}.png"
            cv2.imwrite(filename, image_np)

        model.train()

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
    parser = argparse.ArgumentParser(description="Trening Faster R-CNN na GPU")
    parser.add_argument("--num_workers", type=int, default=4, help="Liczba wątków dla DataLoadera")
    parser.add_argument("--batch_size", type=int, default=2, help="Rozmiar batcha")
    parser.add_argument("--epochs", type=int, default=10, help="Liczba epok treningowych")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        print(f"Używana karta graficzna: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device("cpu")
        print("CUDA niedostępne - używana zostanie jednostka CPU")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_name = input(f"Podaj nazwę modelu (Enter dla domyślnej: faster_rcnn_{timestamp}): ").strip() or f"faster_rcnn_{timestamp}"

    os.makedirs(f"train/{model_name}", exist_ok=True)
    os.makedirs(f"val/{model_name}", exist_ok=True)
    os.makedirs(f"saved_models/{model_name}", exist_ok=True)

    print("\nWczytywanie danych...")
    train_loader, val_loader, _ = get_data_loaders(batch_size=args.batch_size, num_workers=args.num_workers)

    model = get_model(num_classes=2, device=device)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    train_losses = []
    val_losses = []
    pred_counts = []
    gt_counts = []
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)
        val_loss, pred_count, gt_count = validate_model(model, val_loader, device, epoch, model_name)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        pred_counts.append(pred_count)
        gt_counts.append(gt_count)

        print(f"Epoka {epoch}/{args.epochs} - Strata treningowa: {train_loss:.4f}, Strata walidacyjna: {val_loss:.4f}, Pred: {pred_count}, GT: {gt_count}")

        if epoch % 5 == 0:
            checkpoint_path = f"saved_models/{model_name}/model_epoch_{epoch}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Zapisano model po epoce {epoch}: {checkpoint_path}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = f"saved_models/{model_name}/model_best.pth"
            torch.save(model.state_dict(), best_path)
            print(f"Zapisano najlepszy model (val_loss={val_loss:.4f}): {best_path}")

        if gt_count > 0:
            pred_gt_ratio = pred_count / gt_count
            if SAVE_PERFECT_MODEL_RATIO_RANGE[0] <= pred_gt_ratio <= SAVE_PERFECT_MODEL_RATIO_RANGE[1]:
                perfect_path = f"saved_models/{model_name}/model_almost_perfect_epoch_{epoch}.pth"
                torch.save(model.state_dict(), perfect_path)
                print(f"Zapisano model bliski perfekcji (Pred/GT = {pred_gt_ratio:.2f}): {perfect_path}")

    final_model_path = f"saved_models/{model_name}/model_final.pth"
    torch.save(model.state_dict(), final_model_path)
    print(f"Model końcowy zapisano jako: {final_model_path}")

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
