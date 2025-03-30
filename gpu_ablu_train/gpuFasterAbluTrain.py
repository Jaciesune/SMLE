import torch
import torchvision.models.detection as detection
import torch.optim as optim
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import argparse
import torchvision
from datetime import datetime
from gpuDataLoaderAblu import get_data_loaders
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models import ResNet50_Weights

CONFIDENCE_THRESHOLD = 0.32
NMS_THRESHOLD = 15000
SAVE_PERFECT_MODEL_RATIO_RANGE = (0.93, 1.07)

def get_model(num_classes, device):
    anchor_generator = AnchorGenerator(
        sizes=((16,), (32,), (64,), (128,), (256,)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 5
    )

    model = fasterrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=ResNet50_Weights.DEFAULT,
        rpn_anchor_generator=anchor_generator
    )

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

    model.roi_heads.score_thresh = 0.25
    model.roi_heads.nms_thresh = 0.14
    model.roi_heads.detections_per_img = 4000
    model.to(device)

    print(f"Model działa na: {device} ({torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'})")
    return model

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

            h_img, w_img = image_np.shape[:2]
            for box, score in zip(output["boxes"], output["scores"]):
                if score >= CONFIDENCE_THRESHOLD:
                    x_min, y_min, x_max, y_max = box.detach().cpu().numpy()
                    width = x_max - x_min
                    height = y_max - y_min
                    area = width * height
                    image_area = w_img * h_img
                    aspect_ratio = max(width / height, height / width)

                    if area > 0.3 * image_area or aspect_ratio > 3.5:
                        continue

                    x_min, y_min, x_max, y_max = map(int, [x_min, y_min, x_max, y_max])
                    cv2.rectangle(image_np, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    pred_count += 1

            total_pred_objects += pred_count
            filename = f"{save_path}/img_{idx}_{i}.png"
            cv2.imwrite(filename, image_np)

        model.train()

    avg_val_loss = total_val_loss / len(dataloader) if len(dataloader) > 0 else 0
    return avg_val_loss, total_pred_objects, total_gt_objects

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
    parser = argparse.ArgumentParser(description="Trening Faster R-CNN ResNet50 z mocną augmentacją")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=40)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"Używana karta graficzna: {torch.cuda.get_device_name(device)}")
    else:
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
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)
        val_loss, pred_count, gt_count = validate_model(model, val_loader, device, epoch, model_name)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        pred_counts.append(pred_count)
        gt_counts.append(gt_count)

        print(f"Epoka {epoch}/{args.epochs} - Strata treningowa: {train_loss:.4f}, Strata walidacyjna: {val_loss:.4f}, Pred: {pred_count}, GT: {gt_count}")

        if epoch % 5 == 0:
            torch.save(model.state_dict(), f"saved_models/{model_name}/model_epoch_{epoch}.pth")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), f"saved_models/{model_name}/model_best_epoch_{epoch}.pth")

    torch.save(model.state_dict(), f"saved_models/{model_name}/model_final.pth")
    print(f"Model końcowy zapisano jako: saved_models/{model_name}/model_final.pth")
    print(f"Najlepszy model pochodzi z epoki {best_epoch} (val_loss = {best_val_loss:.4f})")

    # WYKRESY
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

    print("Wykresy zapisane.")
