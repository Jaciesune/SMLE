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

# Pobranie modelu Faster R-CNN
def get_model(num_classes, device):
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    model.to(device)
    print(f"Model działa na: {device}")
    return model

# Wizualizacja wyników po każdej epoce
def visualize_predictions(model, dataloader, device, epoch, model_name, phase="train"):
    model.eval()
    save_path = f"{phase}/{model_name}/epoch_{epoch:02d}"  # Foldery numerowane 01, 02, ...
    os.makedirs(save_path, exist_ok=True)

    with torch.no_grad():
        for idx, (images, targets) in enumerate(dataloader):
            if idx >= 5:  # Ograniczamy do kilku obrazów
                break

            images = [img.to(device) for img in images]
            outputs = model(images)

            for i, (image, output) in enumerate(zip(images, outputs)):
                image_np = (image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

                # Rysowanie predykcji modelu
                for box in output["boxes"]:
                    x_min, y_min, x_max, y_max = map(int, box.cpu().numpy())
                    cv2.rectangle(image_np, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                # Jeśli to walidacja, rysujemy też prawdziwe adnotacje
                if phase == "val":
                    for box in targets[i]["boxes"]:
                        x_min, y_min, x_max, y_max = map(int, box.cpu().numpy())
                        cv2.rectangle(image_np, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

                filename = f"{save_path}/img_{idx}_{i}.png"
                cv2.imwrite(filename, image_np)

# Rysowanie wykresu strat
def plot_losses(train_losses, model_name):
    save_path = f"train/{model_name}/loss_plot.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, marker="o", linestyle="-", color="b", label="Strata treningowa")
    plt.xlabel("Epoka")
    plt.ylabel("Strata")
    plt.title("Strata treningowa w czasie")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f"Wykres strat zapisano: {save_path}")

# Funkcja treningu
def train_one_epoch(model, dataloader, optimizer, device, epoch):
    model.train()
    total_loss = 0
    processed_images = 0
    epoch_start_time = time.time()

    print(f"\nEpoka {epoch}... (Batchy: {len(dataloader)})")

    for batch_idx, (images, targets) in enumerate(dataloader):
        batch_start_time = time.time()
        images = [image.to(device) for image in images]

        new_targets = []
        for t in targets:
            if isinstance(t, list):  # Obsługa sytuacji, gdy target to lista list
                target_dict = {
                    "boxes": torch.tensor([obj["bbox"] for obj in t], dtype=torch.float32, device=device),
                    "labels": torch.tensor([obj["category_id"] for obj in t], dtype=torch.int64, device=device),
                }
                new_targets.append(target_dict)
            else:  # Jeśli target jest poprawnym słownikiem, konwertujemy wartości
                new_targets.append({k: v.to(device) for k, v in t.items()})

        if not new_targets:
            print(f"Pominięto pusty batch {batch_idx+1}")
            continue  # Pominięcie batcha, jeśli nie zawiera poprawnych targetów

        loss_dict = model(images, new_targets)
        loss = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        processed_images += len(images)

        batch_time = time.time() - batch_start_time
        print(f"Batch {batch_idx+1}/{len(dataloader)} - Czas: {batch_time:.4f}s, Strata: {loss.item():.4f}")

    epoch_time = time.time() - epoch_start_time
    print(f"\nEpoka {epoch} zakończona! Czas epoki: {epoch_time:.2f}s | Przetworzone obrazy: {processed_images}")
    
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

    # Zapytanie o nazwę modelu na początku programu
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_name = input(f"Podaj nazwę modelu (Enter dla domyślnej: faster_rcnn_{timestamp}): ").strip() or f"faster_rcnn_{timestamp}"

    # Tworzymy folder dla wyników tej sesji uczenia
    os.makedirs(f"train/{model_name}", exist_ok=True)
    os.makedirs(f"val/{model_name}", exist_ok=True)

    print("\nWczytywanie danych...")
    train_loader, val_loader, test_loader = get_data_loaders(batch_size=args.batch_size, num_workers=args.num_workers)
    print(f"DataLoader załadowany: Trening: {len(train_loader.dataset)} | Walidacja: {len(val_loader.dataset)} | Test: {len(test_loader.dataset)}")

    model = get_model(num_classes=2, device=device)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    train_losses = []

    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, device, epoch)
        train_losses.append(loss)

        # Każda epoka zapisywana w osobnym folderze
        visualize_predictions(model, train_loader, device, epoch, model_name, "train")
        visualize_predictions(model, val_loader, device, epoch, model_name, "val")

    print("\nTrening zakończony!")

    # Zapis modelu
    model_filename = f"saved_models/{model_name}.pth"
    torch.save(model.state_dict(), model_filename)
    print(f"💾 Model zapisano jako: {model_filename}")

    # Zapis wykresu strat
    plot_losses(train_losses, model_name)