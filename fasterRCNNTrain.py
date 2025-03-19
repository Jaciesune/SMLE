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
    print(f"✅ Model działa na: {device}")
    return model

# Wizualizacja wyników co 5 epok
def visualize_predictions(model, dataloader, device, epoch, model_name, phase="train"):
    model.eval()
    save_path = f"{phase}/{model_name}/epoch_{epoch}"
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

# Funkcja treningu
def train_one_epoch(model, dataloader, optimizer, device, epoch):
    model.train()
    total_loss = 0
    processed_images = 0
    epoch_start_time = time.time()

    print(f"\n🔄 Rozpoczynam epokę {epoch}... (Batchy: {len(dataloader)})")

    for batch_idx, (images, targets) in enumerate(dataloader):
        batch_start_time = time.time()
        images = [image.to(device) for image in images]

        # 🔧 Poprawka: Unifikacja formatu `targets`
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
            print(f"⚠️ Pominięto pusty batch {batch_idx+1}")
            continue  # Pominięcie batcha, jeśli nie zawiera poprawnych targetów

        loss_dict = model(images, new_targets)
        loss = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        processed_images += len(images)

        batch_time = time.time() - batch_start_time
        print(f"✅ Batch {batch_idx+1}/{len(dataloader)} - Czas: {batch_time:.4f}s, Strata: {loss.item():.4f}")

    epoch_time = time.time() - epoch_start_time
    print(f"\n✅ Epoka {epoch} zakończona! Czas epoki: {epoch_time:.2f}s | Przetworzone obrazy: {processed_images}")
    
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

    print("\n📥 Wczytywanie danych...")
    train_loader, val_loader, test_loader = get_data_loaders(batch_size=args.batch_size, num_workers=args.num_workers)
    print(f"✅ DataLoader załadowany: Trening: {len(train_loader.dataset)} | Walidacja: {len(val_loader.dataset)} | Test: {len(test_loader.dataset)}")

    model = get_model(num_classes=2, device=device)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    train_losses = []

    for epoch in range(1, args.epochs + 1):
        print(f"\n🚀 Epoka {epoch}/{args.epochs} rozpoczyna się...")
        loss = train_one_epoch(model, train_loader, optimizer, device, epoch)
        train_losses.append(loss)
        print(f"✅ Epoka {epoch}/{args.epochs} zakończona, Strata: {loss:.4f}")

        # Wizualizacja co 5 epok
        if epoch % 5 == 0 or epoch == args.epochs:
            visualize_predictions(model, train_loader, device, epoch, "train_model", "train")

    print("\n🎉 Trening zakończony!")

    # Pytanie o zapis modelu
    save_model = input("\nCzy zapisać model? (Y/N): ").strip().lower()
    
    if save_model == "y":
        os.makedirs("saved_models", exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        custom_name = input(f"Podaj nazwę modelu (Enter dla domyślnej: faster_rcnn_{timestamp}): ").strip()
        model_name = custom_name or f"faster_rcnn_{timestamp}"

        model_filename = f"saved_models/{model_name}.pth"
        torch.save(model.state_dict(), model_filename)
        print(f"✅ Model zapisano jako: {model_filename}")
    else:
        print("❌ Model nie został zapisany.")