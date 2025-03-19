import torch
import torchvision.models.detection
import torch.optim as optim
from dataLoader import get_data_loaders
import argparse
import time
import os
from datetime import datetime

def get_model(num_classes, device):
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights="DEFAULT")

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

    model.to(device)
    print(f"Model działa na: {device}")
    return model

def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    epoch_start_time = time.time()

    for batch_idx, (images, targets) in enumerate(dataloader):
        batch_start_time = time.time()

        images = [image.to(device) for image in images]
        new_targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, new_targets)
        loss = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        batch_time = time.time() - batch_start_time
        print(f"Batch {batch_idx+1}/{len(dataloader)} - Czas: {batch_time:.4f}s, Strata: {loss.item():.4f}")

    epoch_time = time.time() - epoch_start_time
    print(f"Czas epoki: {epoch_time:.2f}s")

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

    train_loader, test_loader = get_data_loaders(batch_size=args.batch_size, num_workers=args.num_workers)

    model = get_model(num_classes=2, device=device)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(args.epochs):
        loss = train_one_epoch(model, train_loader, optimizer, device)
        print(f"Epoka {epoch+1}/{args.epochs}, Strata: {loss:.4f}")

    print("Trening zakończony!")

    # 📌 Po zakończeniu treningu zapytaj, czy zapisać model
    save_model = input("Czy zapisać model? (Y/N): ").strip().lower()
    
    if save_model == "y":
        os.makedirs("saved_models", exist_ok=True)  # Tworzymy katalog, jeśli nie istnieje
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        # Pobieranie niestandardowej nazwy modelu od użytkownika
        custom_name = input(f"Podaj nazwę modelu (Enter dla domyślnej: faster_rcnn_{timestamp}): ").strip()
        
        if custom_name:
            model_filename = f"saved_models/{custom_name}_{timestamp}.pth"
        else:
            model_filename = f"saved_models/faster_rcnn_{timestamp}.pth"

        torch.save(model.state_dict(), model_filename)
        print(f"Model zapisano jako: {model_filename}")
    else:
        print("Model nie został zapisany.")
