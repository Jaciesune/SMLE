import torch
import torchvision.models.detection
import torch.optim as optim
from dataLoader import get_data_loaders
import argparse
import time

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

        new_targets = []
        for target in targets:
            if isinstance(target, list):
                target_dict = {
                    "boxes": torch.tensor([obj["bbox"] for obj in target], dtype=torch.float32, device=device),
                    "labels": torch.tensor([obj["category_id"] for obj in target], dtype=torch.int64, device=device),
                }
                new_targets.append(target_dict)
            else:
                new_targets.append({k: v.to(device) for k, v in target.items()})

        if not new_targets:
            continue

        loss_dict = model(images, new_targets)
        loss = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        batch_end_time = time.time()
        batch_time = batch_end_time - batch_start_time
        print(f"Batch {batch_idx+1}/{len(dataloader)} - Czas: {batch_time:.4f}s, Strata: {loss.item():.4f}")

    epoch_end_time = time.time()
    epoch_time = epoch_end_time - epoch_start_time
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

    # 🔧 Upewniamy się, że przekazujemy tylko dopuszczalne argumenty
    train_loader, test_loader = get_data_loaders(batch_size=args.batch_size, num_workers=args.num_workers)

    model = get_model(num_classes=2, device=device)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(args.epochs):
        loss = train_one_epoch(model, train_loader, optimizer, device)
        print(f"Epoka {epoch+1}/{args.epochs}, Strata: {loss:.4f}")

    print("Trening zakończony!")
