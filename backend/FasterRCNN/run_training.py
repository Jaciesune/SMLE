import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import argparse
import matplotlib.pyplot as plt
from datetime import datetime

from data_loader import get_data_loaders
from model import get_model
from train import train_one_epoch
from val_utils import validate_model

from config import CONFIDENCE_THRESHOLD

def main():
    parser = argparse.ArgumentParser(description="Trening Faster R-CNN z ResNet50")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, help="Wielkość batcha")
    parser.add_argument("--epochs", type=int, help="Liczba epok")
    parser.add_argument("--model_name", type=str, help="Nazwa modelu")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate")
    parser.add_argument("--train_dir", type=str, help="Ścieżka do katalogu danych treningowych")
    parser.add_argument("--coco_train_path", type=str, help="Ścieżka do pliku COCO z adnotacjami treningowymi")
    parser.add_argument("--coco_gt_path", type=str, help="Ścieżka do pliku COCO z adnotacjami walidacyjnymi")

    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    model_name = args.model_name or f"faster_rcnn_{timestamp}"
    batch_size = args.batch_size if args.batch_size is not None else 2
    epochs = args.epochs if args.epochs is not None else 40

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\nUżywane urządzenie: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}")

    os.makedirs(f"/app/FasterRCNN/train/{model_name}", exist_ok=True)
    os.makedirs(f"/app/FasterRCNN/val/{model_name}", exist_ok=True)
    os.makedirs(f"/app/FasterRCNN/saved_models/{model_name}", exist_ok=True)
    os.makedirs(f"/app/FasterRCNN/test/{model_name}", exist_ok=True)

    print("\nDebug - Ścieżki danych:")
    train_path = os.path.join(args.train_dir, "images")
    val_path = os.path.join(os.path.dirname(os.path.dirname(args.coco_gt_path)), "images")
    test_path = os.path.join(os.path.dirname(args.train_dir), "test", "images")
    print("+ train_path:", os.path.join(args.train_dir, "images"))
    print("+ train_annotations:", args.coco_train_path)
    print("+ val_path:", os.path.abspath(os.path.join(os.path.dirname(args.coco_gt_path), "images")))
    print("+ val_annotations:", args.coco_gt_path)
    print("+ test_path:", os.path.abspath(os.path.join(args.train_dir, "../test/images")))

    print("\nWczytywanie danych...")
    train_loader, val_loader, _ = get_data_loaders(
        batch_size=batch_size,
        num_workers=args.num_workers,
        train_path=os.path.join(args.train_dir, "images"),
        train_annotations=args.coco_train_path,
        val_path = os.path.abspath(os.path.join(os.path.dirname(args.coco_gt_path), "..", "images")),
        val_annotations=args.coco_gt_path,
        test_path = os.path.abspath(os.path.join(args.train_dir, "../test/images"))
    )

    model = get_model(num_classes=2, device=device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    # Scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3
    )

    best_val_loss = float("inf")
    best_epoch = 0
    train_losses, val_losses, pred_counts, gt_counts = [], [], [], []
    map_5095_list, map_50_list, precision_list, recall_list = [], [], [], []

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)
        val_loss, pred_count, gt_count, map_5095, map_50, precision, recall = validate_model(
            model, val_loader, device, epoch, model_name
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        pred_counts.append(pred_count)
        gt_counts.append(gt_count)
        map_5095_list.append(map_5095)
        map_50_list.append(map_50)
        precision_list.append(precision)
        recall_list.append(recall)

        print(f"Epoka {epoch}/{epochs} - Strata treningowa: {train_loss:.4f}, Strata walidacyjna: {val_loss:.4f}")
        print(f"                - Detekcje: {pred_count} | GT: {gt_count}")
        print(f"                - mAP@0.5: {map_50:.4f} | mAP@0.5:0.95: {map_5095:.4f} | Prec: {precision:.4f} | Rec: {recall:.4f}")

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Learning rate: {current_lr:.6f}")

        if epoch % 5 == 0:
            torch.save(model.state_dict(), f"/app/FasterRCNN/saved_models/{model_name}/model_epoch_{epoch}.pth")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), f"/app/FasterRCNN/saved_models/{model_name}/model_best_epoch_{epoch}.pth")

    torch.save(model.state_dict(), f"/app/FasterRCNN/saved_models/{model_name}_final_checkpoint.pth")
    print(f"\nModel końcowy zapisano jako: saved_models/{model_name}/model_final_checkpoint.pth")
    print(f"Najlepszy model pochodzi z epoki {best_epoch} (val_loss = {best_val_loss:.4f})")

    # Wykres strat
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Strata treningowa")
    plt.plot(val_losses, label="Strata walidacyjna")
    plt.xlabel("Epoka")
    plt.ylabel("Strata")
    plt.legend()
    plt.grid(True)
    plt.title("Strata w czasie treningu")
    plt.savefig(f"/app/FasterRCNN/test/{model_name}/loss_plot.png")
    plt.close()

    # Wykres liczby detekcji
    plt.figure(figsize=(10, 5))
    plt.plot(pred_counts, label="Wykryte obiekty")
    plt.plot(gt_counts, label="Obiekty GT")
    plt.xlabel("Epoka")
    plt.ylabel("Liczba obiektów")
    plt.legend()
    plt.grid(True)
    plt.title("Porównanie predykcji i GT")
    plt.savefig(f"/app/FasterRCNN/test/{model_name}/detections_plot.png")
    plt.close()

    # Wykres mAP
    plt.figure(figsize=(10, 5))
    plt.plot(map_50_list, label="mAP@0.5")
    plt.plot(map_5095_list, label="mAP@0.5:0.95")
    plt.xlabel("Epoka")
    plt.ylabel("mAP")
    plt.title("Mean Average Precision")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"/app/FasterRCNN/test/{model_name}/map_plot.png")
    plt.close()

    print("Wykresy zapisane w folderze test.")

if __name__ == "__main__":
    main()
