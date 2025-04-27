import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import os
import argparse
import matplotlib.pyplot as plt
import sys
import io
from datetime import datetime
import glob

from data_loader import get_data_loaders
from model import get_model
from train import train_one_epoch
from val_utils import validate_model
from config import CONFIDENCE_THRESHOLD, NUM_CLASSES

# Wymuszenie UTF-8 z fallbackiem na błędy
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

def main():
    parser = argparse.ArgumentParser(description="Trening Faster R-CNN z ResNet50", allow_abbrev=False)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--train_dir", type=str)
    parser.add_argument("--coco_train_path", type=str)
    parser.add_argument("--coco_gt_path", type=str)

    args, _ = parser.parse_known_args()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_name = args.model_name or f"faster_rcnn_{timestamp}"
    batch_size = args.batch_size or 2
    epochs = args.epochs or 40

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\nUżywane urządzenie: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}")

    os.makedirs(f"/app/backend/FasterRCNN/train/{model_name}", exist_ok=True)
    os.makedirs(f"/app/backend/FasterRCNN/val/{model_name}", exist_ok=True)
    os.makedirs(f"/app/backend/FasterRCNN/saved_models", exist_ok=True)
    os.makedirs(f"/app/backend/FasterRCNN/test/{model_name}", exist_ok=True)

    print("\nDebug - Ścieżki danych:")
    train_path = os.path.join(args.train_dir, "images")
    val_path = os.path.join(os.path.dirname(os.path.dirname(args.coco_gt_path)), "images")
    test_path = os.path.join(os.path.dirname(args.train_dir), "test", "images")
    print("+ train_path:", train_path)
    print("+ train_annotations:", args.coco_train_path)
    print("+ val_path:", val_path)
    print("+ val_annotations:", args.coco_gt_path)
    print("+ test_path:", test_path)

    print("\nWczytywanie danych...")
    train_loader, val_loader = get_data_loaders(
        batch_size=batch_size,
        num_workers=args.num_workers,
        train_path=train_path,
        train_annotations=args.coco_train_path,
        val_path=val_path,
        val_annotations=args.coco_gt_path
    )

    model = get_model(num_classes=NUM_CLASSES, device=device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)  # Zmiana tu

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

        scheduler.step()
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_model_path = f"/app/backend/FasterRCNN/saved_models/{model_name}_checkpoint.pth"

            # Zapis najlepszego (obecnie) modelu
            torch.save(model.state_dict(), best_model_path)
            print(f"[✓] Zapisano nowy najlepszy model: {best_model_path}")

            # Usunięcie pozostałych
            for path in glob.glob(f"/app/backend/FasterRCNN/saved_models/{model_name}_*.pth"):
                if path != best_model_path:
                    try:
                        os.remove(path)
                        print(f"[✓] Usunięto stary model: {path}")
                    except Exception as e:
                        print(f"[✗] Błąd przy usuwaniu {path}: {e}")

    print(f"\nModel końcowy zapisano jako: {best_model_path}")
    print(f"Najlepszy model pochodzi z epoki {best_epoch} (val_loss = {best_val_loss:.4f})")

    def save_plot(data1, data2, labels, title, filename, ylabel):
        plt.figure(figsize=(10, 5))
        plt.plot(data1, label=labels[0])
        plt.plot(data2, label=labels[1])
        plt.xlabel("Epoka")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.savefig(f"/app/backend/FasterRCNN/test/{model_name}/{filename}")
        plt.close()

    save_plot(train_losses, val_losses, ["Strata treningowa", "Strata walidacyjna"], "Strata w czasie treningu", "loss_plot.png", "Strata")
    save_plot(pred_counts, gt_counts, ["Wykryte obiekty", "Obiekty GT"], "Porównanie predykcji i GT", "detections_plot.png", "Liczba obiektów")
    save_plot(map_50_list, map_5095_list, ["mAP@0.5", "mAP@0.5:0.95"], "Mean Average Precision", "map_plot.png", "mAP")

    print("Wykresy zapisane w folderze test.")

if __name__ == "__main__":
    main()
