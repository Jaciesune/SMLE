# run_training.py
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import os
import argparse
import matplotlib.pyplot as plt
import sys
import io
from datetime import datetime
import glob
import logging

from data_loader import get_data_loaders
from model import get_model
from train import train_one_epoch
from val_utils import validate_model
from config import CONFIDENCE_THRESHOLD, NUM_CLASSES

# Konfiguracja logowania
logger = logging.getLogger(__name__)

# Wymuszenie UTF-8 z fallbackiem na błędy
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Stałe konfiguracyjne
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 2
MIN_LR = 1e-6

def main():
    parser = argparse.ArgumentParser(description="Trening Faster R-CNN z ResNet50", allow_abbrev=False)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=None)  # None oznacza automatyczne dopasowanie
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--train_dir", type=str)
    parser.add_argument("--coco_train_path", type=str)
    parser.add_argument("--coco_gt_path", type=str)

    args, _ = parser.parse_known_args()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_name = args.model_name or f"faster_rcnn_{timestamp}"
    epochs = args.epochs or 20  # Domyślnie 20 epok, jak w Mask R-CNN

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Używane urządzenie: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}")

    os.makedirs(f"/app/backend/FasterRCNN/train/{model_name}", exist_ok=True)
    os.makedirs(f"/app/backend/FasterRCNN/val/{model_name}", exist_ok=True)
    os.makedirs(f"/app/backend/FasterRCNN/saved_models", exist_ok=True)
    os.makedirs(f"/app/backend/FasterRCNN/test/{model_name}", exist_ok=True)

    logger.info("Ścieżki danych:")
    train_path = os.path.join(args.train_dir, "images")
    val_path = os.path.join(os.path.dirname(os.path.dirname(args.coco_gt_path)), "images")
    test_path = os.path.join(os.path.dirname(args.train_dir), "test", "images")
    logger.info(f"+ train_path: {train_path}")
    logger.info(f"+ train_annotations: {args.coco_train_path}")
    logger.info(f"+ val_path: {val_path}")
    logger.info(f"+ val_annotations: {args.coco_gt_path}")
    logger.info(f"+ test_path: {test_path}")

    logger.info("Wczytywanie danych...")
    train_loader, val_loader = get_data_loaders(
        batch_size=args.batch_size,  # Może być None, wtedy estymowane
        num_workers=args.num_workers,
        train_path=train_path,
        train_annotations=args.coco_train_path,
        val_path=val_path,
        val_annotations=args.coco_gt_path
    )

    model = get_model(num_classes=NUM_CLASSES, device=device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE, min_lr=MIN_LR)

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
        logger.info(f"Learning rate: {scheduler.get_last_lr()[0]:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_model_path = f"/app/backend/FasterRCNN/saved_models/{model_name}_checkpoint.pth"

            torch.save(model.state_dict(), best_model_path)
            logger.info(f"Zapisano nowy najlepszy model: {best_model_path}")

            for path in glob.glob(f"/app/backend/FasterRCNN/saved_models/{model_name}_*.pth"):
                if path != best_model_path:
                    try:
                        os.remove(path)
                        logger.info(f"Usunięto stary model: {path}")
                    except Exception as e:
                        logger.error(f"Błąd przy usuwaniu {path}: {e}")

    logger.info(f"Model końcowy zapisano jako: {best_model_path}")
    logger.info(f"Najlepszy model pochodzi z epoki {best_epoch} (val_loss = {best_val_loss:.4f})")

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

    logger.info("Wykresy zapisane w folderze test.")

if __name__ == "__main__":
    main()