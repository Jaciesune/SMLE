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
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Wyłączenie debugowania dla Pillow
logging.getLogger('PIL').setLevel(logging.WARNING)

# Wymuszenie UTF-8 z fallbackiem na błędy
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Stałe konfiguracyjne
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 2
MIN_LR = 1e-6
DEFAULT_EPOCHS = 20  # Domyślna liczba epok, jak w Mask R-CNN

def main():
    # Parsowanie argumentów
    parser = argparse.ArgumentParser(description="Trening Faster R-CNN z ResNet50", allow_abbrev=False)
    parser.add_argument("--num_workers", type=int, default=4, help="Liczba wątków w DataLoader")
    parser.add_argument("--batch_size", type=int, default=None, help="Rozmiar partii (None oznacza automatyczne dopasowanie)")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Liczba epok do wykonania")
    parser.add_argument("--model_name", type=str, help="Nazwa modelu (wymagane)")
    parser.add_argument("--lr", type=float, default=0.0005, help="Początkowa wartość learning rate")
    parser.add_argument("--train_dir", type=str, help="Ścieżka do danych treningowych")
    parser.add_argument("--coco_train_path", type=str, help="Ścieżka do pliku COCO z adnotacjami treningowymi")
    parser.add_argument("--coco_gt_path", type=str, help="Ścieżka do pliku COCO z adnotacjami walidacyjnymi")
    parser.add_argument("--val_dir", type=str, help="Ścieżka do danych walidacyjnych")
    parser.add_argument("--resume", type=str, default=None, help="Nazwa checkpointa do wczytania (bez ścieżki)")

    args, _ = parser.parse_known_args()

    # Ustalanie nazwy modelu
    if not args.model_name:
        raise ValueError("Nazwa modelu (--model_name) jest wymagana.")
    base_model_name = args.model_name

    # Ustalanie liczby epok
    epochs = args.epochs or DEFAULT_EPOCHS

    # Ustalanie urządzenia
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Używane urządzenie: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}")

    # Tworzenie katalogów w strukturze logs
    os.makedirs(f"/app/backend/FasterRCNN/logs/train/{base_model_name}", exist_ok=True)
    os.makedirs(f"/app/backend/FasterRCNN/logs/val/{base_model_name}", exist_ok=True)
    os.makedirs(f"/app/backend/FasterRCNN/saved_models", exist_ok=True)
    os.makedirs(f"/app/backend/FasterRCNN/logs/test/{base_model_name}", exist_ok=True)

    # Logowanie ścieżek danych
    logger.info("Ścieżki danych:")
    train_path = os.path.join(args.train_dir, "images")
    val_path = os.path.join(args.val_dir, "images") if args.val_dir else os.path.join(os.path.dirname(os.path.dirname(args.coco_gt_path)), "images")
    logger.info(f"+ train_path: {train_path}")
    logger.info(f"+ train_annotations: {args.coco_train_path}")
    logger.info(f"+ val_path: {val_path}")
    logger.info(f"+ val_annotations: {args.coco_gt_path}")

    # Wczytywanie danych
    logger.info("Wczytywanie danych...")
    train_loader, val_loader = get_data_loaders(
        batch_size=args.batch_size,  # Może być None, wtedy estymowane
        num_workers=args.num_workers,
        train_path=train_path,
        train_annotations=args.coco_train_path,
        val_path=val_path,
        val_annotations=args.coco_gt_path
    )

    # Inicjalizacja modelu, optymalizatora i scheduler'a
    model = get_model(num_classes=NUM_CLASSES, device=device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE, min_lr=MIN_LR)

    # Inicjalizacja zmiennych treningowych
    best_val_loss = float("inf")
    best_epoch = 0
    train_losses, val_losses, pred_counts, gt_counts = [], [], [], []
    map_5095_list, map_50_list, precision_list, recall_list = [], [], [], []
    start_epoch = 1
    last_epoch = start_epoch - 1  # Inicjalizacja last_epoch

    # Ścieżka do najlepszego checkpointa
    best_checkpoint_path = f"/app/backend/FasterRCNN/saved_models/{base_model_name}_bestepoch_checkpoint.pth"

    # Wczytywanie checkpointa, jeśli podano --resume
    resume_path = None
    if args.resume:
        resume_path = os.path.join(f"/app/backend/FasterRCNN/saved_models", args.resume)
        logger.info(f"Argument --resume: {args.resume}, pełna ścieżka: {resume_path}")
    else:
        logger.info("Brak argumentu --resume, wczytuję nowy model")

    if resume_path and os.path.exists(resume_path):
        try:
            checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                start_epoch = checkpoint['epoch'] + 1  # Kontynuacja od następnej epoki
                train_losses = checkpoint.get('train_losses', [])
                val_losses = checkpoint.get('val_losses', [])
                pred_counts = checkpoint.get('num_predictions', [])
                gt_counts = checkpoint.get('num_gt', [])
                map_5095_list = checkpoint.get('mAPs_5095', [])
                map_50_list = checkpoint.get('mAPs_50', [])
                precision_list = checkpoint.get('precisions', [])
                recall_list = checkpoint.get('recalls', [])
                best_val_loss = min(val_losses) if val_losses else float("inf")
                best_epoch = val_losses.index(best_val_loss) + 1 if val_losses else 0
                last_epoch = checkpoint['epoch']  # Ustawienie last_epoch na epokę z checkpointa
                # Resetowanie learning rate do wartości z argumentów
                for param_group in optimizer.param_groups:
                    param_group['lr'] = args.lr
                logger.info(f"Wczytano checkpoint z {resume_path}. Kontynuuję od epoki {start_epoch} z zresetowanym lr={args.lr}")
            else:
                logger.error(f"Plik {resume_path} nie jest checkpointem. Nie można wczytać jako checkpoint.")
                raise ValueError("Plik nie jest poprawnym checkpointem.")
        except Exception as e:
            logger.error(f"Błąd podczas wczytywania: {e}")
            logger.info("Rozpoczynam trening od nowa")
    else:
        logger.info("Wczytano nowy model")

    # Ustalanie końcowej epoki
    end_epoch = start_epoch + epochs - 1

    # Pętla treningowa
    if start_epoch <= end_epoch:
        for epoch in range(start_epoch, end_epoch + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)
            val_loss, pred_count, gt_count, map_5095, map_50, precision, recall = validate_model(
                model, val_loader, device, epoch, base_model_name
            )

            # Zapisywanie metryk
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            pred_counts.append(pred_count)
            gt_counts.append(gt_count)
            map_5095_list.append(map_5095)
            map_50_list.append(map_50)
            precision_list.append(precision)
            recall_list.append(recall)

            # Logowanie wyników epoki
            print(f"Epoka {epoch}/{end_epoch} - Strata treningowa: {train_loss:.4f}, Strata walidacyjna: {val_loss:.4f}")
            print(f"                - Detekcje: {pred_count} | GT: {gt_count}")
            print(f"                - mAP@0.5: {map_50:.4f} | mAP@0.5:0.95: {map_5095:.4f} | Prec: {precision:.4f} | Rec: {recall:.4f}")

            # Aktualizacja scheduler'a
            scheduler.step(val_loss)
            logger.info(f"Learning rate: {scheduler.get_last_lr()[0]:.6f}")

            # Zapisywanie najlepszego modelu
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'epoch': epoch,
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                    'num_predictions': pred_counts,
                    'num_gt': gt_counts,
                    'mAPs_5095': map_5095_list,
                    'mAPs_50': map_50_list,
                    'precisions': precision_list,
                    'recalls': recall_list
                }
                torch.save(checkpoint, best_checkpoint_path)
                logger.info(f"Zapisano nowy najlepszy checkpoint: {best_checkpoint_path}")

                # Usuwanie starych modeli
                for path in glob.glob(f"/app/backend/FasterRCNN/saved_models/{base_model_name}_*.pth"):
                    if path != best_checkpoint_path:
                        try:
                            os.remove(path)
                            logger.info(f"Usunięto stary model: {path}")
                        except Exception as e:
                            logger.error(f"Błąd przy usuwaniu {path}: {e}")
            
            # Aktualizacja last_epoch po każdej epoce
            last_epoch = epoch
    else:
        logger.info(f"start_epoch ({start_epoch}) jest większe lub równe końcowej epoce ({end_epoch + 1}). Trening nie zostanie wykonany.")

    # Zapisywanie końcowego checkpointa
    final_checkpoint_path = f"/app/backend/FasterRCNN/saved_models/{base_model_name}_checkpoint.pth"
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': last_epoch,  # Zapisujemy ostatnią wykonaną epokę
        'train_losses': train_losses,
        'val_losses': val_losses,
        'num_predictions': pred_counts,
        'num_gt': gt_counts,
        'mAPs_5095': map_5095_list,
        'mAPs_50': map_50_list,
        'precisions': precision_list,
        'recalls': recall_list
    }
    torch.save(checkpoint, final_checkpoint_path)
    logger.info(f"Checkpoint końcowy zapisano jako: {final_checkpoint_path}")
    logger.info(f"Najlepszy model pochodzi z epoki {best_epoch} (val_loss = {best_val_loss:.4f})")

    # Usuwanie tymczasowego checkpointa
    if os.path.exists(best_checkpoint_path):
        os.remove(best_checkpoint_path)
        logger.info(f"Usunięto tymczasowy plik: {best_checkpoint_path}")

    # Zapisywanie wykresów
    def save_plot(data1, data2, labels, title, filename, ylabel):
        plt.figure(figsize=(10, 5))
        plt.plot(data1, label=labels[0])
        plt.plot(data2, label=labels[1])
        plt.xlabel("Epoka")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.savefig(f"/app/backend/FasterRCNN/logs/test/{base_model_name}/{filename}")
        plt.close()

    save_plot(train_losses, val_losses, ["Strata treningowa", "Strata walidacyjna"], "Strata w czasie treningu", "loss_plot.png", "Strata")
    save_plot(pred_counts, gt_counts, ["Wykryte obiekty", "Obiekty GT"], "Porównanie predykcji i GT", "detections_plot.png", "Liczba obiektów")
    save_plot(map_50_list, map_5095_list, ["mAP@0.5", "mAP@0.5:0.95"], "Mean Average Precision", "map_plot.png", "mAP")

    logger.info("Wykresy zapisane w folderze logs/test.")

if __name__ == "__main__":
    main()