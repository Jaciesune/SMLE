import torch
import torchvision.models.detection
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import matplotlib.pyplot as plt
import os
import argparse
from datetime import datetime
from dataset import get_data_loaders
from utils import train_one_epoch, validate_model
import numpy as np
import shutil
import sys
from tqdm import tqdm

# === KONFIGURACJA ===
CONFIDENCE_THRESHOLD = 0.7
NMS_THRESHOLD = 0.5
DETECTION_PER_IMAGE = 500
NUM_CLASSES = 2

BASE_DIR = "/app/backend/Mask_RCNN"
PRETRAINED_WEIGHTS_DIR = os.path.join(BASE_DIR, "pretrained_weights")
PRETRAINED_WEIGHTS_PATH = os.path.join(PRETRAINED_WEIGHTS_DIR, "maskrcnn_resnet50_fpn_v2.pth")
MODELS_DIR = os.path.join(BASE_DIR, "models")
LOGS_TRAIN_DIR = os.path.join(BASE_DIR, "logs/train")
LOGS_VAL_DIR = os.path.join(BASE_DIR, "logs/val")
DEFAULT_TRAIN_DIR = "/app/train_data"
DEFAULT_VAL_DIR = "/app/backend/data/val"

DEFAULT_NUM_WORKERS = 1
DEFAULT_BATCH_SIZE = 1
DEFAULT_EPOCHS = 20
DEFAULT_LR = 0.0005
DEFAULT_PATIENCE = 8
DEFAULT_NUM_AUGMENTATIONS = 8

MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 2
MIN_LR = 1e-6

# === FUNKCJE POMOCNICZE ===
def parse_args():
    parser = argparse.ArgumentParser(description="Trening Mask R-CNN (v2)")
    parser.add_argument("--train_dir", type=str, default=DEFAULT_TRAIN_DIR, help="Ścieżka do danych treningowych")
    parser.add_argument("--val_dir", type=str, default=DEFAULT_VAL_DIR, help="Ścieżka do danych walidacyjnych")  # Dodajemy --val_dir
    parser.add_argument("--num_workers", type=int, default=DEFAULT_NUM_WORKERS, help="Liczba wątków dla DataLoadera")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="Rozmiar batcha")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Liczba epok do wykonania")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR, help="Początkowa wartość learning rate")
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE, help="Liczba epok bez poprawy dla Early Stopping")
    parser.add_argument("--coco_train_path", type=str, default=None, help="Ścieżka do pliku COCO z adnotacjami treningowymi")
    parser.add_argument("--coco_gt_path", type=str, default=None, help="Ścieżka do pliku COCO z adnotacjami walidacyjnymi")
    parser.add_argument("--num_augmentations", type=int, default=DEFAULT_NUM_AUGMENTATIONS, help="Liczba augmentacji na obraz")
    parser.add_argument("--resume", type=str, default=None, help="Nazwa modelu do wczytania (bez ścieżki)")
    parser.add_argument("--model_name", type=str, default=None, help="Nazwa modelu (wymagane przy wywołaniu przez API)")
    return parser.parse_args()

def download_pretrained_weights():
    print(f"Pobieranie pretrained weights do {PRETRAINED_WEIGHTS_PATH}...", flush=True)
    os.makedirs(PRETRAINED_WEIGHTS_DIR, exist_ok=True)
    weights = torchvision.models.detection.MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights=weights)
    torch.save(model.state_dict(), PRETRAINED_WEIGHTS_PATH)
    print(f"Pretrained weights zapisano jako: {PRETRAINED_WEIGHTS_PATH}", flush=True)

def get_model(num_classes, device):
    if not os.path.exists(PRETRAINED_WEIGHTS_PATH):
        print(f"Pretrained weights nie istnieją w {PRETRAINED_WEIGHTS_PATH}. Pobieranie...", flush=True)
        download_pretrained_weights()
    else:
        print(f"Używam istniejących pretrained weights z {PRETRAINED_WEIGHTS_PATH}", flush=True)

    model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights=None)
    state_dict = torch.load(PRETRAINED_WEIGHTS_PATH, map_location=device)
    model.load_state_dict(state_dict)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(
        in_channels=256, dim_reduced=256, num_classes=num_classes
    )

    if isinstance(model.rpn.pre_nms_top_n, dict):
        model.rpn.pre_nms_top_n["training"] = NMS_THRESHOLD
        model.rpn.pre_nms_top_n["testing"] = NMS_THRESHOLD
        model.rpn.post_nms_top_n["training"] = NMS_THRESHOLD
        model.rpn.post_nms_top_n["testing"] = NMS_THRESHOLD
    model.roi_heads.score_thresh = CONFIDENCE_THRESHOLD
    model.roi_heads.detections_per_img = DETECTION_PER_IMAGE
    model.to(device)
    print(f"Model działa na: {device}", flush=True)
    return model

def train_model(args, is_api_call=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Czy CUDA jest dostępna: {torch.cuda.is_available()}", flush=True)
    try:
        if torch.cuda.is_available():
            print(f"Nazwa GPU: {torch.cuda.get_device_name(0)}", flush=True)
            print(f"Liczba dostępnych GPU: {torch.cuda.device_count()}", flush=True)
        shm_path = "/dev/shm"
        shm_usage = shutil.disk_usage(shm_path)
        print(f"Pamięć współdzielona (/dev/shm):", flush=True)
        print(f"Całkowita: {shm_usage.total / (1024**3):.2f} GB", flush=True)
        print(f"Użyta: {shm_usage.used / (1024**3):.2f} GB", flush=True)
        print(f"Wolna: {shm_usage.free / (1024**3):.2f} GB", flush=True)
    except Exception as e:
        print(f"Błąd po sprawdzeniu CUDA: {e}", flush=True)
        raise

    # Ustalanie nazwy modelu bez timestampu
    if is_api_call:
        if not args.model_name:
            raise ValueError("Nazwa modelu (--model_name) jest wymagana przy wywołaniu przez API.")
        base_model_name = args.model_name
    else:
        if args.model_name:
            base_model_name = args.model_name
        else:
            default_model_name = "train_3"
            try:
                nazwa_modelu_input = input(f"Podaj nazwę modelu (Enter dla domyślnej: {default_model_name}): ").strip()
                base_model_name = nazwa_modelu_input if nazwa_modelu_input else default_model_name
            except EOFError:
                print("Brak interaktywnego wejścia, używam domyślnej nazwy modelu.", flush=True)
                base_model_name = default_model_name

    os.makedirs(os.path.join(LOGS_TRAIN_DIR, base_model_name), exist_ok=True)
    os.makedirs(os.path.join(LOGS_VAL_DIR, base_model_name), exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    train_dir = args.train_dir
    val_dir = args.val_dir  # Używamy podanego --val_dir zamiast DEFAULT_VAL_DIR

    coco_train_path = args.coco_train_path if args.coco_train_path else os.path.join(train_dir, "annotations", "instances_train.json")
    coco_val_path = args.coco_gt_path if args.coco_gt_path else os.path.join(val_dir, "annotations", "instances_val.json")

    if not os.path.exists(coco_train_path):
        raise FileNotFoundError(f"Plik adnotacji treningowych nie istnieje: {coco_train_path}")
    if not os.path.exists(coco_val_path):
        raise FileNotFoundError(f"Plik adnotacji walidacyjnych nie istnieje: {coco_val_path}")

    print("\nWczytywanie danych...", flush=True)
    train_loader, val_loader = get_data_loaders(
        train_dir=train_dir,
        val_dir=val_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_augmentations=args.num_augmentations,
        coco_train_path=coco_train_path,
        coco_val_path=coco_val_path
    )

    straty_treningowe = []
    straty_walidacyjne = []
    liczby_predykcji = []
    liczby_gt = []
    mAPs_bbox = []
    mAPs_seg = []
    start_epoch = 1
    last_epoch = start_epoch - 1

    best_checkpoint_path = os.path.join(MODELS_DIR, f"{base_model_name}_bestepoch_checkpoint.pth")

    model = get_model(num_classes=NUM_CLASSES, device=device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE, min_lr=MIN_LR)

    # Wczytywanie modelu na podstawie nazwy
    resume_path = None
    if args.resume:
        resume_path = os.path.join(MODELS_DIR, args.resume)
        print(f"Argument --resume: {args.resume}, pełna ścieżka: {resume_path}", flush=True)
    else:
        print("Brak argumentu --resume, wczytuję nowy model", flush=True)

    if resume_path and os.path.exists(resume_path):
        try:
            checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                start_epoch = checkpoint['epoch'] + 1
                straty_treningowe = checkpoint.get('train_losses', [])
                straty_walidacyjne = checkpoint.get('val_losses', [])
                liczby_predykcji = checkpoint.get('num_predictions', [])
                liczby_gt = checkpoint.get('num_gt', [])
                mAPs_bbox = checkpoint.get('mAPs_bbox', [])
                mAPs_seg = checkpoint.get('mAPs_seg', [])
                last_epoch = checkpoint['epoch']
                for param_group in optimizer.param_groups:
                    param_group['lr'] = args.lr
                print(f"Wczytano checkpoint z {resume_path}. Kontynuuję od epoki {start_epoch} z zresetowanym lr={args.lr}", flush=True)
            else:
                print(f"Plik {resume_path} nie jest checkpointem. Nie można wczytać jako checkpoint.", flush=True)
                raise ValueError("Plik nie jest poprawnym checkpointem.")
        except Exception as e:
            print(f"Błąd podczas wczytywania: {e}", flush=True)
            print("Rozpoczynam trening od nowa", flush=True)
    else:
        print(f"Wczytano nowy model", flush=True)

    end_epoch = start_epoch + args.epochs - 1

    najlepsza_strata_walidacyjna = float("inf")
    licznik_cierpliwości = 0
    cierpliwość = args.patience

    if start_epoch <= end_epoch:
        with tqdm(total=end_epoch - start_epoch + 1, desc="Trening", unit="epoka", file=sys.stdout) as pbar:
            for epoch in range(start_epoch, end_epoch + 1):
                strata_treningowa = train_one_epoch(model, train_loader, optimizer, device, epoch)
                strata_walidacyjna, liczba_predykcji, liczba_gt, mAP_bbox, mAP_seg = validate_model(
                    model, val_loader, device, epoch, base_model_name, coco_val_path
                )

                straty_treningowe.append(strata_treningowa)
                straty_walidacyjne.append(strata_walidacyjna)
                liczby_predykcji.append(liczba_predykcji)
                liczby_gt.append(liczba_gt)
                mAPs_bbox.append(mAP_bbox)
                mAPs_seg.append(mAP_seg)

                log_message = (
                    f"Epoka {epoch}/{end_epoch} - Strata treningowa: {strata_treningowa:.4f}, "
                    f"Strata walidacyjna: {strata_walidacyjna:.4f}, Pred: {liczba_predykcji}, "
                    f"GT: {liczba_gt}, Ratio: {liczba_predykcji/liczba_gt:.2f}, "
                    f"mAP_bbox: {mAP_bbox:.4f}, mAP_seg: {mAP_seg:.4f}"
                )
                pbar.set_description(log_message)
                pbar.update(1)

                if strata_walidacyjna < najlepsza_strata_walidacyjna:
                    najlepsza_strata_walidacyjna = strata_walidacyjna
                    licznik_cierpliwości = 0
                    checkpoint = {
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'epoch': epoch,
                        'train_losses': straty_treningowe,
                        'val_losses': straty_walidacyjne,
                        'num_predictions': liczby_predykcji,
                        'num_gt': liczby_gt,
                        'mAPs_bbox': mAPs_bbox,
                        'mAPs_seg': mAPs_seg
                    }
                    torch.save(checkpoint, best_checkpoint_path)
                    print(f"Zapisano najlepszy checkpoint: {best_checkpoint_path}", flush=True)
                else:
                    licznik_cierpliwości += 1
                    print(f"Brak poprawy przez {licznik_cierpliwości}/{cierpliwość} epok", flush=True)
                    if licznik_cierpliwości >= cierpliwość:
                        print("Early Stopping: Zakończono trening przedwcześnie.", flush=True)
                        break

                scheduler.step(strata_walidacyjna)
                print(f"Learning rate: {scheduler.get_last_lr()[0]}", flush=True)
                last_epoch = epoch
    else:
        print(f"start_epoch ({start_epoch}) jest większe lub równe końcowej epoce ({end_epoch + 1}). Trening nie zostanie wykonany.", flush=True)

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': last_epoch,
        'train_losses': straty_treningowe,
        'val_losses': straty_walidacyjne,
        'num_predictions': liczby_predykcji,
        'num_gt': liczby_gt,
        'mAPs_bbox': mAPs_bbox,
        'mAPs_seg': mAPs_seg
    }
    nazwa_pliku_checkpointu = os.path.join(MODELS_DIR, f"{base_model_name}_checkpoint.pth")
    torch.save(checkpoint, nazwa_pliku_checkpointu)
    print(f"Checkpoint zapisano jako: {nazwa_pliku_checkpointu}", flush=True)

    if os.path.exists(best_checkpoint_path):
        os.remove(best_checkpoint_path)
        print(f"Usunięto tymczasowy plik: {best_checkpoint_path}", flush=True)

    plt.figure(figsize=(10, 5))
    plt.plot(straty_treningowe, label="Strata treningowa")
    plt.plot(straty_walidacyjne, label="Strata walidacyjna")
    plt.xlabel("Epoka")
    plt.ylabel("Strata")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(LOGS_TRAIN_DIR, base_model_name, "loss_plot.png"))

    plt.figure(figsize=(10, 5))
    plt.plot(liczby_predykcji, label="Wykryte obiekty")
    plt.plot(liczby_gt, label="Obiekty GT")
    plt.xlabel("Epoka")
    plt.ylabel("Liczba obiektów")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(LOGS_TRAIN_DIR, base_model_name, "detections_plot.png"))

    plt.figure(figsize=(10, 5))
    plt.plot(mAPs_bbox, label="mAP (bbox)")
    plt.plot(mAPs_seg, label="mAP (segmentacja)")
    plt.xlabel("Epoka")
    plt.ylabel("mAP")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(LOGS_TRAIN_DIR, base_model_name, "mAP_plot.png"))

    print("Trening zakończony!", flush=True)
    return f"Trening zakończony! Checkpoint zapisany jako: {nazwa_pliku_checkpointu}"

if __name__ == "__main__":
    args = parse_args()
    train_model(args, is_api_call=False)